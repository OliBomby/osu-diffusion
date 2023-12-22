"""
Sample new images from a pre-trained DiT.
"""
import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from data_loading import beatmap_to_sequence, CachedDataset, playfield_size, split_and_process_sequence_no_augment
from data_loading import feature_size
from diffusion import create_diffusion
from models import DiT_models
from slider import Beatmap

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def find_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f"Could not find DiT checkpoint at {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def generate_predictions(model, diffusion, device, seq_no_embed, args, progress=False):
    (seq_x, seq_o, seq_c), seq_len = split_and_process_sequence_no_augment(seq_no_embed.squeeze(0))
    seq_o = seq_o - seq_o[0]  # Normalize to relative time

    # Use null class
    class_labels = [args.num_classes for _ in range(args.num_predictions)]

    n = len(class_labels)
    z = torch.randn(n, 2, seq_len, device=device)
    x = seq_x.repeat(n, 1, 1).to(device)
    o = seq_o.repeat(n, 1).to(device)
    c = seq_c.repeat(n, 1, 1).to(device)
    y = torch.tensor(class_labels, device=device)

    model_kwargs = dict(o=o, c=c, y=y)

    # Make in-paint mask
    mask = torch.full_like(z, False, dtype=torch.bool, device=device)
    mask[:, :, -1] = True

    def in_paint_mask(x2):
        return torch.where(mask, x2, x)

    z = in_paint_mask(z)

    samples = diffusion.p_sample_loop(
        model.forward,
        z.shape,
        z,
        denoised_fn=in_paint_mask,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        progress=progress,
        device=device,
    )

    return samples[:, :, -1] * playfield_size.to(device).unsqueeze(0)


def example_from_beatmap(beatmap):
    seq = beatmap_to_sequence(beatmap)
    hit_objects = beatmap.hit_objects(spinners=False)
    posterior = hit_objects[-1]
    label = torch.tensor(posterior.position)

    # Trim the last slider body steps in the sequence
    type_index = torch.argmax(seq[3:], 0)
    bad_steps = type_index > 5
    num_bad = 0
    # noinspection PyTypeChecker
    for i in torch.flip(bad_steps, [0]):
        if i:
            num_bad += 1
        else:
            break

    if num_bad > 0:
        seq = seq[:, :-num_bad]

    assert (seq[:2, -1] == label).all()

    return seq, label


def load_example_folder(name):
    data = []
    for filename in os.listdir(os.path.join("testing", "toy_datasets", name)):
        path = os.path.join("testing", "toy_datasets", name, filename)
        beatmap = Beatmap.from_path(path)
        example = example_from_beatmap(beatmap)
        data.append(example)
    return CachedDataset(data)


def get_dataloader(dataset):
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        batch_size=1,
    )

    return dataloader


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=feature_size - 3 + 128,
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        noise_schedule="squaredcos_cap_v2",
    )

    if args.generate is not None:
        generate_path = os.path.join("testing", "toy_datasets", args.generate)
        beatmap = Beatmap.from_path(generate_path)
        (seq, pos) = example_from_beatmap(beatmap)
        seq, pos = seq.to(device), pos.to(device)
        predictions = generate_predictions(model, diffusion, device, seq, args, progress=True)
        distances = torch.norm(predictions - pos, dim=1)
        # noinspection PyTypeChecker
        good_count = torch.sum(distances < 30).item()
        print(f"Generate example correct predictions = {good_count / len(predictions) * 100}% ({good_count}/{len(predictions)})")
        for p in predictions.cpu():
            print(f"{round(p[0].item())},{round(p[1].item())},{round(seq[2, -1].item())},1,0,0:0:0:0:")
        return


    for test in args.tests:
        print(test)

        test_dataloader = get_dataloader(load_example_folder(test))

        num_predictions = 0
        num_good_predictions = 0

        for seq, pos in tqdm.tqdm(test_dataloader):
            pos = pos.to(device)

            predictions = generate_predictions(model, diffusion, device, seq, args)

            distances = torch.norm(predictions - pos, dim=1)
            # noinspection PyTypeChecker
            good_count = torch.sum(distances < 30).item()

            num_predictions += len(predictions)
            num_good_predictions += good_count

        print(f"Correct predictions = {num_good_predictions / num_predictions * 100}% ({num_good_predictions}/{num_predictions})")


datasets = [
    "geometry",
    "stream",
    "symmetry",
    "visual_spacing",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-B",
    )
    parser.add_argument("--num-classes", type=int, default=52670)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-predictions", type=int, default=100)
    parser.add_argument("--tests", type=list[str], default=datasets)
    parser.add_argument("--generate", type=str, default=None)
    args = parser.parse_args()
    main(args)
