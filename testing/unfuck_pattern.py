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


filler_seq = beatmap_to_sequence(Beatmap.from_path(os.path.join("testing", "toy_datasets", "kimi_no_bouken.osu")))


def find_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f"Could not find DiT checkpoint at {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def example_from_beatmap(beatmap, args):
    seq = beatmap_to_sequence(beatmap)
    seq_len = seq.shape[1]

    # Fix length to seq_len
    if args.seq_len is not None:
        if args.seq_len > seq_len:
            to_add = args.seq_len - seq_len
            filler_add = filler_seq[:, -to_add:]
            # Fix the timing offset
            seq[2] += filler_add[2, -1] - seq[2, 0] + 300
            seq = torch.concatenate([filler_add, seq], dim=1)
        elif args.seq_len < seq_len:
            seq = seq[:, -args.seq_len:]

    return seq, seq.shape[1] - seq_len, seq.shape[1]


def load_example_folder(name, args):
    data = []
    for filename in os.listdir(os.path.join("testing", "toy_datasets", name)):
        path = os.path.join("testing", "toy_datasets", name, filename)
        beatmap = Beatmap.from_path(path)
        example = example_from_beatmap(beatmap, args)
        data.append(example)
    return CachedDataset(data)


def get_dataloader(dataset):
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        batch_size=1,
    )

    return dataloader


def fuckup_pattern(seq, start, end, magnitude):
    seq_fucked = torch.clone(seq)
    noise = torch.randn_like(seq[:, :2, start:end]) * magnitude
    seq_fucked[:, :2, start:end] += noise
    return seq_fucked


def unfuck_pattern(model, diffusion, device, seq_no_embed, seq_no_embed_fucked, start, end, args, progress=False):
    (_, seq_o, seq_c), seq_len = split_and_process_sequence_no_augment(seq_no_embed_fucked.squeeze(0))
    (seq_x, _, _), _ = split_and_process_sequence_no_augment(seq_no_embed_fucked.squeeze(0))
    seq_o = seq_o - seq_o[0]  # Normalize to relative time

    # Create banded matrix attention mask for increased sequence length
    attn_mask = None
    max_seq_len = 128
    if seq_len > max_seq_len:
        attn_mask = torch.full((seq_len, seq_len), True, dtype=torch.bool, device=device)
        for i in range(seq_len):
            attn_mask[max(0, i - max_seq_len) : min(seq_len, i + max_seq_len), i] = False

    # Use null class
    class_labels = [args.num_classes for _ in range(args.num_predictions)]

    n = len(class_labels)
    x = seq_x.repeat(n, 1, 1).to(device)
    z = torch.clone(x)
    o = seq_o.repeat(n, 1).to(device)
    c = seq_c.repeat(n, 1, 1).to(device)
    y = torch.tensor(class_labels, device=device)

    model_kwargs = dict(o=o, c=c, y=y, attn_mask=attn_mask)

    # Make in-paint mask
    mask = torch.full_like(z, False, dtype=torch.bool, device=device)
    mask[:, :, start:end] = True

    def in_paint_mask(x2):
        return torch.where(mask, x2, x)

    img = z
    indices = [0] * args.num_sampling_steps

    if progress:
        indices = tqdm.tqdm(indices)

    for i in indices:
        t = torch.tensor([i] * z.shape[0], device=device)
        with torch.no_grad():
            out = diffusion.p_sample(
                model,
                img,
                t,
                clip_denoised=True,
                denoised_fn=in_paint_mask,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]

    result = seq_no_embed.repeat(n, 1, 1)
    result[:, :2, start:end] = img[:, :, start:end] * playfield_size.to(device).unsqueeze(0).unsqueeze(2)

    return result


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
        None,
        noise_schedule=args.noise_schedule,
    )

    for test in args.tests:
        print(test)

        test_dataloader = get_dataloader(load_example_folder(test, args))
        
        num_predictions = 0
        total_distance = 0
        total_distance2 = 0

        for seq, start, end in tqdm.tqdm(test_dataloader):
            seq_pos = seq[:, :2, start:end].repeat(args.num_predictions, 1, 1)

            seq_fucked = fuckup_pattern(seq, start, end, args.fucking_magnitude)
            fucked_pos = seq_fucked[:, :2, start:end].repeat(args.num_predictions, 1, 1)

            predictions = unfuck_pattern(model, diffusion, device, seq, seq_fucked, start, end, args)
            pred_pos = predictions[:, :2, start:end]

            distances = torch.norm(fucked_pos - seq_pos, dim=1)
            distances2 = torch.norm(pred_pos - seq_pos, dim=1)
            # noinspection PyTypeChecker
            distances_sum = torch.sum(distances).item()
            distances_sum2 = torch.sum(distances2).item()

            num_predictions += len(predictions) * (end - start).item()
            total_distance += distances_sum
            total_distance2 += distances_sum2

        print(f"Mean fucked distance = {total_distance / num_predictions} units (out of {num_predictions})")
        print(f"Mean unfucked distance = {total_distance2 / num_predictions} units (out of {num_predictions})")


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
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--noise-schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--fucking-magnitude", type=float, default=5)
    args = parser.parse_args()
    main(args)
