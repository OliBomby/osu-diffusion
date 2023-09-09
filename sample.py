"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import os
from slider import Beatmap

from export.create_beatmap import create_beatmap, plot_beatmap
from diffusion import create_diffusion
from models import DiT_models
from data_loading import context_size, beatmap_to_sequence, get_beatmap_idx, split_and_process_sequence


def find_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f'Could not find DiT checkpoint at {ckpt_path}'
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load beatmap to sample coordinates for
    beatmap = Beatmap.from_path(args.beatmap)

    result_dir = os.path.join("results", str(beatmap.beatmap_id))
    os.makedirs(result_dir, exist_ok=True)

    seq_no_embed = beatmap_to_sequence(beatmap)

    if args.plot_time is not None:
        # noinspection PyTypeChecker
        start_index = torch.nonzero(seq_no_embed[2] >= args.plot_time)[0]
        seq_no_embed = seq_no_embed[:, start_index:start_index + args.seq_len]
        print(f"Sequence trimmed to length {seq_no_embed.shape[1]}")

    seq_len = seq_no_embed.shape[1]
    print(f"seq len {seq_len}")
    seq_x, seq_o, seq_c = split_and_process_sequence(seq_no_embed)
    seq_o = seq_o - seq_o[0]  # Normalize to relative time

    # Load model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=context_size
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Create banded matrix attention mask for increased sequence length
    attn_mask = torch.full((seq_len, seq_len), True, dtype=torch.bool, device=device)
    for i in range(seq_len):
        attn_mask[max(0, i - args.seq_len):min(seq_len, i + args.seq_len), i] = False

    # Labels to condition the model with (feel free to change):
    if args.style_id is not None:
        beatmap_idx = get_beatmap_idx()
        idx = beatmap_idx[args.style_id]
        class_labels = [idx + i for i in range(args.num_variants)]
    else:
        # Use null class
        class_labels = [args.num_classes]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 2, seq_len, device=device)
    o = seq_o.repeat(n, 1).to(device)
    c = seq_c.repeat(n, 1, 1).to(device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    o = torch.cat([o, o], 0)
    c = torch.cat([c, c], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(o=o, c=c, y=y, cfg_scale=args.cfg_scale, attn_mask=attn_mask)

    # Sample images:
    sampled_seq = None
    if args.plot_time is not None and args.make_animation:
        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.set_xlim([0, 512])
        ax.set_ylim([384, 0])
        artists = []

        for samples in diffusion.p_sample_loop_progressive(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device):
            samples, _ = samples["sample"].chunk(2, dim=0)  # Remove null class samples
            sampled_seq = torch.concatenate([samples.cpu(), seq_no_embed[2:].repeat(n, 1, 1)], 1)
            new_beatmap = create_beatmap(sampled_seq[0], beatmap, f"Diffusion {args.style_id}")
            artists.append(plot_beatmap(ax, new_beatmap, args.plot_time, args.plot_width))

        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000 // 24)
        ani.save(filename=os.path.join(result_dir, "animation.gif"), writer="pillow")
    else:
        samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        sampled_seq = torch.concatenate([samples.cpu(), seq_no_embed[2:].repeat(n, 1, 1)], 1)

    # Save beatmaps:
    for i in range(n):
        try:
            new_beatmap = create_beatmap(sampled_seq[i], beatmap, f"Diffusion {args.style_id} {i}")
            new_beatmap.write_path(os.path.join(result_dir, f"{beatmap.beatmap_id} result {args.style_id} {i}.osu"))

            if args.plot_time is not None:
                fig, ax = plt.subplots()
                plot_beatmap(ax, new_beatmap, args.plot_time, args.plot_width)
                ax.axis('equal')
                ax.set_xlim([0, 512])
                ax.set_ylim([384, 0])
                plt.show()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beatmap", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B")
    parser.add_argument("--num-classes", type=int, default=52670)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--use-amp", type=bool, default=True)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--plot-time", type=float, default=None)
    parser.add_argument("--plot-width", type=float, default=2000)
    parser.add_argument("--num-variants", type=int, default=1)
    parser.add_argument("--make-animation", type=bool, default=False)
    args = parser.parse_args()
    # for style_id in [2592760, 1451282, 1995061, 3697057, 2799753, 1772923, 1907310]:
    #     args.style_id = style_id
    #     main(args)
    main(args)