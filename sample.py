"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import os
from slider import Beatmap
from slider.beatmap import Slider
from slider.curve import Perfect, Catmull, Linear

from export.create_beatmap import create_beatmap
from export.slider_path import SliderPath
from positional_embedding import timestep_embedding
from diffusion import create_diffusion
from models import DiT_models
from data_loading import context_size, beatmap_to_sequence, get_beatmap_idx


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
        seq_no_embed = seq_no_embed[:, (seq_no_embed[2] > args.plot_time - args.plot_width) & (seq_no_embed[2] < args.plot_time + args.plot_width)]
        print(f"Sequence trimmed to length {seq_no_embed.shape[1]}")

    seq_len = seq_no_embed.shape[1]
    seq_x = seq_no_embed[:2, :]
    seq_y = torch.concatenate(
        [
            timestep_embedding(seq_no_embed[2, :], 128, 36000).T,
            seq_no_embed[4:, :]
        ], 0)

    # Load model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=142
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model with (feel free to change):
    if args.style_id is not None:
        beatmap_idx = get_beatmap_idx()
        idx = beatmap_idx[args.style_id]
        class_labels = [idx]
    else:
        # Use null class
        class_labels = [args.num_classes]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 2, seq_len, device=device)
    c = seq_y.repeat(n, 1, 1).to(device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    c = torch.cat([c, c], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(c=c, y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    sampled_seq = None
    if args.plot_time is not None and False:
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
        new_beatmap = create_beatmap(sampled_seq[i], beatmap, f"Diffusion {args.style_id} {i}")
        new_beatmap.write_path(os.path.join(result_dir, f"result{i}.osu"))

        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.set_xlim([0, 512])
        ax.set_ylim([384, 0])
        plt.cla()
        plot_beatmap(ax, new_beatmap, args.plot_time, args.plot_width)
        plt.show()


def plot_beatmap(ax: plt.Axes, beatmap: Beatmap, time, window_size):
    width = beatmap.cs() * 2
    hit_objects = beatmap.hit_objects(spinners=False)
    min_time, max_time = timedelta(seconds=(time - window_size) / 1000), timedelta(seconds=(time + window_size) / 1000)
    windowed = [ho for ho in hit_objects if min_time < ho.time < max_time]
    artists = []
    for ho in windowed:
        if isinstance(ho, Slider):
            slider_path = SliderPath("PerfectCurve" if isinstance(ho.curve, Perfect) else ("CatmulL" if isinstance(ho.curve, Catmull) else ("Linear" if isinstance(ho.curve, Linear) else "Bezier")),
                              np.array(ho.curve.points, dtype=float),
                              ho.curve.req_length)
            path = []
            slider_path.get_path_to_progress(path, 0, 1)
            p = np.vstack(path)
            artists.append(ax.plot(p[:, 0], p[:, 1], color="green", linewidth=width, solid_capstyle="round", solid_joinstyle="round")[0])
    p = np.array([ho.position for ho in hit_objects])
    artists.append(ax.scatter(p[:, 0], p[:, 1], s=width**2, c="Lime"))
    return artists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beatmap", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B")
    parser.add_argument("--num-classes", type=int, default=41189)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--use-amp", type=bool, default=True)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--plot-time", type=float, default=None)
    parser.add_argument("--plot-width", type=float, default=1000)
    args = parser.parse_args()
    main(args)