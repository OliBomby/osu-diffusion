"""
Sample new images from a pre-trained DiT.
"""
import numpy as np
import torch
from matplotlib import animation
from slider import Beatmap
from slider.beatmap import Slider
import matplotlib.pyplot as plt
from slider.curve import Perfect, Catmull, Linear

from export.create_beatmap import create_beatmap
from export.slider_path import SliderPath
from positional_embedding import timestep_embedding

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from models import DiT_models
import argparse
import os

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

    seq_no_embed = beatmap_to_sequence(beatmap)
    seq_len = seq_no_embed.shape[1]
    seq_x = seq_no_embed[:2, :]
    seq_y = torch.concatenate(
        [
            timestep_embedding(seq_no_embed[2, :], 128, 36000).T,
            seq_no_embed[3:, :]
        ], 0)

    # Load model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=context_size
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model with (feel free to change):
    if args.style_id is not None:
        beatmap_idx = get_beatmap_idx()
        class_labels = [beatmap_idx[args.style_id]]
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
    if args.plot_time is not None:
        fig, ax = plt.subplots()
        artists = []

        for samples in diffusion.p_sample_loop_progressive(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device):
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            sampled_seq = torch.concatenate([samples, seq_no_embed[2:].repeat(n, 1, 1)])
            new_beatmap = create_beatmap(sampled_seq[0], beatmap, f"Diffusion {args.style_id}")
            artists.append(plot_beatmap(ax, new_beatmap, args.plot_time))

        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
        ani.save(filename=os.path.join("results", args.beatmap, "animation.gif"), writer="pillow")
    else:
        samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        sampled_seq = torch.concatenate([samples, seq_no_embed[2:].repeat(n, 1, 1)])

    # Save beatmaps:
    for i in range(n):
        new_beatmap = create_beatmap(sampled_seq[i], beatmap, f"Diffusion {args.style_id} {i}")
        new_beatmap.write_path(os.path.join("results", args.beatmap, f"result{i}.osu"))


def plot_beatmap(ax: plt.Axes, beatmap: Beatmap, time):
    hit_objects = beatmap.hit_objects(spinners=False)
    windowed = [ho for ho in hit_objects if time - 1000 < ho.time < time + 1000]
    artists = []
    for ho in windowed:
        if isinstance(ho, Slider):
            path = SliderPath("PerfectCurve" if ho.curve is Perfect else ("CatmulL" if ho.curve is Catmull else ("Linear" if ho.curve is Linear else "Bezier")), np.array(ho.curve.points))
            p = np.vstack(path.calculatedPath)
            artists.append(ax.plot(p[:, 0], p[:, 1], color="green"))
    p = np.array([ho.position for ho in hit_objects])
    artists.append(ax.scatter(p[0, :], p[1, :], beatmap.cs()))
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
    args = parser.parse_args()
    main(args)