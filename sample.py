"""
Sample new images from a pre-trained DiT.
"""
from datetime import timedelta

import numpy as np
import torch
from slider import Beatmap, Position
from slider.beatmap import Circle, Slider, Spinner, Curve
from slider.curve import MultiBezier, Perfect, Catmull

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
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    for i in range(n):
        objects = []
        curr_object = None
        for j in range(seq_len):
            time = timedelta(seconds=float(seq_no_embed[2, j] / 1000))
            type_index = int(torch.argmax(seq_no_embed[3:, j]))
            x = int(round(float(samples[i, 0, j] * 512)))
            y = int(round(float(samples[i, 1, j] * 384)))
            pos = Position(x, y)
            if type_index == 0:
                objects.append(Circle(pos, time, 0))
            elif type_index == 1:
                curr_object = Spinner(pos, time, 0, time)
            elif type_index == 2 and isinstance(curr_object, Spinner):
                curr_object.end_time = time
                objects.append(curr_object)
            elif type_index == 3:
                curr_object = Slider(pos, time, time, 0, MultiBezier([pos], 0), 0, 0, 0, 0, 0, 0, [], [])
            elif type_index == 4 and isinstance(curr_object, Slider):
                curr_object.curve.points.append(pos)
            elif type_index == 5 and isinstance(curr_object, Slider):
                prev_points = curr_object.curve.points
                curr_object.curve = Perfect([Position(0, 0), Position(1, 0), Position(0, 1)], 0)
                curr_object.curve.points = prev_points + [pos]
            elif type_index == 6 and isinstance(curr_object, Slider):
                curr_object.curve = Catmull(curr_object.curve.points, 0)
                curr_object.curve.points.append(pos)
            elif type_index == 7 and isinstance(curr_object, Slider):
                curr_object.curve.points.append(pos)
                curr_object.curve.points.append(pos)
            elif type_index == 8 and isinstance(curr_object, Slider):
                curr_object.curve.points.append(pos)
            elif type_index >= 9 and isinstance(curr_object, Slider):
                curr_object.end_time = time
                curr_object.repeat = type_index - 8
                curr_object.length = np.sqrt((curr_object.position.x - x)**2 + (curr_object.position.y - y)**2)
                curr_object.edge_sounds = [0] * curr_object.repeat
                curr_object.edge_additions = ['0:0'] * curr_object.repeat
                objects.append(curr_object)

        for ho in objects:
            print(ho.pack())



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
    args = parser.parse_args()
    main(args)