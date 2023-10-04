import argparse
import glob

import matplotlib.pyplot as plt
import torch

from data_loading import beatmap_to_sequence, calc_distances
from slider import Beatmap


def main(args):
    ref_beatmap = Beatmap.from_path(args.ref_beatmap)
    ref_seq = beatmap_to_sequence(ref_beatmap)
    ref_seq_d = calc_distances(ref_seq)

    beatmap = Beatmap.from_path(args.beatmap)
    name = beatmap.version if args.name is None else args.name
    seq = beatmap_to_sequence(beatmap)
    seq_d = calc_distances(seq)

    if len(seq_d) != len(ref_seq_d):
        return

    seq_d_d = ref_seq_d - seq_d

    mse = torch.mean(torch.square(seq_d_d))
    mae = torch.mean(torch.abs(seq_d_d))

    print(f"{name}: MSE = {mse}, MAE = {mae}")

    num_bins = 41
    bin_edges = torch.linspace(-20, 20, num_bins + 1)
    plt.hist(seq_d_d, bins=bin_edges, alpha=0.75, color='b', edgecolor='k')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    title = f"Distance similarity gen. beatmap [{name}]"
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-beatmap", type=str, required=True)
    parser.add_argument("--beatmap", type=str, required=True)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    for beatmap in glob.glob(args.beatmap + "\*.osu"):
        args.beatmap = beatmap
        main(args)
