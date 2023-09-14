import json
import math
import os.path
import random
import pickle
from datetime import timedelta
from pathlib import Path

from slider import Position
from slider.beatmap import Beatmap, HitObject, Slider, Spinner
import torch
from slider.curve import Linear, Catmull, Perfect, MultiBezier
from torch.utils.data import IterableDataset, DataLoader

from positional_embedding import timestep_embedding, position_sequence_embedding, offset_sequence_embedding

playfield_size = torch.tensor((512, 384))
feature_size = 19


def create_datapoint(time: timedelta, pos: Position, datatype):
    features = torch.zeros(19)
    features[0] = pos.x
    features[1] = pos.y
    features[2] = time.total_seconds() * 1000
    features[datatype + 3] = 1

    return features


def repeat_type(repeat):
    if repeat < 4:
        return repeat - 1
    elif repeat % 2 == 0:
        return 3
    else:
        return 4


def append_control_points(datapoints, ho: Slider, datatype, duration):
    control_point_count = len(ho.curve.points)

    for i in range(1, control_point_count - 1):
        time = ho.time + i / (control_point_count - 1) * duration
        pos = ho.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype))


def get_data(ho: HitObject):
    if isinstance(ho, Slider) and len(ho.curve.points) < 100:
        datapoints = [create_datapoint(ho.time, ho.position, 5 if ho.new_combo else 4)]

        assert ho.repeat >= 1
        duration = (ho.end_time - ho.time) / ho.repeat

        if isinstance(ho.curve, Linear):
            append_control_points(datapoints, ho, 9, duration)
        elif isinstance(ho.curve, Catmull):
            append_control_points(datapoints, ho, 8, duration)
        elif isinstance(ho.curve, Perfect):
            append_control_points(datapoints, ho, 7, duration)
        elif isinstance(ho.curve, MultiBezier):
            control_point_count = len(ho.curve.points)

            for i in range(1, control_point_count - 1):
                time = ho.time + i / (control_point_count - 1) * duration
                pos = ho.curve.points[i]

                if pos == ho.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, 9))
                elif pos != ho.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, 6))

        datapoints.append(create_datapoint(ho.time + duration, ho.curve.points[-1], 10))
        datapoints.append(create_datapoint(ho.end_time, ho.curve(1), 11 + repeat_type(ho.repeat)))

        return torch.stack(datapoints, 0)

    if isinstance(ho, Spinner):
        return torch.stack((create_datapoint(ho.time, ho.position, 2), create_datapoint(ho.end_time, ho.position, 3)), 0)

    return create_datapoint(ho.time, ho.position, 1 if ho.new_combo else 0).unsqueeze(0)


def beatmap_to_sequence(beatmap):
    # Get the hit objects
    hit_objects = beatmap.hit_objects(stacking=False)
    data_chunks = [get_data(ho) for ho in hit_objects]

    sequence = torch.concatenate(data_chunks, 0)
    sequence = torch.swapaxes(sequence, 0, 1)

    return sequence.float()


def random_flip(seq: torch.Tensor):
    if random.random() < 0.5:
        seq[0] = 512 - seq[0]
    if random.random() < 0.5:
        seq[1] = 384 - seq[1]
    return seq


def split_and_process_sequence(seq: torch.Tensor):
    offset = torch.roll(seq[:2, :], 1, 1)
    offset[0, 0] = 256
    offset[1, 0] = 192
    seq_d = torch.linalg.vector_norm(seq[:2, :] - offset, ord=2, dim=0)
    # Augment and normalize positions for diffusion
    seq_x = random_flip(seq[:2, :]) / playfield_size.unsqueeze(1)
    seq_o = seq[2, :]
    seq_c = torch.concatenate(
        [
            timestep_embedding(seq_d, 128).T,
            seq[3:, :],
        ], 0)

    return (seq_x, seq_o, seq_c), seq.shape[1]


def load_and_process_beatmap(beatmap: Beatmap):
    seq = beatmap_to_sequence(beatmap)
    return split_and_process_sequence(seq)


def window_and_relative_time(seq, s, e):
    seq_x, seq_o, seq_c = seq
    x = seq_x[:, s:e]
    # Obscure the absolute time by normalizing to zero and adding a random offset between zero and the max period
    # We do this to make sure the offset embedding utilizes the full range of values, which is also the case when sampling the model
    o = seq_o[s:e] - seq_o[s] + random.random() * 100000
    c = seq_c[:, s:e]

    return x, o, c


class BeatmapDatasetIterable:
    def __init__(self, beatmap_files, beatmap_idx, seq_len, stride, seq_func=None, win_func=None):
        self.beatmap_files = beatmap_files
        self.beatmap_idx = beatmap_idx
        self.seq_len = seq_len
        self.stride = stride
        self.index = 0
        self.current_idx = 0
        self.current_seq = None
        self.current_seq_len = -1
        self.seq_index = 0
        self.seq_func = seq_func if seq_func is not None else lambda x: beatmap_to_sequence(x)
        self.win_func = win_func if win_func is not None else lambda x, s, e: x[:, s:e]

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_seq is None or self.seq_index + self.seq_len > self.current_seq_len:
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap = Beatmap.from_path(beatmap_path)

            self.current_idx = self.beatmap_idx[beatmap.beatmap_id]
            self.current_seq, self.current_seq_len = self.seq_func(beatmap)
            self.seq_index = random.randint(0, self.stride - 1)
            self.index += 1

        # Return the preprocessed hit objects as a sequence of overlapping windows
        window = self.win_func(self.current_seq, self.seq_index, self.seq_index + self.seq_len)
        self.seq_index += self.stride
        return window, self.current_idx


class InterleavingBeatmapDatasetIterable:
    def __init__(self, beatmap_files, beatmap_idx, seq_len, stride, cycle_length, seq_func=None, win_func=None):
        per_worker = int(math.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [BeatmapDatasetIterable(beatmap_files[i * per_worker:min(len(beatmap_files), (i + 1) * per_worker)], beatmap_idx, seq_len, stride, seq_func, win_func) for i in range(cycle_length)]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDataset(IterableDataset):
    def __init__(self, dataset_path, beatmap_idx, start, end, seq_len, stride=1, cycle_length=1, shuffle=False, subset_ids=None, seq_func=None, win_func=None):
        super(BeatmapDataset).__init__()
        self.dataset_path = dataset_path
        self.beatmap_idx = beatmap_idx
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.stride = stride
        self.cycle_length = cycle_length
        self.shuffle = shuffle
        self.subset_ids = subset_ids
        self.seq_func = seq_func
        self.win_func = win_func

    def _get_beatmap_files(self):
        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            if self.subset_ids is not None:
                metadata_File = os.path.join(self.dataset_path, track_name, "metadata.json")
                with open(metadata_File, 'r') as f:
                    metadata = json.load(f)
                for beatmap_name in metadata["Beatmaps"]:
                    beatmap_metadata = metadata["Beatmaps"][beatmap_name]
                    if beatmap_metadata['BeatmapId'] in self.subset_ids:
                        beatmap_files.append(os.path.join(self.dataset_path, track_name, "beatmaps", beatmap_name + '.osu'))
            else:
                for beatmap_file in os.listdir(os.path.join(self.dataset_path, track_name, "beatmaps")):
                    beatmap_files.append(os.path.join(self.dataset_path, track_name, "beatmaps", beatmap_file))

        return beatmap_files

    def __iter__(self):
        beatmap_files = self._get_beatmap_files()

        if self.shuffle:
            random.shuffle(beatmap_files)

        if self.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(beatmap_files, self.beatmap_idx, self.seq_len, self.stride, self.cycle_length, self.seq_func, self.win_func)

        return BeatmapDatasetIterable(beatmap_files, self.beatmap_idx, self.seq_len, self.stride, self.seq_func, self.win_func)


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def get_beatmap_idx():
    p = Path(__file__).with_name('beatmap_idx.pickle')
    with p.open('rb') as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


def get_processed_data_loader(dataset_path,
                              start,
                              end,
                              seq_len,
                              stride=1,
                              cycle_length=1,
                              batch_size=1,
                              num_workers=0,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False,
                              subset_ids=None,
                              seq_func=None,
                              win_func=None
                              ):
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        beatmap_idx=get_beatmap_idx(),
        start=start,
        end=end,
        seq_len=seq_len,
        stride=stride,
        cycle_length=cycle_length,
        shuffle=shuffle,
        subset_ids=subset_ids,
        seq_func=seq_func,
        win_func=win_func,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


if __name__ == '__main__':
    # batch_size = 256
    # num_workers = 4
    batch_size = 1
    num_workers = 0
    # import pandas as pd
    # subset_ids = pd.read_csv("C:\\Users\\Olivier\\Documents\\GitHub\\osu-diffusion\\results\\tags\\clean_10000.csv")["BeatmapID"].tolist()
    dataloader = get_processed_data_loader(
        dataset_path = "D:\\Osu! Dingen\\Beatmap ML Datasets\\ORS13402",
        start = 0,
        end = 13402,
        seq_len = 128,
        stride = 16,
        cycle_length = 1,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False,
        pin_memory = False,
        drop_last = True,
        # subset_ids=subset_ids,
        seq_func=load_and_process_beatmap,
        win_func=window_and_relative_time,
    )

    import matplotlib.pyplot as plt
    for (x, o, c), y in dataloader:
        x = torch.swapaxes(x, 1, 2)   # (N, T, C)
        c = torch.swapaxes(c, 1, 2)   # (N, T, E)
        print(x.shape, o.shape, c.shape, y.shape)
        batch_pos_emb = position_sequence_embedding(x * playfield_size, 128)
        print(batch_pos_emb.shape)
        batch_offset_emb = offset_sequence_embedding(o / 10, 128)
        print(batch_offset_emb.shape)
        print(y)

        for j in range(batch_size):
            fig, axs = plt.subplots(3, figsize=(5, 20))
            axs[0].imshow(batch_pos_emb[j])
            axs[1].imshow(batch_offset_emb[j])
            axs[2].imshow(c[j])
            print(y[j])
            plt.show()
        break

    # import time
    # import tqdm
    # count = 0
    # start = time.time()
    # for f in tqdm.tqdm(dataloader, total=76200, smoothing=0.01):
    #     count += 1
    #     # print(f"\r{count}, {count / (time.time() - start)} per second, beatmap index {torch.max(f[1])}", end='')

