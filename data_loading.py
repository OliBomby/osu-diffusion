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

from positional_embedding import timestep_embedding


playfield_size = torch.tensor((512, 384))
context_size = 14 + 128 * 2


def create_datapoint(time: timedelta, pos: Position, datatype, last_pos: Position):
    dist = math.sqrt((pos.x - last_pos.x)**2 + (pos.y - last_pos.y)**2)
    pos_enc = torch.tensor(pos) / playfield_size
    type_enc = torch.zeros(18)
    type_enc[0] = time.total_seconds() * 1000
    type_enc[1] = dist
    type_enc[datatype + 2] = 1

    return torch.concatenate([pos_enc, type_enc], 0)


def repeat_type(repeat):
    if repeat < 4:
        return repeat - 1
    elif repeat % 2 == 0:
        return 3
    else:
        return 4


def append_control_points(datapoints, ho: Slider, datatype, duration, last_pos: Position):
    control_point_count = len(ho.curve.points)

    for i in range(1, control_point_count - 1):
        time = ho.time + i / (control_point_count - 1) * duration
        pos = ho.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype, last_pos))
        last_pos = pos

    return last_pos


def get_data(ho: HitObject, last_pos: Position):
    if isinstance(ho, Slider) and len(ho.curve.points) < 100:
        datapoints = [create_datapoint(ho.time, ho.position, 5 if ho.new_combo else 4, last_pos)]
        last_pos = ho.position

        assert ho.repeat >= 1
        duration = (ho.end_time - ho.time) / ho.repeat

        if isinstance(ho.curve, Linear):
            last_pos = append_control_points(datapoints, ho, 9, duration, last_pos)
        elif isinstance(ho.curve, Catmull):
            last_pos = append_control_points(datapoints, ho, 8, duration, last_pos)
        elif isinstance(ho.curve, Perfect):
            last_pos = append_control_points(datapoints, ho, 7, duration, last_pos)
        elif isinstance(ho.curve, MultiBezier):
            control_point_count = len(ho.curve.points)

            for i in range(1, control_point_count - 1):
                time = ho.time + i / (control_point_count - 1) * duration
                pos = ho.curve.points[i]

                if pos == ho.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, 9, last_pos))
                    last_pos = pos
                elif pos != ho.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, 6, last_pos))
                    last_pos = pos

        datapoints.append(create_datapoint(ho.time + duration, ho.curve.points[-1], 10, last_pos))
        last_pos = ho.curve.points[-1]

        slider_end_pos = ho.curve(1)
        datapoints.append(create_datapoint(ho.end_time, slider_end_pos, 11 + repeat_type(ho.repeat), last_pos))

        return torch.stack(datapoints, 0), slider_end_pos

    if isinstance(ho, Spinner):
        return torch.stack((create_datapoint(ho.time, ho.position, 2, last_pos), create_datapoint(ho.end_time, ho.position, 3, ho.position)), 0), ho.position

    return create_datapoint(ho.time, ho.position, 1 if ho.new_combo else 0, last_pos).unsqueeze(0), ho.position


def beatmap_to_sequence(beatmap):
    # Get the hit objects
    hit_objects = beatmap.hit_objects(stacking=False)
    data_chunks = []
    last_pos = Position(256, 192)
    for ho in hit_objects:
        data_chunk, last_pos = get_data(ho, last_pos)
        data_chunks.append(data_chunk)

    sequence = torch.concatenate(data_chunks, 0)
    sequence = torch.swapaxes(sequence, 0, 1)

    return sequence.float()


def random_flip(seq_x: torch.Tensor):
    if random.random() < 0.5:
        seq_x[0] = 1 - seq_x[0]
    if random.random() < 0.5:
        seq_x[1] = 1 - seq_x[1]
    return seq_x


def split_and_process_sequence(seq: torch.Tensor):
    seq_x = seq[:2, :]
    seq_y = torch.concatenate(
        [
            timestep_embedding(seq[2, :] / 100, 128, 36000).T,
            timestep_embedding(seq[3, :], 128).T,
            seq[4, :].unsqueeze(0),
            seq[6:9, :],
            seq[10:, :],
        ], 0)

    return seq_x, seq_y


class BeatmapDatasetIterable:
    def __init__(self, beatmap_files, beatmap_idx, seq_len, stride):
        self.beatmap_files = beatmap_files
        self.beatmap_idx = beatmap_idx
        self.seq_len = seq_len
        self.stride = stride
        self.index = 0
        self.current_idx = 0
        self.current_seq_x = None
        self.current_seq_y = None
        self.seq_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_seq_x is None or self.seq_index + self.seq_len > self.current_seq_x.shape[1]:
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap = Beatmap.from_path(beatmap_path)

            self.current_idx = self.beatmap_idx[beatmap.beatmap_id]

            seq_no_embed = beatmap_to_sequence(beatmap)
            self.current_seq_x, self.current_seq_y = split_and_process_sequence(seq_no_embed)

            # Augment data
            self.current_seq_x = random_flip(self.current_seq_x)
            self.seq_index = random.randint(0, self.stride - 1)

            self.index += 1

        # Return the preprocessed hit objects as a sequence of overlapping windows
        x = self.current_seq_x[:, self.seq_index:self.seq_index + self.seq_len]
        y = self.current_seq_y[:, self.seq_index:self.seq_index + self.seq_len]
        self.seq_index += self.stride
        return x, y, self.current_idx


class InterleavingBeatmapDatasetIterable:
    def __init__(self, beatmap_files, beatmap_idx, seq_len, stride, cycle_length):
        per_worker = int(math.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [BeatmapDatasetIterable(beatmap_files[i * per_worker:min(len(beatmap_files), (i + 1) * per_worker)], beatmap_idx, seq_len, stride) for i in range(cycle_length)]
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
    def __init__(self, dataset_path, beatmap_idx, start, end, seq_len, stride=1, cycle_length=1, shuffle=False):
        super(BeatmapDataset).__init__()
        self.dataset_path = dataset_path
        self.beatmap_idx = beatmap_idx
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.stride = stride
        self.cycle_length = cycle_length
        self.shuffle = shuffle

    def _get_beatmap_files(self):
        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            for beatmap_file in os.listdir(os.path.join(self.dataset_path, track_name, "beatmaps")):
                beatmap_files.append(os.path.join(self.dataset_path, track_name, "beatmaps", beatmap_file))

        return beatmap_files

    def __iter__(self):
        beatmap_files = self._get_beatmap_files()

        if self.shuffle:
            random.shuffle(beatmap_files)

        if self.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(beatmap_files, self.beatmap_idx, self.seq_len, self.stride, self.cycle_length)

        return BeatmapDatasetIterable(beatmap_files, self.beatmap_idx, self.seq_len, self.stride)


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


def get_processed_data_loader(dataset_path, start, end, seq_len, stride=1, cycle_length=1, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=False):
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        beatmap_idx=get_beatmap_idx(),
        start=start,
        end=end,
        seq_len=seq_len,
        stride=stride,
        cycle_length=cycle_length,
        shuffle=shuffle
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
    batch_size = 256
    num_workers = 16
    dataloader = get_processed_data_loader(
        dataset_path = "D:\\Osu! Dingen\\Beatmap ML Datasets\\ORS13402",
        start = 0,
        end = 13402,
        seq_len = 64,
        stride = 16,
        cycle_length = 1,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False,
        pin_memory = False,
        drop_last = True
    )

    # import matplotlib.pyplot as plt
    # for x, c, y in dataloader:
    #     x = torch.swapaxes(x, 1, 2)   # (N, T, C)
    #     c = torch.swapaxes(c, 1, 2)   # (N, T, E)
    #     print(x.shape, c.shape, y.shape)
    #     batch_pos_emb = position_sequence_embedding(x * playfield_size, 128)
    #     print(batch_pos_emb.shape)
    #     print(y)
    #
    #     for j in range(batch_size):
    #         fig, axs = plt.subplots(2, figsize=(10, 5))
    #         axs[0].imshow(batch_pos_emb[j])
    #         axs[1].imshow(c[j])
    #         print(y[j])
    #         plt.show()
    #     break

    import time
    import tqdm
    count = 0
    start = time.time()
    for f in tqdm.tqdm(dataloader, total=76200, smoothing=0.01):
        count += 1
        # print(f"\r{count}, {count / (time.time() - start)} per second, beatmap index {torch.max(f[1])}", end='')

