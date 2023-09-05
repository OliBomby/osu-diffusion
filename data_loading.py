import math
import os.path
import pickle
import random
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from positional_embedding import offset_sequence_embedding
from positional_embedding import position_sequence_embedding
from positional_embedding import timestep_embedding
from slider import Position
from slider.beatmap import Beatmap
from slider.beatmap import HitObject
from slider.beatmap import Slider
from slider.beatmap import Spinner
from slider.curve import Catmull
from slider.curve import Linear
from slider.curve import MultiBezier
from slider.curve import Perfect

playfield_size = torch.tensor((512, 384))
context_size = 16 + 128


def create_datapoint(
    time: timedelta,
    pos: Position,
    datatype: int,
    last_pos: Position,
) -> torch.Tensor:
    dist = math.sqrt((pos.x - last_pos.x) ** 2 + (pos.y - last_pos.y) ** 2)
    pos_enc = torch.tensor(pos) / playfield_size
    type_enc = torch.zeros(18)
    type_enc[0] = time.total_seconds() * 1000
    type_enc[1] = dist
    type_enc[datatype + 2] = 1

    return torch.concatenate([pos_enc, type_enc], 0)


def repeat_type(repeat: int) -> int:
    if repeat < 4:
        return repeat - 1
    elif repeat % 2 == 0:
        return 3
    else:
        return 4


def append_control_points(
    datapoints: list[torch.Tensor],
    slider: Slider,
    datatype: int,
    duration: float,
    last_pos: Position,
) -> Position:
    control_point_count = len(slider.curve.points)

    for i in range(1, control_point_count - 1):
        time = slider.time + i / (control_point_count - 1) * duration
        pos = slider.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype, last_pos))
        last_pos = pos

    return last_pos


def get_data(hitobj: HitObject, last_pos: Position) -> tuple[torch.Tensor, Position]:
    if isinstance(hitobj, Slider) and len(hitobj.curve.points) < 100:
        datapoints = [
            create_datapoint(
                hitobj.time,
                hitobj.position,
                5 if hitobj.new_combo else 4,
                last_pos,
            ),
        ]
        last_pos = hitobj.position

        assert hitobj.repeat >= 1
        duration: float = (hitobj.end_time - hitobj.time) / hitobj.repeat

        if isinstance(hitobj.curve, Linear):
            last_pos = append_control_points(datapoints, hitobj, 9, duration, last_pos)
        elif isinstance(hitobj.curve, Catmull):
            last_pos = append_control_points(datapoints, hitobj, 8, duration, last_pos)
        elif isinstance(hitobj.curve, Perfect):
            last_pos = append_control_points(datapoints, hitobj, 7, duration, last_pos)
        elif isinstance(hitobj.curve, MultiBezier):
            control_point_count = len(hitobj.curve.points)

            for i in range(1, control_point_count - 1):
                time = hitobj.time + i / (control_point_count - 1) * duration
                pos = hitobj.curve.points[i]

                if pos == hitobj.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, 9, last_pos))
                    last_pos = pos
                elif pos != hitobj.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, 6, last_pos))
                    last_pos = pos

        datapoints.append(
            create_datapoint(
                hitobj.time + duration,
                hitobj.curve.points[-1],
                10,
                last_pos,
            ),
        )
        last_pos = hitobj.curve.points[-1]

        slider_end_pos = hitobj.curve(1)
        datapoints.append(
            create_datapoint(
                hitobj.end_time,
                slider_end_pos,
                11 + repeat_type(hitobj.repeat),
                last_pos,
            ),
        )

        return torch.stack(datapoints, 0), slider_end_pos

    if isinstance(hitobj, Spinner):
        return (
            torch.stack(
                (
                    create_datapoint(hitobj.time, hitobj.position, 2, last_pos),
                    create_datapoint(
                        hitobj.end_time,
                        hitobj.position,
                        3,
                        hitobj.position,
                    ),
                ),
                0,
            ),
            hitobj.position,
        )

    return (
        create_datapoint(
            hitobj.time,
            hitobj.position,
            1 if hitobj.new_combo else 0,
            last_pos,
        ).unsqueeze(0),
        hitobj.position,
    )


def beatmap_to_sequence(beatmap: Beatmap) -> torch.Tensor:
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


def random_flip(seq_x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        seq_x[0] = 1 - seq_x[0]
    if random.random() < 0.5:
        seq_x[1] = 1 - seq_x[1]
    return seq_x


def split_and_process_sequence(
    seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_x = seq[:2, :]
    seq_o = seq[2, :]
    seq_c = torch.concatenate(
        [
            timestep_embedding(seq[3, :], 128).T,
            seq[4:, :],
        ],
        0,
    )

    return seq_x, seq_o, seq_c


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "beatmap_idx",
        "seq_len",
        "stride",
        "index",
        "current_idx",
        "current_seq_x",
        "current_seq_o",
        "current_seq_c",
        "seq_index",
    )

    def __init__(
        self,
        beatmap_files: list[str],
        beatmap_idx: dict[int, int],
        seq_len: int,
        stride: int,
    ):
        self.beatmap_files = beatmap_files
        self.beatmap_idx = beatmap_idx
        self.seq_len = seq_len
        self.stride = stride
        self.index = 0
        self.current_idx = 0
        self.current_seq_x: torch.Tensor | None = None
        self.current_seq_o: torch.Tensor | None = None
        self.current_seq_c: torch.Tensor | None = None
        self.seq_index = 0

    def __iter__(self) -> "BeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        while (
            self.current_seq_x is None
            or self.seq_index + self.seq_len > self.current_seq_x.shape[1]
        ):
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap = Beatmap.from_path(beatmap_path)

            self.current_idx = self.beatmap_idx[beatmap.beatmap_id]

            seq_no_embed = beatmap_to_sequence(beatmap)
            (
                self.current_seq_x,
                self.current_seq_o,
                self.current_seq_c,
            ) = split_and_process_sequence(seq_no_embed)

            # Augment data
            self.current_seq_x = random_flip(self.current_seq_x)
            self.seq_index = random.randint(0, self.stride - 1)

            self.index += 1

        # Return the preprocessed hit objects as a sequence of overlapping windows
        x = self.current_seq_x[:, self.seq_index : self.seq_index + self.seq_len]
        # Obscure the absolute time by normalizing to zero and adding a random offset between zero and the max period
        # We do this to make sure the offset embedding utilizes the full range of values, which is also the case when sampling the model
        o = (
            self.current_seq_o[self.seq_index : self.seq_index + self.seq_len]
            - self.current_seq_o[self.seq_index]
            + random.random() * 100000
        )
        c = self.current_seq_c[:, self.seq_index : self.seq_index + self.seq_len]
        self.seq_index += self.stride
        return x, o, c, self.current_idx


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
        self,
        beatmap_files: list[str],
        beatmap_idx: dict[int, int],
        seq_len: int,
        stride: int,
        cycle_length: int,
    ):
        per_worker = int(math.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            BeatmapDatasetIterable(
                beatmap_files[
                    i * per_worker : min(len(beatmap_files), (i + 1) * per_worker)
                ],
                beatmap_idx,
                seq_len,
                stride,
            )
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
    def __init__(
        self,
        dataset_path: str,
        beatmap_idx: dict[int, int],
        start: int,
        end: int,
        seq_len: int,
        stride: int = 1,
        cycle_length: int = 1,
        shuffle: bool = False,
    ):
        super(BeatmapDataset).__init__()
        self.dataset_path = dataset_path
        self.beatmap_idx = beatmap_idx
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.stride = stride
        self.cycle_length = cycle_length
        self.shuffle = shuffle

    def _get_beatmap_files(self) -> list[str]:
        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            for beatmap_file in os.listdir(
                os.path.join(self.dataset_path, track_name, "beatmaps"),
            ):
                beatmap_files.append(
                    os.path.join(
                        self.dataset_path,
                        track_name,
                        "beatmaps",
                        beatmap_file,
                    ),
                )

        return beatmap_files

    def __iter__(self) -> InterleavingBeatmapDatasetIterable | BeatmapDatasetIterable:
        beatmap_files = self._get_beatmap_files()

        if self.shuffle:
            random.shuffle(beatmap_files)

        if self.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self.beatmap_idx,
                self.seq_len,
                self.stride,
                self.cycle_length,
            )

        return BeatmapDatasetIterable(
            beatmap_files,
            self.beatmap_idx,
            self.seq_len,
            self.stride,
        )


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def get_beatmap_idx() -> dict[int, int]:
    p = Path(__file__).with_name("beatmap_idx.pickle")
    with p.open("rb") as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


def get_processed_data_loader(
    dataset_path: str,
    start: int,
    end: int,
    seq_len: int,
    stride: int = 1,
    cycle_length=1,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        beatmap_idx=get_beatmap_idx(),
        start=start,
        end=end,
        seq_len=seq_len,
        stride=stride,
        cycle_length=cycle_length,
        shuffle=shuffle,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


if __name__ == "__main__":
    batch_size = 256
    num_workers = 16
    dataloader = get_processed_data_loader(
        dataset_path="D:\\Osu! Dingen\\Beatmap ML Datasets\\ORS13402",
        start=0,
        end=13402,
        seq_len=128,
        stride=16,
        cycle_length=128,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    # import matplotlib.pyplot as plt
    # for x, o, c, y in dataloader:
    #     x = torch.swapaxes(x, 1, 2)   # (N, T, C)
    #     c = torch.swapaxes(c, 1, 2)   # (N, T, E)
    #     print(x.shape, o.shape, c.shape, y.shape)
    #     batch_pos_emb = position_sequence_embedding(x * playfield_size, 128)
    #     print(batch_pos_emb.shape)
    #     batch_offset_emb = offset_sequence_embedding(o / 10, 128)
    #     print(batch_offset_emb.shape)
    #     print(y)
    #
    #     for j in range(batch_size):
    #         fig, axs = plt.subplots(3, figsize=(5, 30))
    #         axs[0].imshow(batch_pos_emb[j])
    #         axs[1].imshow(batch_offset_emb[j])
    #         axs[2].imshow(c[j])
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
