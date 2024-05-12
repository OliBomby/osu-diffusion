from datetime import timedelta

import numpy as np
import torch
from matplotlib import pyplot as plt

from export.slider_path import SliderPath
from slider import Position
from slider.beatmap import Beatmap
from slider.beatmap import Circle
from slider.beatmap import HitObject
from slider.beatmap import Slider
from slider.beatmap import Spinner
from slider.beatmap import TimingPoint
from slider.curve import Catmull
from slider.curve import Curve
from slider.curve import Linear
from slider.curve import MultiBezier
from slider.curve import Perfect


def create_beatmap(seq: torch.Tensor, ref_beatmap: Beatmap, version: str) -> Beatmap:
    seq_len = seq.shape[1]
    hit_objects = []
    timing_points = [tp for tp in ref_beatmap.timing_points if tp.parent is None]
    curr_object = None
    curr_slider_path = []
    curr_slider_type = None
    span_duration = 0
    for j in range(seq_len):
        x = int(round(float(seq[0, j] * 512)))
        y = int(round(float(seq[1, j] * 384)))
        time = timedelta(seconds=float(seq[2, j] / 1000))
        type_index = int(torch.argmax(seq[3:, j]))
        pos = Position(x, y)

        match type_index:
            case 0:
                hit_objects.append(Circle(pos, time, 0, new_combo=False))
            case 1:
                hit_objects.append(Circle(pos, time, 0, new_combo=True))
            case 2:
                curr_object = Spinner(pos, time, 0, time, new_combo=True)
            case 3 if isinstance(curr_object, Spinner):
                curr_object.end_time = time
                hit_objects.append(curr_object)
            case 4:
                curr_object = Slider(
                    position=pos,
                    time=time,
                    combo_skip=0,
                    end_time=time,
                    hitsound=0,
                    curve=MultiBezier([pos], 0),
                    repeat=0,
                    length=0,
                    ticks=0,
                    num_beats=0,
                    tick_rate=0,
                    ms_per_beat=0,
                    edge_sounds=[],
                    edge_additions=[],
                    new_combo=False,
                )
                curr_slider_path = [pos]
                curr_slider_type = "Bezier"
            case 5:
                curr_object = Slider(
                    position=pos,
                    time=time,
                    combo_skip=0,
                    end_time=time,
                    hitsound=0,
                    curve=MultiBezier([pos], 0),
                    repeat=0,
                    length=0,
                    ticks=0,
                    num_beats=0,
                    tick_rate=0,
                    ms_per_beat=0,
                    edge_sounds=[],
                    edge_additions=[],
                    new_combo=True,
                )
                curr_slider_path = [pos]
                curr_slider_type = "Bezier"
            case 6 if isinstance(curr_object, Slider):
                curr_slider_path.append(pos)
            case 7 if isinstance(curr_object, Slider):
                curr_slider_path.append(pos)
                curr_slider_type = "PerfectCurve"
            case 8 if isinstance(curr_object, Slider):
                curr_slider_path.append(pos)
                curr_slider_type = "Catmull"
            case 9 if isinstance(curr_object, Slider):
                curr_slider_path.append(pos)
                curr_slider_path.append(pos)
            case 10 if isinstance(curr_object, Slider):
                curr_slider_path.append(pos)
                span_duration = (time - curr_object.time).total_seconds() * 1000
            case _ if isinstance(curr_object, Slider):
                # Determine length by finding the closest position to pos on the slider path
                slider_path = SliderPath(
                    curr_slider_type,
                    np.array(curr_slider_path, dtype=float),
                )
                req_length = slider_path.get_distance() * position_to_progress(
                    slider_path,
                    np.array(pos),
                )
                curr_object.curve = slider_path_to_curve(slider_path, req_length)
                curr_object.length = req_length
                curr_object.end_time = time
                duration = (time - curr_object.time).total_seconds() * 1000
                curr_object.repeat = (
                    int(round(duration / span_duration))
                    if type_index > 13
                    else type_index - 10
                )
                curr_object.edge_sounds = [0] * curr_object.repeat
                curr_object.edge_additions = ["0:0"] * curr_object.repeat
                hit_objects.append(curr_object)

                # Add a timing point for slider velocity
                tp = ref_beatmap.timing_point_at(curr_object.time)
                parent = tp.parent if tp.parent is not None else tp
                ms_per_beat = (
                    tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat
                )
                global_sv = ref_beatmap.slider_multiplier
                new_sv_multiplier = (
                    req_length * ms_per_beat / (100 * global_sv * span_duration)
                )
                timing_points.append(
                    TimingPoint(
                        curr_object.time,
                        -100 / new_sv_multiplier if new_sv_multiplier > 0 else -100,
                        tp.meter,
                        tp.sample_type,
                        tp.sample_set,
                        tp.volume,
                        parent,
                        tp.kiai_mode,
                    ),
                )

    return new_difficulty(ref_beatmap, version, hit_objects, timing_points)


def slider_path_to_curve(slider_path: SliderPath, req_length: float) -> Curve:
    points = [Position(p[0], p[1]) for p in slider_path.control_points]

    return Curve.from_kind_and_points(slider_path.path_type[0], points, req_length)


def position_to_progress(slider_path: SliderPath, pos: np.ndarray) -> np.ndarray:
    eps = 1e-4
    lr = 1
    t = 1
    for i in range(100):
        grad = np.linalg.norm(slider_path.position_at(t) - pos) - np.linalg.norm(
            slider_path.position_at(t - eps) - pos,
        )
        t -= lr * grad

        if grad == 0 or t < 0 or t > 1:
            break

    return np.clip(t, 0, 1)


def new_difficulty(
    ref_beatmap: Beatmap,
    version: str,
    hit_objects: list[HitObject],
    timing_points: list[TimingPoint],
) -> Beatmap:
    return Beatmap(
        format_version=ref_beatmap.format_version,
        audio_filename=ref_beatmap.audio_filename,
        audio_lead_in=ref_beatmap.audio_lead_in,
        preview_time=ref_beatmap.preview_time,
        countdown=ref_beatmap.countdown,
        sample_set=ref_beatmap.sample_set,
        stack_leniency=ref_beatmap.stack_leniency,
        mode=ref_beatmap.mode,
        letterbox_in_breaks=ref_beatmap.letterbox_in_breaks,
        widescreen_storyboard=ref_beatmap.widescreen_storyboard,
        bookmarks=ref_beatmap.bookmarks,
        distance_spacing=ref_beatmap.distance_spacing,
        beat_divisor=ref_beatmap.beat_divisor,
        grid_size=ref_beatmap.grid_size,
        timeline_zoom=ref_beatmap.timeline_zoom,
        title=ref_beatmap.title,
        title_unicode=ref_beatmap.title_unicode,
        artist=ref_beatmap.artist,
        artist_unicode=ref_beatmap.artist_unicode,
        creator=ref_beatmap.creator,
        version=version,
        source=ref_beatmap.source,
        tags=ref_beatmap.tags,
        beatmap_id=0,
        beatmap_set_id=ref_beatmap.beatmap_set_id,
        hp_drain_rate=ref_beatmap.hp_drain_rate,
        circle_size=ref_beatmap.circle_size,
        overall_difficulty=ref_beatmap.overall_difficulty,
        approach_rate=ref_beatmap.approach_rate,
        slider_multiplier=ref_beatmap.slider_multiplier,
        slider_tick_rate=ref_beatmap.slider_tick_rate,
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def plot_beatmap(ax: plt.Axes, beatmap: Beatmap, time, window_size) -> list:
    width = beatmap.cs() * 8
    hit_objects = beatmap.hit_objects(spinners=False)
    min_time, max_time = timedelta(seconds=(time - window_size) / 1000), timedelta(
        seconds=(time + window_size) / 1000,
    )
    windowed = [ho for ho in hit_objects if min_time < ho.time < max_time]
    artists = []
    for hitobj in windowed:
        if not isinstance(hitobj, Slider):
            continue

        path_type = "Bezier"
        if isinstance(hitobj.curve, Perfect):
            path_type = "PerfectCurve"
        elif isinstance(hitobj.curve, Catmull):
            path_type = "Catmull"
        elif isinstance(hitobj.curve, Linear):
            path_type = "Linear"

        slider_path = SliderPath(
            path_type,
            np.array(hitobj.curve.points, dtype=float),
            hitobj.curve.req_length,
        )
        path = []
        slider_path.get_path_to_progress(path, 0, 1)
        p = np.vstack(path)
        artists.append(
            ax.plot(
                p[:, 0],
                p[:, 1],
                color="green",
                linewidth=width,
                solid_capstyle="round",
                solid_joinstyle="round",
            )[0],
        )

    p = np.array([ho.position for ho in windowed]).reshape((-1, 2))
    artists.append(ax.scatter(p[:, 0], p[:, 1], s=width**2, c="Lime"))
    return artists
