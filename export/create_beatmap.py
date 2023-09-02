import numpy as np
import torch
from datetime import timedelta
from slider import Position
from slider.beatmap import Circle, Slider, Spinner, Beatmap, TimingPoint, HitObject
from slider.curve import MultiBezier, Perfect, Catmull, Linear

from export.slider_path import SliderPath


def create_beatmap(seq, ref_beatmap: Beatmap, version: str):
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
        type_index = int(torch.argmax(seq[4:, j]))
        pos = Position(x, y)

        if type_index == 0:
            hit_objects.append(Circle(pos, time, 0))
        elif type_index == 1:
            curr_object = Spinner(pos, time, 0, time)
        elif type_index == 2 and isinstance(curr_object, Spinner):
            curr_object.end_time = time
            hit_objects.append(curr_object)
        elif type_index == 3:
            curr_object = Slider(pos, time, time, 0, MultiBezier([pos], 0), 0, 0, 0, 0, 0, 0, [], [])
            curr_slider_path = [pos]
            curr_slider_type = "Bezier"
        elif type_index == 4 and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
        elif type_index == 5 and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_type = "PerfectCurve"
        elif type_index == 6 and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_type = "Catmull"
        elif type_index == 7 and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_path.append(pos)
        elif type_index == 8 and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            span_duration = (time - curr_object.time).total_seconds() * 1000
        elif type_index >= 9 and isinstance(curr_object, Slider):
            # Determine length by finding the closest position to pos on the slider path
            slider_path = SliderPath(curr_slider_type, np.array(curr_slider_path, dtype=float))
            req_length = slider_path.get_distance() * position_to_progress(slider_path, np.array(pos))
            curr_object.curve = slider_path_to_curve(slider_path, req_length)
            curr_object.length = req_length
            curr_object.end_time = time
            duration = (time - curr_object.time).total_seconds() * 1000
            curr_object.repeat = int(round(duration / span_duration)) if type_index > 11 else type_index - 8
            curr_object.edge_sounds = [0] * curr_object.repeat
            curr_object.edge_additions = ['0:0'] * curr_object.repeat
            hit_objects.append(curr_object)

            # Add a timing point for slider velocity
            tp = ref_beatmap.timing_point_at(curr_object.time)
            parent = tp.parent if tp.parent is not None else tp
            ms_per_beat = tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat
            global_sv = ref_beatmap.slider_multiplier
            new_sv_multiplier = req_length * ms_per_beat / (100 * global_sv * span_duration)
            timing_points.append(TimingPoint(
                curr_object.time,
                -100 / new_sv_multiplier if new_sv_multiplier > 0 else -100,
                tp.meter,
                tp.sample_type,
                tp.sample_set,
                tp.volume,
                parent,
                tp.kiai_mode))

    return new_difficulty(ref_beatmap, version, hit_objects, timing_points)

def slider_path_to_curve(slider_path: SliderPath, req_length: float):
    points = [Position(p[0], p[1]) for p in slider_path.controlPoints]

    if slider_path.pathType == "PerfectCurve" and is_collinear(*points):
        slider_path.pathType = "Bezier"

    if slider_path.pathType == "Bezier":
        return MultiBezier(points, req_length)
    elif slider_path.pathType == "PerfectCurve":
        return Perfect(points, req_length)
    elif slider_path.pathType == "Catmull":
        return Catmull(points, req_length)
    else:
        return Linear(points, req_length)


def is_collinear(a, b, c):
    a, b, c = np.array([a, b, c], dtype=np.float64)

    a_squared = np.sum(np.square(b - c))
    b_squared = np.sum(np.square(a - c))
    c_squared = np.sum(np.square(a - b))

    if np.isclose([a_squared, b_squared, c_squared], 0).any():
        return True

    s = a_squared * (b_squared + c_squared - a_squared)
    t = b_squared * (a_squared + c_squared - b_squared)
    u = c_squared * (a_squared + b_squared - c_squared)

    sum_ = s + t + u

    if np.isclose(sum_, 0):
        return True

    return False


def position_to_progress(slider_path: SliderPath, pos: np.ndarray):
    eps = 1e-4
    lr = 1
    t = 1
    for i in range(100):
        grad = np.linalg.norm(slider_path.position_at(t) - pos) - np.linalg.norm(slider_path.position_at(t - eps) - pos)
        t -= lr * grad

        if grad == 0 or t < 0 or t > 10:
            break

    return np.clip(t, 0, 10)

def new_difficulty(ref_beatmap: Beatmap, version: str, hit_objects: list[HitObject], timing_points: list[TimingPoint]):
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
        hit_objects=hit_objects
    )
