import os
import json
import pandas as pd
import slider

beatmap_df = pd.DataFrame(columns=[
    "BeatmapID",
    "BeatmapSetID",
    "TrackIndex",
    "SetIndex",
    "TrackName",
    "BeatmapName",
    "AudioFilename",
    "AudioLeadIn",
    "PreviewTime",
    "Countdown",
    "SampleSet",
    "StackLeniency",
    "Mode",
    "LetterboxInBreaks",
    "WidescreenStoryboard",
    "Title",
    "TitleUnicode",
    "Artist",
    "ArtistUnicode",
    "Creator",
    "Version",
    "Source",
    "Tags",
    "HPDrainRate",
    "CircleSize",
    "OverallDifficulty",
    "ApproachRate",
    "SliderMultiplier",
    "SliderTickRate",
    "OnlineOffset",
    "DrainTime",
    "TotalTime",
    "RankedStatus",
    "StarRating"
])
dataset_path = "D:\\Osu! Dingen\\Beatmap ML Datasets\\ORS10548"
idx = 0

for i in range(0, 10548):
    track_name = "Track" + str(i).zfill(5)
    metadata_File = os.path.join(dataset_path, track_name, "metadata.json")
    with open(metadata_File, 'r') as f:
        metadata = json.load(f)
    for j in range(len(metadata["Beatmaps"])):
        beatmap_name = "M" + str(j).zfill(3)
        beatmap_path = os.path.join(dataset_path, track_name, "beatmaps", beatmap_name)
        beatmap = slider.Beatmap.from_path(beatmap_path + ".osu")
        beatmap_metadata = metadata["Beatmaps"][beatmap_name]
        beatmap_df.loc[idx] = [
            beatmap_metadata["BeatmapId"],
            metadata["BeatmapSetId"],
            i,
            j,
            track_name,
            beatmap_name,
            beatmap.audio_filename,
            beatmap.audio_lead_in,
            beatmap.preview_time,
            beatmap.countdown,
            beatmap.sample_set,
            beatmap.stack_leniency,
            beatmap.mode,
            beatmap.letterbox_in_breaks,
            beatmap.widescreen_storyboard,
            beatmap.title,
            beatmap.title_unicode,
            beatmap.artist,
            beatmap.artist_unicode,
            beatmap.creator,
            beatmap.version,
            beatmap.source,
            beatmap.tags,
            beatmap.hp_drain_rate,
            beatmap.circle_size,
            beatmap.overall_difficulty,
            beatmap.approach_rate,
            beatmap.slider_multiplier,
            beatmap.slider_tick_rate,
            beatmap_metadata["OnlineOffset"],
            beatmap_metadata["DrainTime"],
            beatmap_metadata["TotalTime"],
            beatmap_metadata["RankedStatus"],
            beatmap_metadata["StandardStarRating"]["0"],
        ]
        idx += 1
        print(f"\r{idx}", end='')

beatmap_df.to_pickle("beatmap_df.pkl")
beatmap_df.describe()
print(beatmap_df.head())
