import json
import os

import pandas as pd

beatmap_df = pd.DataFrame(
    columns=[
        "BeatmapID",
        "BeatmapSetID",
        "TrackIndex",
        "SetIndex",
        "TrackName",
        "BeatmapName",
        "Artist",
        "Title",
        "Creator",
        "Source",
        "Tags",
        "Ruleset",
        "MD5Hash",
        "Difficulty",
        "OnlineOffset",
        "DrainTime",
        "TotalTime",
        "RankedStatus",
        "CirclesCount",
        "SpinnersCount",
        "SlidersCount",
        "CircleSize",
        "ApproachRate",
        "OverallDifficulty",
        "HPDrainRate",
        "SliderVelocity",
        "StackLeniency",
        "StarRating",
    ],
)
dataset_path = "D:\\Osu! Dingen\\Beatmap ML Datasets\\ORS13402"
idx = 0

for i in range(0, 13402):
    track_name = "Track" + str(i).zfill(5)
    metadata_File = os.path.join(dataset_path, track_name, "metadata.json")
    with open(metadata_File) as f:
        metadata = json.load(f)
    for j in range(len(metadata["Beatmaps"])):
        beatmap_name = str(idx).zfill(6) + "M" + str(j).zfill(3)
        beatmap_metadata = metadata["Beatmaps"][beatmap_name]
        beatmap_df.loc[idx] = [
            beatmap_metadata["BeatmapId"],
            metadata["BeatmapSetId"],
            i,
            j,
            track_name,
            beatmap_name,
            metadata["Artist"],
            metadata["Title"],
            metadata["Creator"],
            metadata["Source"],
            metadata["Tags"],
            beatmap_metadata["Ruleset"],
            beatmap_metadata["MD5Hash"],
            beatmap_metadata["Difficulty"],
            beatmap_metadata["OnlineOffset"],
            beatmap_metadata["DrainTime"],
            beatmap_metadata["TotalTime"],
            beatmap_metadata["RankedStatus"],
            beatmap_metadata["CirclesCount"],
            beatmap_metadata["SpinnersCount"],
            beatmap_metadata["SlidersCount"],
            beatmap_metadata["CircleSize"],
            beatmap_metadata["ApproachRate"],
            beatmap_metadata["OverallDifficulty"],
            beatmap_metadata["HPDrain"],
            beatmap_metadata["SliderVelocity"],
            beatmap_metadata["StackLeniency"],
            beatmap_metadata["StandardStarRating"]["0"],
        ]
        idx += 1
        print(f"\r{idx}", end="")

beatmap_df.to_pickle("beatmap_df.pkl")
beatmap_df.describe()
print(beatmap_df.head())
