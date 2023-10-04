import json
import os
import pickle

beatmap_idx = {}
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
        beatmap_idx[beatmap_metadata["BeatmapId"]] = idx
        idx += 1
        print(f"\r{idx}", end="")

with open("../beatmap_idx.pickle", "wb") as f:
    pickle.dump(beatmap_idx, f)
