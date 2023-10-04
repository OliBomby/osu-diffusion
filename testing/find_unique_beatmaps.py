import numpy as np
import pandas as pd
import scipy
import torch


def get_maps(mapper):
    regex = f"(?!\\s?(de\\s)?(it|that|{mapper}))(((^|[^\\S\r\n])(\\S)*([sz]'|'s))|((^|[^\\S\r\n])de\\s(\\S)*))"
    return df[
        ((df["Creator"] == mapper) | df["Version"].str.contains(mapper))
        & ~df["Version"].str.contains(regex)
    ]


mapper = input("Input mapper name: ")
mapper = "Sotarks"

df = pd.read_pickle("beatmap_df.pkl")
# print("Number of unique beatmap IDs = %s" % df["BeatmapID"].nunique())
# id_counts = df["BeatmapID"].value_counts()
# duplicated = id_counts[id_counts > 1]
# print(df[df["BeatmapID"].isin(duplicated.index)])
ckpt = torch.load("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\DiT-B-0130000.pt")

maps = get_maps(mapper)
print(f"Found {len(maps)} beatmaps.")
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"].cpu().numpy()

query = embedding_table[maps.index]
dist = np.mean(scipy.spatial.distance.cdist(embedding_table, query), 0)

k = min(10, len(dist))
min_idx = np.argpartition(dist, -k)[-k:]
for x in min_idx:
    print(
        dist[x],
        f"{maps.iloc[x]['Title']} [{maps.iloc[x]['Version']}]",
        maps.iloc[x]["BeatmapID"],
    )
