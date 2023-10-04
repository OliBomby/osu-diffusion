import pickle
from pathlib import Path

import numpy as np
import scipy
import torch


# beatmap_id = int(input("Input beatmap ID: "))
beatmap_id = 2116103

p = Path(__file__).parent.with_name("beatmap_idx.pickle")
with p.open("rb") as f:
    beatmap_idx = pickle.load(f)
    idx_beatmap = {v: k for k, v in beatmap_idx.items()}

idx = beatmap_idx[beatmap_id]

ckpt = torch.load(
    "D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s512\\0080000.pt",
)
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"].cpu()

query = embedding_table[idx]
dist = scipy.spatial.distance.cdist(embedding_table, query.unsqueeze(0))[:, 0]

k = 10
min_idx = np.argpartition(dist, -k)[-k:]
for x in min_idx:
    try:
        print(idx_beatmap[x], dist[x])
    except KeyError:
        pass
