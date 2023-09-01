import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Some helper functions for plotting annotated t-SNE visualizations

try:
    from adjustText import adjust_text
except ImportError:
    def adjust_text(*args, **kwargs):
        pass

def plot_bg(bg_alpha=.01, figsize=(13, 9), emb_2d=None):
    """Create and return a plot of all our beatmap embeddings with very low opacity.
    (Intended to be used as a basis for further - more prominent - plotting of a
    subset of beatmaps. Having the overall shape of the map space in the background is
    useful for context.)
    """
    if emb_2d is None:
        emb_2d = embs
    fig, ax = plt.subplots(figsize=figsize)
    X = emb_2d[:, 0]
    Y = emb_2d[:, 1]
    ax.scatter(X, Y, alpha=bg_alpha)
    return fig, ax


def annotate_sample(n, star_rating_thresh=0.0):
    """Plot our embeddings with a random sample of n beatmaps annotated.
    Only selects beatmaps where the star rating is at least star_rating_thresh.
    """
    sample = df[df["StarRating"] >= star_rating_thresh].sample(n, random_state=1)
    plot_with_annotations(sample.index)


def plot_by_title_pattern(pattern, **kwargs):
    """Plot all beatmaps whose titles match the given regex pattern.
    """
    match = df[df["Title"].str.contains(pattern)]
    return plot_with_annotations(match.index, **kwargs)


def add_annotations(ax, label_indices, emb_2d=None, **kwargs):
    if emb_2d is None:
        emb_2d = embs
    X = emb_2d[label_indices, 0]
    Y = emb_2d[label_indices, 1]
    ax.scatter(X, Y, **kwargs)


def plot_with_annotations(label_indices, text=True, labels=None, alpha=1, **kwargs):
    fig, ax = plot_bg(**kwargs)
    Xlabeled = embs[label_indices, 0]
    Ylabeled = embs[label_indices, 1]
    if labels is not None:
        for x, y, label in zip(Xlabeled, Ylabeled, labels):
            ax.scatter(x, y, alpha=alpha, label=label, marker='1',
                       s=90,
                       )
        fig.legend()
    else:
        ax.scatter(Xlabeled, Ylabeled, alpha=alpha, color='green')

    if text:
        titles = df.loc[label_indices, 'Title'].values
        texts = []
        for label, x, y in zip(titles, Xlabeled, Ylabeled):
            t = ax.annotate(label, xy=(x, y))
            texts.append(t)
        adjust_text(texts,
                    # expand_text=(1.01, 1.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    )
    return ax


FS = (13, 9)


def plot_region(x0, x1, y0, y1, text=True):
    """Plot the region of the mapping space bounded by the given x and y limits.
    """
    fig, ax = plt.subplots(figsize=FS)
    pts = df[
        (df.x >= x0) & (df.x <= x1)
        & (df.y >= y0) & (df.y <= y1)
        ]
    ax.scatter(pts.x, pts.y, alpha=.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if text:
        texts = []
        for label, label2, x, y in zip(pts["Title"].values, pts["Version"].values, pts.x.values, pts.y.values):
            t = ax.annotate(f"{label} [{label2}]", xy=(x, y))
            texts.append(t)
        adjust_text(texts, expand_text=(1.01, 1.05))
    return ax


def plot_region_around(beatmap_id, margin=5.0, **kwargs):
    """Plot the region of the mapping space in the neighbourhood of the beatmap with
    the given title. The margin parameter controls the size of the neighbourhood around
    the beatmap.
    """
    xmargin = ymargin = margin
    match = df[df["BeatmapID"] == beatmap_id]
    assert len(match) == 1
    row = match.iloc[0]
    return plot_region(row.x - xmargin, row.x + xmargin, row.y - ymargin, row.y + ymargin, **kwargs)


def plot_mappers(mappers):
    regex = "(?!\s?(de\s)?(it|that|" + '|'.join(mappers) + "))(((^|[^\S\r\n])(\S)*([sz]'|'s))|((^|[^\S\r\n])de\s(\S)*))"
    fig, ax = plot_bg(figsize=(16, 10))
    for i, mapper in enumerate(mappers):
        m = df[((df["Creator"] == mapper) | df["Version"].str.contains(mapper)) & ~df["Version"].str.contains(regex)]
        marker = str(i+1)
        add_annotations(ax, m.index, label=mapper, alpha=.5, marker=marker, s=150, linewidths=5)
    plt.legend()


df = pd.read_pickle("beatmap_df.pkl")
ckpt = torch.load("D:\\DiT-B-0130000.pt")
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"]

embs_file = "2d-embs.npy"
if os.path.isfile(embs_file):
    embs = np.load(embs_file)
else:
    # The default of 1,000 iterations gives fine results
    tsne = TSNE(random_state=1, n_iter=1000, metric="cosine")
    embs = tsne.fit_transform(embedding_table.cpu())[:41189]
    np.save(embs_file, embs)

df['x'] = embs[:, 0]
df['y'] = embs[:, 1]

# mappers = ['wafer', 'Sotarks', 'IOException']
plot_region(-31, -29, -50, -45)
plt.show()