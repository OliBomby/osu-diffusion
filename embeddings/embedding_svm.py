import numpy as np
import pandas as pd
import torch
from sklearn import svm, metrics


def get_by_mapper(mapper):
    regex = "(?!\s?(?:de\s)?(?:it|that|" + mapper + "))(?:(?:(?:^|[^\S\r\n])(?:\S)*(?:[sz]'|'s))|(?:(?:^|[^\S\r\n])de\s(?:\S)*))"
    return df[((df["Source"] == mapper) | df["Difficulty"].str.contains(mapper)) & ~df["Difficulty"].str.contains(regex)]


def get_by_tag(tag):
    return df[df["omdb"].apply(lambda x: isinstance(x, list) and tag in x)]


def get_mappers_data(mappers, classes):
    x, y = list(map(np.concatenate, zip(*map(get_data, map(get_by_mapper, mappers), classes))))
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    return x, y

def get_tags_data(tags, classes):
    x, y = list(map(np.concatenate, zip(*map(get_data, map(get_by_tag, tags), classes))))
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    return x, y


def get_data(slice, clas):
    x = embedding_table[slice.index]
    y = np.array([clas] * len(slice))
    return x, y


def test_positive(mapper):
    asdf = get_by_mapper(mapper)
    X_test, y_test = get_data(asdf, 1)
    y_pred = clf.predict(X_test)
    asdf["pred"] = y_pred
    print(asdf[['Title', 'Difficulty', 'pred']])
    scores = []
    for i in range(len(tags)):
        other_tag = tags[i]
        clas = classes[i]
        y_test = np.array([clas] * len(y_test))
        scores.append(f"{metrics.accuracy_score(y_test, y_pred) * 100:.0f}% {other_tag}")
    print(f"{mapper} is {', '.join(scores)}")


def test_positive_tag(tag):
    X_test, y_test = get_data(get_by_tag(tag), 1)
    y_pred = clf.predict(X_test)
    scores = []
    for i in range(len(tags)):
        other_tag = tags[i]
        clas = classes[i]
        y_test = np.array([clas] * len(y_test))
        scores.append(f"{metrics.accuracy_score(y_test, y_pred) * 100:.0f}% {other_tag}")
    print(f"{tag} is {', '.join(scores)}")


beatmap_df = pd.read_pickle("beatmap_df.pkl")
tags_df = pd.read_csv("D:\\Osu! Dingen\\Beatmap ML Datasets\\omdb_tags.csv", names=["BeatmapID", "omdb"]).groupby(["BeatmapID"]).agg(list)
df = pd.merge(beatmap_df, tags_df, on="BeatmapID", how='left')
ckpt = torch.load("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\0240000.pt")
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"].cpu()


# tags = ['clean', 'messy']
tags = ['geometric', 'freeform']
# tags = ['sharp aim', 'wide aim', 'linear aim', 'aim control', 'flow aim']
classes = list(range(len(tags)))
X_train, y_train = get_tags_data(tags, classes)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# test_positive_tag("geometric")
# test_positive_tag("freeform")
# test_positive_tag("symmetrical")
# test_positive_tag("grid snap")
# test_positive_tag("hexgrid")
# test_positive_tag("clean")
# test_positive_tag("messy")
# test_positive_tag("jump aim")
# test_positive_tag("sharp aim")
# test_positive_tag("wide aim")
# test_positive_tag("linear aim")
# test_positive_tag("aim control")
# test_positive_tag("flow aim")
# test_positive("Kroytz")
# test_positive("Sotarks")
# test_positive("R3m")
# test_positive("SMOKELIND")
# test_positive("Nevo")
# test_positive("wafer")
# test_positive("IOException")
# test_positive("Kalibe")
# test_positive("Mizunashi Akari")
# test_positive("jasontime12345")
# test_positive("Akari")
# test_positive("Cheri")
# test_positive("Uberzolik")
# test_positive("ScubDomino")
# test_positive("momothx")
# test_positive("Heroine")
# test_positive("Some Hero")
test_positive("Venix")


# df['clean'] = clf.predict_proba(embedding_table[df.index])[:, 1]
# top1000 = df.nlargest(10000, 'clean')
# top1000.to_csv("../results/tags/clean_10000.csv")
