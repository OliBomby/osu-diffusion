import numpy as np
import pandas as pd
import torch
from sklearn import svm, metrics


def get_by_mapper(mapper):
    regex = "(?!\s?(?:de\s)?(?:it|that|" + mapper + "))(?:(?:(?:^|[^\S\r\n])(?:\S)*(?:[sz]'|'s))|(?:(?:^|[^\S\r\n])de\s(?:\S)*))"
    return df[((df["Source"] == mapper) | df["Difficulty"].str.contains(mapper)) & ~df["Difficulty"].str.contains(regex)]


def get_many_data(mappers, classes):
    x, y = list(map(np.concatenate, zip(*map(get_data, mappers, classes))))
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    return x, y


def get_data(mapper, clas):
    positive_df = get_by_mapper(mapper)
    x = embedding_table[positive_df.index]
    y = np.array([clas] * len(positive_df))
    return x, y


def test_positive(mapper):
    X_test, y_test = get_data(mapper, 1)
    y_pred = clf.predict(X_test)
    print(f"{mapper} is {metrics.accuracy_score(y_test, y_pred) * 100:.0f}% Kroytz")


df = pd.read_pickle("beatmap_df.pkl")
ckpt = torch.load("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\0240000.pt")
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"].cpu()


X_train, y_train = get_many_data(["Kroytz", "Sotarks"], [1, 0])
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

test_positive("Kroytz")
test_positive("Sotarks")
test_positive("R3m")
test_positive("SMOKELIND")
test_positive("Nevo")
test_positive("wafer")
test_positive("IOException")

df['sotarksness'] = clf.predict_proba(embedding_table[df.index])[:, 0]
top1000 = df.nlargest(1000, 'sotarksness')
top1000.to_csv("sotarks_1000.csv")
