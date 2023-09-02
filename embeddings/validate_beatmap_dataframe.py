import pandas as pd


df = pd.read_pickle("beatmap_df.pkl")
print("Number of unique beatmap IDs = %s" % df["BeatmapID"].nunique())
id_counts = df["BeatmapID"].value_counts()
duplicated = id_counts[id_counts > 1]
print("Duplicates:")
print(df[df["BeatmapID"].isin(duplicated.index)])
