import pandas as pd

df = pd.read_csv("data/H2.csv")
print(df.shape)
print(df.columns.tolist())

df = df.drop_duplicates()
df["Children"] = df["Children"].fillna(0)
print(df.shape)