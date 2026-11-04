from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[2]  # the main project folder
df = pd.read_csv(ROOT / "data" / "H2.csv")
print(df.shape)
print(df.columns.tolist())

df = df.drop_duplicates()
df["Children"] = df["Children"].fillna(0)
df = df[df["ADR"] < 1000]
print(df.shape)

print(df["ADR"].describe())

num_cols = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
            "Adults", "Children", "Babies", "ADR", "TotalOfSpecialRequests"]
cat_cols = ["MarketSegment", "DepositType", "CustomerType"]

X = df[num_cols + cat_cols]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
X_ready = prep.fit_transform(X)
print(X_ready.shape)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_ready)
    sil = silhouette_score(X_ready, labels, sample_size=10000, random_state=42)
    db = davies_bouldin_score(X_ready, labels)
    print(k, round(sil, 3), round(db, 3))