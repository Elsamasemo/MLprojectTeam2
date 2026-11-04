from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# 1. Load and clean
ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "data" / "H2.csv")
df = df.drop_duplicates()
df["Children"] = df["Children"].fillna(0)
df = df[df["ADR"] < 1000]
print(df.shape)

# 2. Choose columns, encode, scale
num_cols = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
            "Adults", "Children", "Babies", "ADR", "TotalOfSpecialRequests"]
cat_cols = ["MarketSegment", "DepositType", "CustomerType"]
X = df[num_cols + cat_cols]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
X_ready = prep.fit_transform(X)

# 3. Try k = 2 to 8
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_ready)
    sil = silhouette_score(X_ready, labels, sample_size=10000, random_state=42)
    db = davies_bouldin_score(X_ready, labels)
    print(k, round(sil, 3), round(db, 3))

# 4. Describe the groups for k = 5
km = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_ready)
print(df["cluster"].value_counts().sort_index())
print(df.groupby("cluster")[num_cols].mean().round(1))
print(df.groupby("cluster")[cat_cols].agg(lambda s: s.mode()[0]))