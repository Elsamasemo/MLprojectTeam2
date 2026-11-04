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