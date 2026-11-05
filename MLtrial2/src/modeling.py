import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET, SEED, N_SPLITS, OUT_DIR

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}


def to_xy(df):
    y = df[TARGET].astype("int64")
    X = df.drop(columns=["RowID", TARGET]).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object)
    return X, y


def build_pipeline(model, X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=100, sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }


def evaluate(pipe, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out = {m: res[f"test_{m}"].mean() for m in SCORING}
    out["f1_std"] = res["test_f1"].std(ddof=1)
    out["fit_time_s"] = res["fit_time"].mean()
    return out


def compare_variants(variants, model_names=None):
    models = get_models()
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}
    base_ids = variants["base"]["RowID"].tolist()
    rows = []
    for vname, df in variants.items():
        assert df["RowID"].tolist() == base_ids, "variants must keep the same rows in the same order"
        X, y = to_xy(df)
        for mname, model in models.items():
            scores = evaluate(build_pipeline(model, X), X, y)
            rows.append({"variant": vname, "model": mname, "n_input_columns": X.shape[1], **scores})
            print(f"{vname:22s} {mname:20s} F1={scores['f1']:.4f}  AUC={scores['roc_auc']:.4f}  "
                  f"PR-AUC={scores['pr_auc']:.4f}  ({scores['fit_time_s']:.1f}s/fold)")
    result = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    result.to_csv(OUT_DIR / "variant_comparison.csv", index=False)
    for mname in models:
        print(f"\n=== {mname}: variants ranked by F1 (mean of {N_SPLITS} folds) ===")
        t = result[result.model == mname].sort_values("f1", ascending=False)
        print(t[["variant", "n_input_columns", "f1", "f1_std", "roc_auc", "pr_auc",
                 "precision", "recall", "accuracy"]].round(4).to_string(index=False))
    return result