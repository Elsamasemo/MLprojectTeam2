import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.config import OUT_DIR, SEED, N_SPLITS, DECISION_THRESHOLD
from src.modeling import to_xy, build_pipeline, SCORING
from src.split import prepare_data

FIG_DIR = OUT_DIR / "figures"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


def run_baseline():

    OUT_DIR.mkdir(exist_ok=True)
    train_base, _test_base, _params, _log = prepare_data()     # test part is NOT used
    X, y = to_xy(train_base)
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1)   # all defaults
    pipe = build_pipeline(model, X)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    print("\n" + "=" * 70 + "\nRANDOM FOREST BASELINE (untuned)\n" + "=" * 70)
    print(f"Train rows: {len(X):,} | input columns: {X.shape[1]} | "
          f"cancel rate: {y.mean():.2%}")
    print(f"Reference: always predicting 'not canceled' gives accuracy {1 - y.mean():.4f}, F1 0")


    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    rows = []
    for m in SCORING:
        rows.append({"metric": m,
                     "cv_mean": res[f"test_{m}"].mean(),
                     "cv_std": res[f"test_{m}"].std(ddof=1),
                     "train_mean": res[f"train_{m}"].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    print("\nMetrics (5-fold CV on train; 'train_mean' = score on the folds it was fitted on):")
    print(table.round(4).to_string(index=False))
    print(f"Average fit time per fold: {res['fit_time'].mean():.1f}s")


    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    pred = (proba > DECISION_THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred)
    print("\nConfusion matrix (out-of-fold, threshold "
          f"{DECISION_THRESHOLD}):  rows = true, columns = predicted")
    print(pd.DataFrame(cm, index=["true not canceled", "true canceled"],
                       columns=["pred not canceled", "pred canceled"]))
    pd.DataFrame(cm, index=["true_0", "true_1"],
                 columns=["pred_0", "pred_1"]).to_csv(OUT_DIR / "baseline_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({v / cm[i].sum():.1%} of row)", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred not canceled", "Pred canceled"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True not canceled", "True canceled"])
    ax.set_title("Random Forest: confusion matrix\n(out-of-fold, train)")
    _save(fig, "baseline_confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fpr, tpr, _ = roc_curve(y, proba)
    axes[0].plot(fpr, tpr, label="Random Forest")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (out-of-fold)"); axes[0].legend()
    prec, rec, _ = precision_recall_curve(y, proba)
    axes[1].plot(rec, prec, label="Random Forest")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=1, label="no skill")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (out-of-fold)"); axes[1].legend()
    _save(fig, "baseline_roc_pr_curves.png")


    fitted = build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1), X).fit(X, y)
    names = [n.split("__", 1)[1] for n in fitted.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(fitted.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(OUT_DIR / "baseline_feature_importance.csv", header=["importance"])
    print("\nTop 10 feature importances:")
    print(imp.head(10).round(3).to_string())
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#4f81bd")
    ax.set_title("Random Forest: top-15 feature importances (train)")
    _save(fig, "baseline_feature_importance.png")

    print(f"\nSaved: baseline_metrics.csv, baseline_confusion_matrix.csv, "
          f"baseline_feature_importance.csv in {OUT_DIR}; 3 figures in {FIG_DIR}")