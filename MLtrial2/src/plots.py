import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TARGET, OUT_DIR, MONTH_ORDER

FIG_DIR = OUT_DIR / "figures"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


def plot_missing(v):
    miss = v.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(miss.index, miss.values, color="#c0504d")
    for i, val in enumerate(miss.values):
        ax.text(val + 0.5, i, f"{val:.2f}%", va="center")
    ax.set_xlim(0, 105)
    ax.set_xlabel("% missing (after turning NULL/NA into missing)")
    ax.set_title("Missing values")
    _save(fig, "01_missing_values.png")


def plot_target(v):
    counts = v[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.bar(["Not canceled (0)", "Canceled (1)"], counts.values, color=["#4f81bd", "#c0504d"])
    for i, n in enumerate(counts.values):
        ax.text(i, n, f"{n:,}\n({n / len(v):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Target balance")
    _save(fig, "02_target_balance.png")


def plot_category_rates(v):
    cols = ["MarketSegment", "DepositType", "CustomerType",
            "DistributionChannel", "ArrivalDateMonth", "ArrivalDateYear"]
    overall = v[TARGET].mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), cols):
        g = v.groupby(c)[TARGET].agg(["mean", "size"])
        if c == "ArrivalDateMonth":
            g = g.reindex(MONTH_ORDER)
        ax.bar(g.index.astype(str), g["mean"], color="#4f81bd")
        ax.axhline(overall, color="k", ls="--", lw=1, label=f"overall {overall:.1%}")
        for i, (rate, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, rate, f"n={n:,}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"Cancel rate by {c}")
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right", fontsize=7)
    _save(fig, "03_cancel_rate_by_category.png")


def plot_leadtime(v):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for value, color, label in [(0, "#4f81bd", "Not canceled"), (1, "#c0504d", "Canceled")]:
        axes[0].hist(v.loc[v[TARGET] == value, "LeadTime"], bins=50, alpha=0.6,
                     color=color, label=label, density=True)
    axes[0].set_xlabel("LeadTime (days)")
    axes[0].set_ylabel("density")
    axes[0].set_title("LeadTime by outcome")
    axes[0].legend()
    bins = pd.cut(v.LeadTime, [-1, 7, 30, 90, 180, 365, 1000])
    rate = v.groupby(bins, observed=True)[TARGET].mean()
    axes[1].bar(rate.index.astype(str), rate.values, color="#8064a2")
    axes[1].set_ylabel("cancel rate")
    axes[1].set_title("Cancel rate by LeadTime bin")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, "04_leadtime_vs_target.png")


def plot_countries(v):
    g = (v.groupby(v.Country.fillna("Missing"))[TARGET].agg(["mean", "size"])
         .sort_values("size", ascending=False).head(12))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(g.index.astype(str), g["mean"], color="#9bbb59")
    ax.axhline(v[TARGET].mean(), color="k", ls="--", lw=1)
    ax.set_ylabel("cancel rate")
    ax.set_title("Cancel rate, 12 most frequent countries")
    _save(fig, "05_cancel_rate_top_countries.png")


NUMERIC_PLOT = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults",
                "Children", "PreviousCancellations", "PreviousBookingsNotCanceled"]


def plot_numeric_distributions(v):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, c in zip(axes.ravel(), NUMERIC_PLOT):
        x = v[c].dropna().astype(float)
        ax.hist(x, bins=40, color="#4bacc6")
        ax.set_yscale("log")
        ax.set_title(f"{c} (skew={x.skew():.1f})")
    axes.ravel()[-1].axis("off")
    _save(fig, "06_numeric_distributions.png")


def plot_outliers(v):
    cols = ["LeadTime", "StaysInWeekNights", "PreviousCancellations", "PreviousBookingsNotCanceled"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, c in zip(axes, cols):
        ax.boxplot(v[c].dropna().astype(float))
        ax.set_title(c)
    _save(fig, "07_outlier_boxplots.png")


def plot_correlation(v):
    d = v[["LeadTime", "ArrivalDateWeekNumber", "StaysInWeekendNights", "StaysInWeekNights",
           "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
           "PreviousBookingsNotCanceled", TARGET]].astype("float64")
    d["TotalNights"] = d.StaysInWeekendNights + d.StaysInWeekNights
    d["TotalGuests"] = d.Adults + d.Children.fillna(0) + d.Babies
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im)
    ax.set_title("Spearman correlation (booking-time numeric features + target)")
    _save(fig, "08_correlation_heatmap.png")


def plot_duplicates(v):
    cols = [c for c in v.columns if c != "RowID"]
    in_group = v.duplicated(subset=cols, keep=False)
    extra = v.duplicated(subset=cols, keep="first")
    g = v.groupby(in_group)[TARGET].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(["not duplicated", "in duplicate group"], g["mean"], color=["#4f81bd", "#c0504d"])
    for i, (rate, n) in enumerate(zip(g["mean"], g["size"])):
        axes[0].text(i, rate, f"{rate:.1%}\nn={n:,}", ha="center", va="bottom")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Cancel rate: duplicates vs the rest")
    both = [v[TARGET].mean(), v.loc[~extra, TARGET].mean()]
    axes[1].bar(["with duplicates", "extra copies removed"], both, color=["#8064a2", "#9bbb59"])
    for i, rate in enumerate(both):
        axes[1].text(i, rate, f"{rate:.1%}", ha="center", va="bottom")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Overall cancel rate")
    _save(fig, "09_duplicates_and_target.png")


def plot_leakage(v):
    status = v.groupby("ReservationStatus")[TARGET].mean()
    flags = {
        "AssignedRoomType != ReservedRoomType": v.AssignedRoomType != v.ReservedRoomType,
        "BookingChanges > 0": v.BookingChanges > 0,
        "DaysInWaitingList > 0": v.DaysInWaitingList > 0,
        "RequiredCarParkingSpaces > 0": v.RequiredCarParkingSpaces > 0,
        "TotalOfSpecialRequests > 0": v.TotalOfSpecialRequests > 0,
        "ADR == 0": v.ADR == 0,
    }
    names = list(flags)
    if_true = [v.loc[m, TARGET].mean() for m in flags.values()]
    if_false = [v.loc[~m, TARGET].mean() for m in flags.values()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 2]})
    axes[0].bar(status.index.astype(str), status.values, color="#c0504d")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Cancel rate by ReservationStatus\n(DIRECT leak)")
    y = range(len(names))
    axes[1].barh([i + 0.2 for i in y], if_true, height=0.4, color="#c0504d", label="condition TRUE")
    axes[1].barh([i - 0.2 for i in y], if_false, height=0.4, color="#4f81bd", label="condition FALSE")
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(names)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Post-booking columns: cancel rate when the condition is true vs false")
    axes[1].legend()
    _save(fig, "10_leakage_evidence.png")


def run_plots(v):
    for f in (plot_missing, plot_target, plot_category_rates, plot_leadtime, plot_countries,
              plot_numeric_distributions, plot_outliers, plot_correlation,
              plot_duplicates, plot_leakage):
        f(v)
    print(f"Figures saved in {FIG_DIR}")