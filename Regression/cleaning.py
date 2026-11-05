"""
H2 Hotel Bookings - ADR Regression Data Preparation

Workflow:
1. Load raw data and create RowID.
2. Create GroupID before splitting.
3. Perform group-aware train/validation split.
4. Learn cleaning parameters from training only.
5. Apply the same cleaning to validation.
6. Generate feature-engineering variants.
7. Prepare three duplicate-handling experiments per variant:
   - Keep all rows.
   - One row per group with sample weights.
   - One row per group without sample weights.
8. Validate, analyze and save the prepared datasets.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path(r"C:\Users\malak\Downloads\H2.csv")
OUT_DIR = DATA_PATH.parent / "cleaned"

SAVE_VARIANTS = True
DROP_POST_OUTCOME = False

# None = run every variant. Or pass a list of names, e.g.
# ENABLED_VARIANTS = ["combined_best_features", "company_flag_total_nights_only"]
# "base" is always kept (the summary uses it as reference).
ENABLED_VARIANTS = None

ID_AND_TEXT_COLUMNS = ["Children", "Country", "Agent", "Company"]

PLACEHOLDERS = {
    "Children": {"NA"},
    "Country": {"NULL"},
    "Agent": {"NULL"},
    "Company": {"NULL"},
}

MODE_COLUMNS = ["MarketSegment", "DistributionChannel", "Country"]

CAPS = {"BookingChanges": (2, 3)}  # column: (limit, replacement value)

NO_AGENT = "No Agent"
NO_COMPANY = "No Company"

COUNTRY_THRESHOLDS = [100, 1000]

SEASONS = {
    "January": "spring", "February": "spring", "March": "spring",
    "April": "summer", "May": "summer", "June": "summer",
    "July": "fall", "August": "fall", "September": "fall",
    "October": "winter", "November": "winter", "December": "winter",
}

WEAK_GROUP = [
    "ArrivalDateDayOfMonth",
    "ArrivalDateWeekNumber",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
]

NIGHT_COLS = ["StaysInWeekendNights", "StaysInWeekNights"]

NON_FEATURE_COLS = ["RowID", "GroupID", "ADR"]


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

def load_raw(path):
    raw = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        dtype={c: str for c in ID_AND_TEXT_COLUMNS},
    )
    raw.insert(0, "RowID", range(len(raw)))
    return raw


# ============================================================
# 3. CREATE GROUP ID
# ============================================================

def add_group_id(raw):
    """Rows with identical original features (excluding RowID and ADR) share a GroupID."""
    features = raw.drop(columns=["RowID", "ADR"]).copy()

    for c in features.columns:
        if not pd.api.types.is_numeric_dtype(features[c]):
            features[c] = features[c].astype("string").str.strip()

    raw = raw.copy()
    raw["GroupID"] = pd.factorize(
        pd.MultiIndex.from_frame(features), sort=False
    )[0]
    return raw


# ============================================================
# 4. GROUP-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

def group_train_val_split(raw, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(raw, groups=raw["GroupID"]))
    return raw.iloc[train_idx].copy(), raw.iloc[val_idx].copy()


# ============================================================
# 5. BASE CLEANING
# ============================================================

def clean_base(raw, is_train, params=None):
    if "RowID" not in raw.columns:
        raise ValueError("RowID is required.")
    if not is_train and params is None:
        raise ValueError("Training parameters are required.")

    df = raw.copy()
    params = {} if params is None else params
    log = [f"Start: {len(df)} rows, {df.shape[1]} columns."]

    # 5.1 Trim spaces in text columns.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    log.append("Trimmed spaces in text columns.")

    # 5.2 Convert placeholders into missing values.
    for col, bad in PLACEHOLDERS.items():
        if col in df.columns:
            mask = df[col].str.upper().isin(bad)
            df[col] = df[col].mask(mask)
            log.append(f"{col}: {int(mask.sum())} placeholders replaced.")

    # 5.3 Children -> integer, missing -> 0.
    df["Children"] = (
        pd.to_numeric(df["Children"], errors="coerce").fillna(0).astype("int64")
    )
    log.append("Children: missing values filled with zero.")

    # 5.4 Remove bad rows from training only.
    if is_train:
        drop = (df["Adults"] == 0) | (df["ADR"] == 5400)
        params["removed_row_ids"] = df.loc[drop, "RowID"].astype(int).tolist()
        log.append(f"Removed {int(drop.sum())} training rows.")
        df = df.loc[~drop].copy()
    else:
        log.append("Validation: no rows removed.")

    # 5.5 Learn modes from training only, reuse them for validation.
    if is_train:
        for c in MODE_COLUMNS:
            valid = df[c].dropna()
            valid = valid[valid.str.lower() != "undefined"]
            if valid.empty:
                raise ValueError(f"No valid values available to learn mode for {c}.")
            params[f"mode_{c}"] = str(valid.mode().iloc[0])

    for c in MODE_COLUMNS:
        bad = df[c].isna() | (df[c].str.lower() == "undefined")
        df.loc[bad, c] = params[f"mode_{c}"]
        log.append(f"{c}: missing values replaced using training mode.")

    # 5.6 Missing Agent / Company.
    df["Agent"] = df["Agent"].fillna(NO_AGENT)
    df["Company"] = df["Company"].fillna(NO_COMPANY)
    log.append("Agent and Company missing values replaced.")

    # 5.7 Capping.
    for col, (limit, new_value) in CAPS.items():
        count = int((df[col] > limit).sum())
        df[col] = df[col].where(df[col] <= limit, new_value)
        log.append(f"{col}: {count} values capped.")

    # 5.8 Drop Babies.
    df = df.drop(columns=["Babies"])
    log.append("Dropped Babies.")

    # 5.9 Optionally drop post-outcome columns.
    if DROP_POST_OUTCOME:
        gone = [
            c for c in ["ReservationStatus", "ReservationStatusDate"]
            if c in df.columns
        ]
        df = df.drop(columns=gone)
        log.append(f"Dropped post-outcome columns: {gone}")

    df = df.reset_index(drop=True)
    log.append(f"End: {len(df)} rows, {df.shape[1]} columns.")
    return df, params, log


# ============================================================
# 6. LEARN VARIANT PARAMETERS (training only)
# ============================================================

def learn_variant_params(base_train, params):
    counts = base_train["Country"].value_counts()
    for t in COUNTRY_THRESHOLDS:
        params[f"country_keep_{t}"] = sorted(counts[counts >= t].index.tolist())
    return params


# ============================================================
# 7. FEATURE-ENGINEERING BUILDING BLOCKS
# ============================================================

def flag_pair(base, col, flag_name, flag_rule):
    """Return (flag version: col replaced by binary flag, drop version: col removed)."""
    flagged = base.copy()
    flagged[flag_name] = flag_rule(base).astype("int64")
    flagged = flagged.drop(columns=[col])
    dropped = base.drop(columns=[col])
    return flagged, dropped


def country_oth(base, keep_list):
    out = base.copy()
    out["Country"] = out["Country"].where(out["Country"].isin(keep_list), "oth")
    return out


def with_season(base, drop_cols):
    out = base.copy()
    season = out["ArrivalDateMonth"].map(SEASONS)
    if season.isna().any():
        raise ValueError("Unknown month names found.")
    out["Season"] = season.astype("string")
    return out.drop(columns=drop_cols)


def make_nights_variants(base):
    """Capped weekend/week nights, and TotalNights (original columns kept)."""
    nights_capped = base.copy()
    nights_capped["StaysInWeekendNights"] = nights_capped["StaysInWeekendNights"].clip(upper=3)
    nights_capped["StaysInWeekNights"] = nights_capped["StaysInWeekNights"].clip(upper=6)

    total_nights = base.copy()
    total_nights["TotalNights"] = (
        total_nights["StaysInWeekendNights"] + total_nights["StaysInWeekNights"]
    )
    return nights_capped, total_nights


def make_company_nights_variants(company_flag_df):
    """HasCompany + TotalNights, with and without the two original nights columns."""
    keep_cols = company_flag_df.copy()
    keep_cols["TotalNights"] = (
        keep_cols["StaysInWeekendNights"] + keep_cols["StaysInWeekNights"]
    )
    total_only = keep_cols.drop(columns=NIGHT_COLS)
    return keep_cols, total_only


def make_combined_variant(base, country_keep):
    """Combination of the best individual features."""
    out = base.copy()

    # Company -> HasCompany.
    out["HasCompany"] = (out["Company"] != NO_COMPANY).astype("int64")
    out = out.drop(columns=["Company"])

    # Add TotalNights, keep the original nights columns.
    out["TotalNights"] = out["StaysInWeekendNights"] + out["StaysInWeekNights"]

    # Drop previous-booking history columns.
    out = out.drop(columns=["PreviousCancellations", "PreviousBookingsNotCanceled"])

    # RequiredCarParkingSpaces -> HasRequiredCarParkingSpaces.
    out["HasRequiredCarParkingSpaces"] = (
        out["RequiredCarParkingSpaces"] != 0
    ).astype("int64")
    out = out.drop(columns=["RequiredCarParkingSpaces"])

    # Group rare countries (threshold 1000, learned on training).
    out["Country"] = out["Country"].where(out["Country"].isin(country_keep), "oth")
    return out


# ============================================================
# 8. GENERATE ALL FEATURE VARIANTS
# ============================================================

def make_variants(base, params):
    v = {"base": base.copy()}

    # Nights.
    v["nights_capped"], v["total_nights"] = make_nights_variants(base)

    # Company.
    v["company_flag"], v["company_drop"] = flag_pair(
        base, "Company", "HasCompany",
        lambda d: d["Company"] != NO_COMPANY,
    )

    # Company flag + total nights.
    (
        v["company_flag_total_nights_keep"],
        v["company_flag_total_nights_only"],
    ) = make_company_nights_variants(v["company_flag"])

    # Parking.
    v["parking_flag"], v["parking_drop"] = flag_pair(
        base, "RequiredCarParkingSpaces", "HasRequiredCarParkingSpaces",
        lambda d: d["RequiredCarParkingSpaces"] != 0,
    )

    # Previous cancellations.
    v["prevcancel_flag"], v["prevcancel_drop"] = flag_pair(
        base, "PreviousCancellations", "HasPreviousCancellations",
        lambda d: d["PreviousCancellations"] != 0,
    )

    # Previous bookings not canceled.
    v["prevnotcanceled_flag"], v["prevnotcanceled_drop"] = flag_pair(
        base, "PreviousBookingsNotCanceled", "HasPreviousBookingsNotCanceled",
        lambda d: d["PreviousBookingsNotCanceled"] != 0,
    )

    # Country grouping.
    for t in COUNTRY_THRESHOLDS:
        v[f"country_oth{t}"] = country_oth(base, params[f"country_keep_{t}"])

    # Season.
    v["season_no_year"] = with_season(
        base,
        ["ArrivalDateMonth", "ArrivalDateDayOfMonth",
         "ArrivalDateWeekNumber", "ArrivalDateYear"],
    )
    v["season_keep_year"] = with_season(
        base,
        ["ArrivalDateMonth", "ArrivalDateDayOfMonth", "ArrivalDateWeekNumber"],
    )

    # Waiting list.
    v["waitlist_flag"], v["waitlist_drop"] = flag_pair(
        base, "DaysInWaitingList", "HasDaysInWaitingList",
        lambda d: d["DaysInWaitingList"] != 0,
    )

    # Weak feature group.
    v["drop_weak_group"] = base.drop(columns=WEAK_GROUP, errors="ignore")

    # Combined best features.
    v["combined_best_features"] = make_combined_variant(
        base, params["country_keep_1000"]
    )

    # Optional filter (base is always kept).
    if ENABLED_VARIANTS is not None:
        unknown = set(ENABLED_VARIANTS) - set(v)
        if unknown:
            raise ValueError(f"Unknown variant names: {sorted(unknown)}")
        v = {k: df for k, df in v.items() if k == "base" or k in ENABLED_VARIANTS}

    return v


# ============================================================
# 9. BUILD ALL
# ============================================================

def build_all(raw, is_train, params=None):
    base, params, log = clean_base(raw, is_train, params)
    if is_train:
        params = learn_variant_params(base, params)
    variants = make_variants(base, params)
    return base, variants, params, log


# ============================================================
# 10. VALIDATE DATA
# ============================================================

def validate(df):
    problems = []

    if "RowID" in df.columns and not df["RowID"].is_unique:
        problems.append("RowID is not unique.")

    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        problems.append(f"Missing values: {nan_cols.to_dict()}")

    placeholders = {"NULL", "NA", "N/A", "NONE", "NAN", "", "UNDEFINED"}

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            count = (
                df[c].dropna().astype("string").str.upper().isin(placeholders).sum()
            )
            if count:
                problems.append(f"{c}: {int(count)} placeholder values.")

    return problems


# ============================================================
# 11. DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis(df, name):
    data = df.drop(columns=["RowID", "GroupID"], errors="ignore")
    duplicate_mask = data.duplicated(keep=False)

    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(data[duplicate_mask].drop_duplicates().shape[0])
    removable_rows = int(data.duplicated(keep="first").sum())

    return {
        "DataFrame": name,
        "Total Rows": len(df),
        "Duplicate Rows": duplicate_rows,
        "Duplicate Groups": duplicate_groups,
        "Rows Removable": removable_rows,
        "Duplicate Percentage": (
            round(duplicate_rows / len(df) * 100, 2) if len(df) else 0
        ),
    }


# ============================================================
# 12. ADR VARIATION ANALYSIS
# ============================================================

def adr_variation_analysis(df):
    features = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    duplicate_mask = features.duplicated(keep=False)

    duplicate_features = features.loc[duplicate_mask].copy()
    duplicate_features["ADR"] = df.loc[duplicate_mask, "ADR"].values

    adr_analysis = (
        duplicate_features
        .groupby(list(features.columns), dropna=False)["ADR"]
        .agg(
            count="count",
            unique_values="nunique",
            mean="mean",
            min="min",
            max="max",
            std="std",
        )
        .reset_index()
    )

    different_adr = adr_analysis[adr_analysis["unique_values"] > 1].copy()
    different_adr["adr_range"] = different_adr["max"] - different_adr["min"]

    groups_with_zero = different_adr[different_adr["min"] == 0]
    groups_without_zero = different_adr[different_adr["min"] > 0]

    return adr_analysis, different_adr, groups_with_zero, groups_without_zero


# ============================================================
# 13. DUPLICATE EXPERIMENTS
# ============================================================

def collapse_groups(df):
    """One row per identical feature group: mean ADR plus group size as SampleWeight."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    return (
        df.groupby(feature_cols, dropna=False, sort=False)
        .agg(ADR=("ADR", "mean"), SampleWeight=("ADR", "size"))
        .reset_index()
    )


def prepare_duplicate_experiments(train_df, val_df):
    train_collapsed = collapse_groups(train_df)

    experiments = {
        "keep_all": train_df.copy(),
        "one_row_weighted": train_collapsed.copy(),
        "one_row_unweighted": train_collapsed.drop(columns=["SampleWeight"]),
    }

    # Validation always keeps every original row.
    return experiments, val_df.copy()


# ============================================================
# 14. MODEL INPUTS
# ============================================================

def prepare_model_data(df, use_weights=False):
    y = df["ADR"].copy()
    X = df.drop(
        columns=["ADR", "RowID", "GroupID", "SampleWeight"], errors="ignore"
    ).copy()

    weights = None
    if use_weights:
        if "SampleWeight" not in df.columns:
            raise ValueError("SampleWeight column is missing.")
        weights = df["SampleWeight"].copy()

    return X, y, weights


# ============================================================
# 15. VARIANT SUMMARY
# ============================================================

def variant_summary(variants):
    base_cols = set(variants["base"].columns)

    rows = [
        {
            "Variant": name,
            "Rows": len(df),
            "Columns": df.shape[1],
            "Added": sorted(set(df.columns) - base_cols),
            "Dropped": sorted(base_cols - set(df.columns)),
            "Problems": validate(df) or "None",
        }
        for name, df in variants.items()
    ]
    return pd.DataFrame(rows)


# ============================================================
# 16. MAIN RUN
# ============================================================

if __name__ == "__main__":

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1: Load.
    raw = load_raw(DATA_PATH)
    print("Raw shape:", raw.shape)

    # STEP 2: GroupID.
    raw = add_group_id(raw)
    print("Number of groups:", raw["GroupID"].nunique())

    # STEP 3: Group-aware split.
    train_raw, val_raw = group_train_val_split(raw, test_size=0.2, random_state=42)
    print("\nRAW SPLIT")
    print("Training:", train_raw.shape)
    print("Validation:", val_raw.shape)

    # STEP 4: Clean training and learn parameters.
    base_train, variants_train, params, train_log = build_all(train_raw, is_train=True)

    # STEP 5: Clean validation with training parameters.
    base_val, variants_val, _, val_log = build_all(
        val_raw, is_train=False, params=params
    )
    print("\nCLEANED SPLIT")
    print("Training:", base_train.shape)
    print("Validation:", base_val.shape)

    # STEP 6: Group separation check.
    shared_groups = set(train_raw["GroupID"]) & set(val_raw["GroupID"])
    print("\nGroups shared between training and validation:", len(shared_groups))

    # STEP 7: Duplicate experiments for every variant.
    duplicate_experiments = {}
    validation_datasets = {}

    for name in variants_train:
        train_versions, validation = prepare_duplicate_experiments(
            variants_train[name], variants_val[name]
        )
        duplicate_experiments[name] = train_versions
        validation_datasets[name] = validation

    print("\nDUPLICATE EXPERIMENTS")
    for variant, experiments in duplicate_experiments.items():
        print(f"\n{variant}")
        for experiment, df in experiments.items():
            print(experiment, df.shape)
        print("Validation:", validation_datasets[variant].shape)

    # STEP 8: Validation rows preserved?
    print("\nVALIDATION ROW CHECK")
    for name in variants_val:
        same = set(variants_val[name]["RowID"]) == set(validation_datasets[name]["RowID"])
        print(name, "All validation rows preserved:", same)

    # STEP 9: Duplicate analysis.
    duplicates_df = pd.DataFrame(
        [duplicate_analysis(df, name) for name, df in variants_train.items()]
    )
    print("\nDUPLICATE ANALYSIS")
    print(duplicates_df.to_string(index=False))
    duplicates_df.to_csv(OUT_DIR / "duplicate_analysis.csv", index=False)

    # STEP 10: ADR variation analysis (base training data).
    (
        adr_analysis,
        different_adr,
        groups_with_zero,
        groups_without_zero,
    ) = adr_variation_analysis(base_train)

    print("\nADR VARIATION ANALYSIS")
    print("Repeated feature groups:", len(adr_analysis))
    print("Groups with different ADR:", len(different_adr))
    print("Groups containing ADR = 0:", len(groups_with_zero))
    print("Groups without ADR = 0:", len(groups_without_zero))
    print("\nADR range statistics:")
    print(different_adr["adr_range"].describe())

    adr_analysis.to_csv(OUT_DIR / "adr_variation_analysis.csv", index=False)

    # STEP 11: Variant summaries.
    print("\nTRAINING VARIANT SUMMARY")
    print(variant_summary(variants_train).to_string(index=False))

    print("\nVALIDATION VARIANT SUMMARY")
    print(variant_summary(variants_val).to_string(index=False))

    # STEP 12: Save cleaned base data, parameters and logs.
    base_train.to_pickle(OUT_DIR / "base_train.pkl")
    base_val.to_pickle(OUT_DIR / "base_validation.pkl")

    (OUT_DIR / "cleaning_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "cleaning_log.txt").write_text(
        "\n".join(train_log + val_log), encoding="utf-8"
    )

    # STEP 13: Save variants and experiments.
    if SAVE_VARIANTS:
        for name, df in variants_train.items():
            df.to_pickle(OUT_DIR / f"{name}_train.pkl")

        for name, df in variants_val.items():
            df.to_pickle(OUT_DIR / f"{name}_validation.pkl")

        for variant, experiments in duplicate_experiments.items():
            for experiment, df in experiments.items():
                df.to_pickle(OUT_DIR / f"{variant}_{experiment}.pkl")

    # STEP 14: Save validation datasets.
    for name, df in validation_datasets.items():
        df.to_pickle(OUT_DIR / f"{name}_validation_original.pkl")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nOutput directory:", OUT_DIR)