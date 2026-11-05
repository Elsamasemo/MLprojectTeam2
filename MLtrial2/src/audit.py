import pandas as pd

from src.config import (
    TARGET, EXPECTED_COLUMNS, PLACEHOLDERS, NULL_TOKENS, MONTH_ORDER, MONTH_NUMBER,
    NUMERIC_FOR_SUMMARY, CATEGORICAL_COLUMNS, CANDIDATE_FEATURES,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def tidy(df):

    v = df.copy()
    for c in v.select_dtypes(exclude="number").columns:
        v[c] = v[c].astype("string").str.strip()
    for col, bad in PLACEHOLDERS.items():
        v[col] = v[col].mask(v[col].str.upper().isin(bad))
    v["Children"] = pd.to_numeric(v["Children"])
    v["ReservationStatusDate"] = pd.to_datetime(v["ReservationStatusDate"])
    return v


def audit_structure(df):
    section("1. STRUCTURE")
    cols = [c for c in df.columns if c != "RowID"]
    print("Shape (without RowID):", (len(df), len(cols)))
    print("Missing expected columns:", sorted(set(EXPECTED_COLUMNS) - set(cols)))
    print("Unexpected columns:", sorted(set(cols) - set(EXPECTED_COLUMNS)))
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())
    text_cols = df.select_dtypes(exclude="number").columns
    padded = {c: int((df[c].astype(str) != df[c].astype(str).str.strip()).sum())
              for c in text_cols}
    print("\nValues with leading/trailing spaces:", {c: n for c, n in padded.items() if n})


def audit_missing(df):
    section("2. MISSING VALUES AND HIDDEN PLACEHOLDERS")
    actual = df.isna().sum()
    blank = pd.Series({c: int(df[c].astype(str).str.strip().eq("").sum())
                       for c in df.columns})
    hidden = pd.Series({c: int(df[c].astype(str).str.strip().str.upper()
                               .isin(NULL_TOKENS).sum()) for c in df.columns})
    out = pd.DataFrame({"Actual_Missing": actual, "Blank_Strings": blank,
                        "Hidden_Null_Tokens": hidden})
    out["Total_Suspicious"] = out.sum(axis=1)
    out["Percent"] = (out["Total_Suspicious"] / len(df) * 100).round(2)
    print(out[out["Total_Suspicious"] > 0].sort_values("Total_Suspicious", ascending=False))


def audit_categories(v):
    section("3. CATEGORICAL COLUMNS")
    for c in CATEGORICAL_COLUMNS:
        counts = v[c].value_counts(dropna=False)
        print(f"\n{c}: {v[c].nunique()} distinct (+{int(v[c].isna().sum())} missing)")
        print(counts.head(6).to_string())


def audit_target(v):
    section("4. TARGET (IsCanceled)")
    y = pd.to_numeric(v[TARGET], errors="coerce")
    print("Invalid (not 0/1):", int((~y.isin([0, 1])).sum()))
    print(pd.DataFrame({"rows": y.value_counts(),
                        "percent": (y.value_counts(normalize=True) * 100).round(2)}))
    print("\nCancel rate by arrival year:")
    print(v.groupby("ArrivalDateYear")[TARGET].agg(["size", "mean"]).round(3))
    print("\nCancel rate by arrival month:")
    print(v.groupby("ArrivalDateMonth")[TARGET].agg(["size", "mean"])
          .reindex(MONTH_ORDER).round(3))


def audit_duplicates(v):
    section("5. DUPLICATES")
    cols = [c for c in v.columns if c != "RowID"]
    extra = v.duplicated(subset=cols, keep="first")
    in_group = v.duplicated(subset=cols, keep=False)
    print("Extra duplicate copies:", int(extra.sum()), f"({extra.mean():.1%})")
    print("Rows inside a duplicate group:", int(in_group.sum()))
    print("Rows left if every extra copy is removed:", int((~extra).sum()))
    print("\nCancel rate inside vs outside duplicate groups:")
    print(v.groupby(in_group)[TARGET].agg(["size", "mean"])
          .rename(index={False: "not duplicated", True: "in duplicate group"}).round(3))
    print("\nOverall cancel rate WITH duplicates:", round(v[TARGET].mean(), 4),
          "| WITHOUT extra copies:", round(v.loc[~extra, TARGET].mean(), 4))
    sizes = v.groupby(cols, dropna=False).size().sort_values(ascending=False)
    print("\nFive biggest groups of identical rows (rows each):", sizes.head(5).tolist())


def audit_twins(v):
    section("6. TWINS: identical on every booking-time feature")
    g = v.groupby(CANDIDATE_FEATURES, dropna=False)[TARGET].agg(["size", "nunique"])
    print("Groups:", len(g))
    print("Rows in groups with 2+ rows:", int(g.loc[g["size"] > 1, "size"].sum()))
    print("Groups with conflicting labels:", int((g["nunique"] > 1).sum()),
          "covering", int(g.loc[g["nunique"] > 1, "size"].sum()), "rows")


def audit_guests(v):
    section("7. GUESTS")
    guests = v.Adults + v.Children.fillna(0) + v.Babies
    masks = {
        "zero guests": guests == 0,
        "no adults (adults == 0)": v.Adults == 0,
        "children/babies but no adults": (v.Adults == 0) & (guests > 0),
        "negative guest count": (v.Adults < 0) | (v.Children < 0) | (v.Babies < 0),
    }
    rows = [{"check": k, "rows": int(m.sum()),
             "cancel_rate": round(v.loc[m, TARGET].mean(), 3) if m.any() else None}
            for k, m in masks.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def audit_nights_adr(v):
    section("8. NIGHTS AND ADR")
    nights = v.StaysInWeekendNights + v.StaysInWeekNights
    masks = {
        "zero total nights": nights == 0,
        "ADR == 0": v.ADR == 0,
        "ADR < 0": v.ADR < 0,
        "ADR > 1000": v.ADR > 1000,
        "zero nights AND ADR == 0": (nights == 0) & (v.ADR == 0),
    }
    rows = [{"check": k, "rows": int(m.sum()),
             "cancel_rate": round(v.loc[m, TARGET].mean(), 3) if m.any() else None}
            for k, m in masks.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def audit_numeric(v):
    section("9. NUMERIC COLUMNS: range, skew, outliers")
    rows = []
    for c in NUMERIC_FOR_SUMMARY:
        x = v[c]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        rows.append({
            "column": c, "min": x.min(), "median": x.median(), "max": x.max(),
            "skew": round(x.skew(), 1), "negatives": int((x < 0).sum()),
            "IQR_outliers": int((x > q3 + 1.5 * iqr).sum() + (x < q1 - 1.5 * iqr).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nUnparseable ReservationStatusDate:", int(v["ReservationStatusDate"].isna().sum()))


def audit_target_relationships(v):
    section("10. WHAT RELATES TO THE TARGET")
    for c in ["MarketSegment", "DistributionChannel", "DepositType", "CustomerType",
              "Meal", "ReservedRoomType", "IsRepeatedGuest"]:
        print(f"\nCancel rate by {c}:")
        print(v.groupby(c)[TARGET].agg(["size", "mean"]).round(3).T.to_string())
    for c in ["Country", "Agent"]:
        print(f"\nCancel rate, 8 most frequent {c} values:")
        print(v.groupby(c)[TARGET].agg(["size", "mean"])
              .sort_values("size", ascending=False).head(8).round(3).T.to_string())
    bins = pd.cut(v.LeadTime, [-1, 7, 30, 90, 180, 365, 1000])
    print("\nCancel rate by LeadTime bin:")
    print(v.groupby(bins, observed=True)[TARGET].agg(["size", "mean"]).round(3).T.to_string())
    num = NUMERIC_FOR_SUMMARY + ["IsRepeatedGuest"]
    print("\nSpearman correlation with the target:")
    print(v[num + [TARGET]].corr(method="spearman")[TARGET]
          .drop(TARGET).round(3).sort_values().to_string())


def audit_leakage(v):
    section("11. LEAKAGE EVIDENCE")
    print("ReservationStatus vs IsCanceled:")
    print(pd.crosstab(v.ReservationStatus, v[TARGET]))

    arrival = pd.to_datetime(dict(
        year=v.ArrivalDateYear,
        month=v.ArrivalDateMonth.map(MONTH_NUMBER).astype("int64"),
        day=v.ArrivalDateDayOfMonth))
    days = (v.ReservationStatusDate - arrival).dt.days
    print("\nDays between arrival date and ReservationStatusDate, by status:")
    print(days.groupby(v.ReservationStatus).describe().round(1).to_string())

    checks = {
        "AssignedRoomType != ReservedRoomType": v.AssignedRoomType != v.ReservedRoomType,
        "BookingChanges > 0": v.BookingChanges > 0,
        "DaysInWaitingList > 0": v.DaysInWaitingList > 0,
        "RequiredCarParkingSpaces > 0": v.RequiredCarParkingSpaces > 0,
        "TotalOfSpecialRequests > 0": v.TotalOfSpecialRequests > 0,
        "ADR == 0": v.ADR == 0,
    }
    rows = [{"check": n, "rows": int(m.sum()),
             "cancel_if_true": round(v.loc[m, TARGET].mean(), 3),
             "cancel_if_false": round(v.loc[~m, TARGET].mean(), 3)}
            for n, m in checks.items()]
    print("\nPost-booking columns:")
    print(pd.DataFrame(rows).to_string(index=False))


def audit_missing_vs_target(v):
    section("12. IS 'MISSING' INFORMATIVE?")
    rows = []
    for c in ["Children", "Country", "Agent", "Company"]:
        m = v[c].isna()
        rows.append({"column": c, "missing_rows": int(m.sum()),
                     "cancel_if_missing": round(v.loc[m, TARGET].mean(), 3),
                     "cancel_if_present": round(v.loc[~m, TARGET].mean(), 3)})
    for c in ["MarketSegment", "DistributionChannel"]:
        m = v[c] == "Undefined"
        rows.append({"column": c + " == Undefined", "missing_rows": int(m.sum()),
                     "cancel_if_missing": round(v.loc[m, TARGET].mean(), 3),
                     "cancel_if_present": round(v.loc[~m, TARGET].mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


def run_audit(raw):
    audit_structure(raw)
    audit_missing(raw)
    v = tidy(raw)                       # cleaned copy for everything below
    for step in (audit_categories, audit_target, audit_duplicates, audit_twins,
                 audit_guests, audit_nights_adr, audit_numeric,
                 audit_target_relationships, audit_leakage, audit_missing_vs_target):
        step(v)