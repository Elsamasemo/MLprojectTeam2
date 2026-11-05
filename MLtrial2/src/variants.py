import pandas as pd

from src.config import COUNTRY_THRESHOLDS, SEASONS, WEAK_GROUP


def learn_variant_params(train_base, params):
    """Country frequency lists - learned on the TRAIN rows only."""
    counts = train_base["Country"].value_counts()
    for t in COUNTRY_THRESHOLDS:
        params[f"country_keep_{t}"] = sorted(counts[counts >= t].index.tolist())
    return params


def _flag_replace(base, col, flag_name):
    out = base.copy()
    out[flag_name] = (out[col] != 0).astype("int64")
    return out.drop(columns=[col])


def _country_oth(base, keep_list):
    out = base.copy()
    out["Country"] = out["Country"].where(out["Country"].isin(keep_list), "oth")
    return out


def _with_season(base, drop_cols):
    out = base.copy()
    out["Season"] = out["ArrivalDateMonth"].map(SEASONS).astype("string")
    return out.drop(columns=drop_cols)


def _nights_capped(base):
    out = base.copy()
    out["StaysInWeekendNights"] = out["StaysInWeekendNights"].clip(upper=3)
    out["StaysInWeekNights"] = out["StaysInWeekNights"].clip(upper=6)
    out["TotalNights"] = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    return out


def make_variants(base, params):
    v = {"base": base.copy()}
    v["no_totals"] = base.drop(columns=["TotalNights", "TotalGuests"])
    v["nights_capped"] = _nights_capped(base)
    v["company_flag_drop"] = base.drop(columns=["HasCompany"])
    v["agent_drop"] = base.drop(columns=["Agent"])
    v["prevcancel_flag"] = _flag_replace(base, "PreviousCancellations", "HasPreviousCancellations")
    v["prevnotcanceled_flag"] = _flag_replace(base, "PreviousBookingsNotCanceled",
                                              "HasPreviousBookingsNotCanceled")
    for t in COUNTRY_THRESHOLDS:
        v[f"country_oth{t}"] = _country_oth(base, params[f"country_keep_{t}"])
    month_week_day = ["ArrivalDateMonth", "ArrivalDateDayOfMonth", "ArrivalDateWeekNumber"]
    v["season_no_year"] = _with_season(base, month_week_day + ["ArrivalDateYear"])
    v["season_keep_year"] = _with_season(base, month_week_day)
    v["drop_weak_group"] = base.drop(columns=WEAK_GROUP)
    return v


def variant_summary(variants):
    base_cols = set(variants["base"].columns)
    rows = [{"variant": name, "rows": len(d), "columns": d.shape[1],
             "added": sorted(set(d.columns) - base_cols),
             "dropped": sorted(base_cols - set(d.columns))}
            for name, d in variants.items()]
    return pd.DataFrame(rows)