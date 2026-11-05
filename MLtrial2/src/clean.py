import pandas as pd

from src.config import (
    PLACEHOLDERS, MODE_COLUMNS, NO_AGENT, DIRECT_LEAKAGE, INDIRECT_LEAKAGE,
    BORDERLINE, DROP_NO_GUEST_ROWS, DROP_BORDERLINE,
)


def remove_exact_duplicates(raw):

    cols = [c for c in raw.columns if c != "RowID"]
    is_extra = raw.duplicated(subset=cols, keep="first")
    return raw.loc[~is_extra].copy(), int(is_extra.sum())


def clean_base(raw, is_train, params=None,
               drop_no_guest_rows=DROP_NO_GUEST_ROWS,
               drop_borderline=DROP_BORDERLINE):

    if "RowID" not in raw.columns:
        raise ValueError("Run load_raw first so RowID exists before cleaning.")
    if not is_train and params is None:
        raise ValueError("A validation/test file needs the params learned on the training file.")

    df = raw.copy()
    params = {} if params is None else params
    log = [f"Start ({'train' if is_train else 'validation/test'}): "
           f"{len(df)} rows, {df.shape[1]} columns"]


    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip()
    log.append(f"Trimmed spaces in {len(text_cols)} text columns.")


    for col, bad in PLACEHOLDERS.items():
        is_bad = df[col].str.upper().isin(bad)
        df[col] = df[col].mask(is_bad)
        log.append(f"{col}: {int(is_bad.sum())} placeholder value(s) set to missing.")


    df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0).astype("int64")
    log.append("Children: missing filled with 0 and converted to integer.")


    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])


    no_guests = (df["Adults"] + df["Children"] + df["Babies"]) == 0
    if is_train and drop_no_guest_rows:
        params["removed_row_ids"] = df.loc[no_guests, "RowID"].astype(int).tolist()
        df = df.loc[~no_guests].copy()
        log.append(f"Removed {int(no_guests.sum())} training rows with no guests at all.")
    else:
        log.append(f"No rows removed. This file has {int(no_guests.sum())} rows with no guests at all.")


    if "mode_MarketSegment" not in params:
        for c in MODE_COLUMNS:
            valid = df[c].dropna()
            valid = valid[valid.str.lower() != "undefined"]
            params[f"mode_{c}"] = str(valid.mode().iloc[0])
    for c in MODE_COLUMNS:
        bad = df[c].isna() | (df[c].str.lower() == "undefined")
        df.loc[bad, c] = params[f"mode_{c}"]
        log.append(f"{c}: {int(bad.sum())} missing/Undefined value(s) "
                   f"replaced by the training mode '{params[f'mode_{c}']}'.")


    df["Agent"] = df["Agent"].fillna(NO_AGENT)
    df["HasCompany"] = df["Company"].notna().astype("int64")
    log.append(f"Agent missing -> '{NO_AGENT}'; HasCompany flag created from Company.")

    # 8. Derived features (row by row, nothing learned from the data)
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    log.append("Created TotalNights and TotalGuests.")


    to_drop = (["Company", "Babies"] + DIRECT_LEAKAGE + INDIRECT_LEAKAGE
               + (BORDERLINE if drop_borderline else []))
    df = df.drop(columns=to_drop)
    log.append(f"Dropped columns: {to_drop}.")

    df = df.reset_index(drop=True)
    log.append(f"End: {len(df)} rows, {df.shape[1]} columns")
    return df, params, log


def validate_clean(df):
    problems = []
    if not df["RowID"].is_unique:
        problems.append("RowID is not unique")
    nan = df.isna().sum()
    nan = nan[nan > 0]
    if len(nan):
        problems.append(f"missing values: {nan.to_dict()}")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            left = int(df[c].dropna().astype("string").str.upper()
                       .isin({"NULL", "NA", "N/A", "NONE", "NAN", "", "UNDEFINED"}).sum())
            if left:
                problems.append(f"{c}: {left} placeholder-like value(s) left")
    forbidden = [c for c in DIRECT_LEAKAGE + INDIRECT_LEAKAGE + BORDERLINE if c in df.columns]
    if forbidden:
        problems.append(f"forbidden columns still present: {forbidden}")
    return problems