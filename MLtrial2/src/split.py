import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (TARGET, SEED, TEST_SIZE, CANDIDATE_FEATURES,
                        DROP_EXACT_DUPLICATES)
from src.load import load_raw
from src.clean import remove_exact_duplicates, clean_base


def split_train_test(raw):
    """Stratified 80/20 split of the RAW rows (RowID is kept)."""
    train_raw, test_raw = train_test_split(
        raw, test_size=TEST_SIZE, stratify=raw[TARGET], random_state=SEED)
    return train_raw.reset_index(drop=True), test_raw.reset_index(drop=True)


def check_split(train_raw, test_raw):
    print("\n" + "=" * 70 + "\nSPLIT CHECK\n" + "=" * 70)
    print(f"Train: {len(train_raw):,} rows, cancel rate {train_raw[TARGET].mean():.4f}")
    print(f"Test : {len(test_raw):,} rows, cancel rate {test_raw[TARGET].mean():.4f}")
    print("RowIDs present in both parts:",
          len(set(train_raw["RowID"]) & set(test_raw["RowID"])))

    both = pd.concat([train_raw, test_raw]).drop(columns="RowID")
    print("Exact duplicate rows across train+test:", int(both.duplicated().sum()))


    rate = (train_raw.groupby(CANDIDATE_FEATURES)[TARGET].mean()
            .rename("train_rate").reset_index())
    twins = test_raw.merge(rate, on=CANDIDATE_FEATURES, how="inner")
    agree = ((twins["train_rate"] >= 0.5).astype(int) == twins[TARGET]).mean()
    print(f"Test rows with a twin in train: {len(twins):,} "
          f"({len(twins) / len(test_raw):.1%}); "
          f"twin's majority label matches in {agree:.1%} of them")


def prepare_data():

    raw = load_raw()
    log = [f"Loaded {len(raw):,} rows."]
    if DROP_EXACT_DUPLICATES:
        raw, n_removed = remove_exact_duplicates(raw)
        log.append(f"Removed {n_removed:,} exact duplicate rows -> {len(raw):,} rows.")

    train_raw, test_raw = split_train_test(raw)
    check_split(train_raw, test_raw)

    train_base, params, log_train = clean_base(train_raw, is_train=True)
    test_base, _, log_test = clean_base(test_raw, is_train=False, params=params)
    log += ["--- TRAIN ---"] + log_train + ["--- TEST ---"] + log_test
    return train_base, test_base, params, log