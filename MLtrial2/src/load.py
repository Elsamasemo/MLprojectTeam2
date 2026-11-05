import pandas as pd

from src.config import DATA_PATH, EXPECTED_COLUMNS, ID_AND_TEXT_COLUMNS


def load_raw(path=DATA_PATH):
    raw = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        dtype={c: str for c in ID_AND_TEXT_COLUMNS},
    )
    raw.columns = raw.columns.str.strip()

    missing = set(EXPECTED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in the file: {sorted(missing)}")

    # RowID follows each row through every later step
    raw.insert(0, "RowID", range(len(raw)))
    return raw