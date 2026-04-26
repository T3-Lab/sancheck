import argparse
import pandas as pd
import numpy as np

def parse_slice_arg(value: str):
    value = str(value).strip().lower()
    if value == "all":
        return "all"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "slice must be a positive integer or 'all'"
        ) from exc
    if n <= 0:
        raise argparse.ArgumentTypeError("slice must be > 0")
    return n


def numeric_ratio(series: pd.Series) -> float:
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean())


def get_numeric_valid_columns(df: pd.DataFrame, thresh: float):
    return [c for c in df.columns if numeric_ratio(df[c]) >= thresh]


def _to_numeric_with_mask(series: pd.Series):
    original = series
    coerced = pd.to_numeric(original, errors="coerce")

    finite_mask = np.isfinite(coerced.to_numpy(dtype="float64", copy=False))
    nan_mask = coerced.isna()

    bad_parse_mask = coerced.isna() & original.notna()

    return coerced, finite_mask, nan_mask, bad_parse_mask


def _label_from_score(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "very high"