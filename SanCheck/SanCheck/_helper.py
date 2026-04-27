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


def _label_from_score(score: float, toplow=False) -> str:
    if toplow:
        if score >= 0.75:
            return "[green]very high[/green]"
        if score >= 0.50:
            return "[yellow]high[/yellow]"
        if score >= 0.25:
            return "[orange1]low[/orange1]"
        return "[red]very low[/red]"
    
    else:
        if score < 0.25:
            return "[green]low[/green]"
        if score < 0.50:
            return "[yellow]medium[/yellow]"
        if score < 0.75:
            return "[orange1]high[/orange1]"
        return "[red]very high[/red]"
    
class InfoAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import _info as Info
        Info.metrics()
        parser.exit()