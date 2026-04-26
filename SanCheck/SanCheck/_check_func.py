from . import _helper as Help
import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================
# Distribution analysis
# =============================
def normalized_entropy(series: pd.Series, eps: float, bins):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) < 2:
        return 0.0

    if np.all(vals == vals[0]):
        return 0.0

    try:
        hist, edges = np.histogram(vals, bins=bins)
    except Exception:
        hist, edges = np.histogram(vals, bins=min(10, max(2, int(np.sqrt(len(vals))))))

    total = hist.sum()
    if total <= 0:
        return 0.0

    probs = hist / total
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0

    H = -(probs * np.log2(probs)).sum()
    H_max = math.log2(len(hist)) if len(hist) > 1 else 1.0
    return float(np.clip(H / max(H_max, eps), 0.0, 1.0))


def entropy_interpretation(score: float) -> str:
    if score < 0.25:
        return "very concentrated / single value or dominant mode"
    if score < 0.50:
        return "fairly concentrated / some structural dominance"
    if score < 0.75:
        return "mixed / moderate spread"
    return "very spread / more uniform or complex distribution"


def normalized_spread_score(series: pd.Series, eps):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0, 0.0, 0.0

    var = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = float(q75 - q25)
    robust_sigma = iqr / 1.349 if iqr > 0 else float(np.std(vals, ddof=1))
    baseline = robust_sigma ** 2

    score = var / (var + baseline + eps)
    score = float(np.clip(score, 0.0, 1.0))
    return score, var, iqr


def spread_interpretation(score: float) -> str:
    if score < 0.25:
        return "compact / small variation"
    if score < 0.50:
        return "moderate / moderate variation"
    if score < 0.75:
        return "wide / large variation"
    return "very wide / data very spread"


def distribution_report(df: pd.DataFrame, numeric_cols: list[str], eps: float, bins):
    rows = []
    for c in numeric_cols:
        ent = normalized_entropy(df[c], eps, bins)
        spread_score, raw_var, iqr = normalized_spread_score(df[c], eps)
        rows.append({
            "column": c,
            "entropy": ent,
            "entropy_label": entropy_interpretation(ent),
            "spread_score": spread_score,
            "spread_label": spread_interpretation(spread_score),
            "variance": raw_var,
            "iqr": iqr,
        })
    return pd.DataFrame(rows)

def class_override_ratio(df: pd.DataFrame, numeric_cols: list[str], target: str):
    sub = df[numeric_cols + [target]].dropna()

    grouped = sub.groupby(numeric_cols)[target].nunique()

    conflict = (grouped > 1).sum()
    total = len(grouped)

    if total == 0:
        return 0.0

    return conflict / total

# =============================
# Column problems
# =============================
def nan_inf_column_report(df: pd.DataFrame, numeric_cols: list[str]):
    rows = []
    for c in numeric_cols:
        s = df[c]
        coerced, finite_mask, nan_mask, bad_parse_mask = Help._to_numeric_with_mask(s)

        total = len(s)
        non_null = int(s.notna().sum())
        invalid_total = int((~finite_mask).sum())  # nan + inf setelah coercion
        inf_total = int(np.isinf(coerced.to_numpy(dtype="float64", copy=False)).sum()) if non_null else 0
        nan_total = int(coerced.isna().sum())
        bad_parse_total = int(bad_parse_mask.sum())

        severity = invalid_total / max(total, 1)

        rows.append({
            "column": c,
            "total": total,
            "non_null": non_null,
            "nan_total": nan_total,
            "inf_total": inf_total,
            "bad_parse_total": bad_parse_total,
            "invalid_total": invalid_total,
            "invalid_ratio": float(severity),
        })
    return pd.DataFrame(rows)


def inconsistent_type_report(df: pd.DataFrame, numeric_cols: list[str], thresh: float = 0.05):
    rows = []
    for c in numeric_cols:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        bad = coerced.isna() & s.notna()
        ratio = float(bad.mean())
        rows.append({
            "column": c,
            "bad_type_total": int(bad.sum()),
            "bad_type_ratio": ratio,
            "flagged": ratio > thresh,
        })
    return pd.DataFrame(rows)


def abnormal_similarity_report(df: pd.DataFrame, 
                               numeric_cols: list[str], 
                               threshold: float,
                               eps: float):
    if len(numeric_cols) < 2:
        return pd.DataFrame(), [], 0.0

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()

    pairs = []
    flagged_cols = set()
    over_threshold_scores = []

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs_df = corr.where(mask).stack().reset_index()
    pairs_df.columns = ["col_a", "col_b", "abs_corr"]
    pairs = pairs_df[pairs_df["abs_corr"] >= threshold][["col_a", "col_b", "abs_corr"]].values.tolist()
    flagged_cols = set(pairs_df[pairs_df["abs_corr"] >= threshold][["col_a", "col_b"]].values.ravel())
    over_threshold_scores = pairs_df[pairs_df["abs_corr"] >= threshold]["abs_corr"].tolist()

    total_pairs = len(pairs)
    issue_pairs = sum(1 for _, _, v in pairs if v >= threshold)

    pair_ratio = issue_pairs / max(total_pairs, 1)
    excess_mean = float(np.mean(over_threshold_scores)) if over_threshold_scores else 0.0

    severity = float(np.clip(0.6 * pair_ratio + 0.4 * excess_mean, 0.0, 1.0))
    report = pd.DataFrame(pairs, columns=["col_a", "col_b", "abs_corr"]).sort_values("abs_corr", ascending=False)

    return report, sorted(flagged_cols), severity


# =============================
# Row problems
# =============================
def problematic_row_report(df: pd.DataFrame, numeric_cols: list[str], eps: float):
    if not numeric_cols:
        return pd.DataFrame(), pd.Series(dtype=float), 0.0

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(numeric.to_numpy(dtype="float64", copy=False))
    invalid_row_mask = invalid_mask.any(axis=1)

    col_scores = []
    for c in numeric_cols:
        vals = numeric[c]
        med = vals.median(skipna=True)
        mad = np.median(np.abs(vals.dropna() - med)) if vals.notna().any() else 0.0
        robust_z = (vals - med).abs() / (mad + eps)
        col_scores.append(robust_z / (robust_z + 1.0))

    if col_scores:
        score_df = pd.concat(col_scores, axis=1)
        row_scores = score_df.mean(axis=1, skipna=True).fillna(0.0)
    else:
        row_scores = pd.Series(np.zeros(len(df)), index=df.index)

    invalid_ratio = float(invalid_row_mask.mean()) if len(df) else 0.0
    anomaly_mean = float(row_scores.mean()) if len(row_scores) else 0.0
    severity = float(np.clip(0.7 * invalid_ratio + 0.3 * anomaly_mean, 0.0, 1.0))

    out = pd.DataFrame({
        "row_index": df.index,
        "has_invalid_numeric": invalid_row_mask,
        "row_anomaly_score": row_scores.values,
    })
    out = out.sort_values("row_anomaly_score", ascending=False)

    return out, row_scores, severity

# =============================
# VIF and sparsity
# =============================
def compute_vif(df: pd.DataFrame, numeric_cols: list[str]):
      df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
      if df.shape[1] < 2:
          return 0.0
      
      vif_scores = []
      for i in range(df.shape[1]):
          vif_scores.append(variance_inflation_factor(df.values, i))
      
      raw = np.mean(vif_scores)
      norm_vif = 1 - np.tanh(raw / 10)

      return {
        "mean": norm_vif,
        "per_feature": dict(zip(numeric_cols, vif_scores))
      }

def sparsity_ratio(df: pd.DataFrame, numeric_cols: list[str]):
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    n_samples, n_features = df.shape

    zero_ratio = np.sum(df == 0) / df.size
    dim_penalty = n_features / (n_samples + n_features)
    
    return 0.7 * zero_ratio + 0.3 * dim_penalty

# =============================
# Normality
# =============================
def shapiro_per_feature(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(vals)
    if n < 3:
        return 0.0

    if n <= 5000:
        try:
            _, p = stats.shapiro(vals)
            return float(p)
        except Exception:
            return 0.0

    rng = np.random.default_rng(42)
    sample = rng.choice(vals, size=5000, replace=False)
    try:
        _, p = stats.shapiro(sample)
        return float(p)
    except Exception:
        return 0.0


def ks_per_feature(series: pd.Series, eps: float):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) < 3:
        return 0.0

    std = np.std(vals)
    if std <= eps:
        return 1.0

    try:
        _, p = stats.kstest(vals, "norm", args=(np.mean(vals), std))
        return float(p)
    except Exception:
        return 0.0

def compute_normality(df: pd.DataFrame, numeric_cols: list[str]):
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        return 0.5

    skew_vals = []
    kurt_vals = []

    for col in df.columns:
        if df[col].nunique() > 1:
            skew_vals.append(abs(skew(df[col].dropna())))
            kurt_vals.append(abs(kurtosis(df[col].dropna())))

    skew_mean = np.mean(skew_vals) if skew_vals else 0.0
    kurt_mean = np.mean(kurt_vals) if kurt_vals else 0.0

    # Normalize (heuristic scaling)
    skew_score = np.tanh(skew_mean / 2)
    kurt_score = np.tanh(kurt_mean / 5)

    normality = 1 - (0.5 * skew_score + 0.5 * kurt_score)

    return float(np.clip(normality, 0.0, 1.0))