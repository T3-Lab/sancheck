import argparse
import math
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# =============================
# Konfigurasi
# =============================
NUMERIC_VALID_RATIO = 0.95
ENTROPY_BINS = "fd"
DEFAULT_SIM_THRESHOLD = 0.95
EPS = 1e-12


# =============================
# Helpers
# =============================
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


def get_numeric_valid_columns(df: pd.DataFrame, thresh: float = NUMERIC_VALID_RATIO):
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


# =============================
# Distribusi: entropy & spread
# =============================
def normalized_entropy(series: pd.Series, bins=ENTROPY_BINS):
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
    return float(np.clip(H / max(H_max, EPS), 0.0, 1.0))


def entropy_interpretation(score: float) -> str:
    if score < 0.25:
        return "very concentrated / single value or dominant mode"
    if score < 0.50:
        return "fairly concentrated / some structural dominance"
    if score < 0.75:
        return "mixed / moderate spread"
    return "very spread / more uniform or complex distribution"


def normalized_spread_score(series: pd.Series, eps=EPS):
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


def distribution_report(df: pd.DataFrame, numeric_cols: list[str]):
    rows = []
    for c in numeric_cols:
        ent = normalized_entropy(df[c])
        spread_score, raw_var, iqr = normalized_spread_score(df[c])
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


# =============================
# Masalah kolom
# =============================
def nan_inf_column_report(df: pd.DataFrame, numeric_cols: list[str]):
    rows = []
    for c in numeric_cols:
        s = df[c]
        coerced, finite_mask, nan_mask, bad_parse_mask = _to_numeric_with_mask(s)

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


def abnormal_similarity_report(df: pd.DataFrame, numeric_cols: list[str], threshold: float = DEFAULT_SIM_THRESHOLD):
    if len(numeric_cols) < 2:
        return pd.DataFrame(), [], 0.0

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()

    pairs = []
    flagged_cols = set()
    over_threshold_scores = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append((c1, c2, float(val)))
            if val >= threshold:
                flagged_cols.add(c1)
                flagged_cols.add(c2)
                # skala 0..1: seberapa jauh melewati threshold
                over_threshold_scores.append((val - threshold) / max(1.0 - threshold, EPS))

    total_pairs = len(pairs)
    issue_pairs = sum(1 for _, _, v in pairs if v >= threshold)

    pair_ratio = issue_pairs / max(total_pairs, 1)
    excess_mean = float(np.mean(over_threshold_scores)) if over_threshold_scores else 0.0

    severity = float(np.clip(0.6 * pair_ratio + 0.4 * excess_mean, 0.0, 1.0))
    report = pd.DataFrame(pairs, columns=["col_a", "col_b", "abs_corr"]).sort_values("abs_corr", ascending=False)

    return report, sorted(flagged_cols), severity


# =============================
# Masalah row
# =============================
def problematic_row_report(df: pd.DataFrame, numeric_cols: list[str]):
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
        robust_z = (vals - med).abs() / (mad + EPS)
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
# Clarity score
# =============================
@dataclass
class ClarityBreakdown:
    missing_severity: float
    type_severity: float
    similarity_severity: float
    row_severity: float

    @property
    def overall(self) -> float:
        weights = {
            "missing_severity": 0.30,
            "type_severity": 0.20,
            "similarity_severity": 0.20,
            "row_severity": 0.30,
        }
        penalty = (
            self.missing_severity * weights["missing_severity"]
            + self.type_severity * weights["type_severity"]
            + self.similarity_severity * weights["similarity_severity"]
            + self.row_severity * weights["row_severity"]
        )
        return float(np.clip(1.0 - penalty, 0.0, 1.0))

    @property
    def label(self) -> str:
        score = self.overall
        if score >= 0.85:
            return "very clean"
        if score >= 0.70:
            return "fairly clean"
        if score >= 0.50:
            return "some issues"
        return "dirty"


def clarity_breakdown(
    df: pd.DataFrame,
    nan_inf_df: pd.DataFrame,
    type_df: pd.DataFrame,
    sim_severity: float,
    row_severity: float,
) -> ClarityBreakdown:
    if len(nan_inf_df) == 0:
        missing_severity = 0.0
    else:
        missing_severity = float(np.clip(nan_inf_df["invalid_ratio"].mean(), 0.0, 1.0))

    if len(type_df) == 0:
        type_severity = 0.0
    else:
        type_severity = float(np.clip(type_df["bad_type_ratio"].mean(), 0.0, 1.0))

    return ClarityBreakdown(
        missing_severity=missing_severity,
        type_severity=type_severity,
        similarity_severity=float(np.clip(sim_severity, 0.0, 1.0)),
        row_severity=float(np.clip(row_severity, 0.0, 1.0)),
    )


# =============================
# Plotting
# =============================
def plot_numeric_boxplot(df: pd.DataFrame, numeric_cols: list[str], title_suffix: str = ""):
    if not numeric_cols:
        print("⚠️ No valid numeric columns for boxplot.")
        return

    plot_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    spread_df = distribution_report(df, numeric_cols).set_index("column")
    ordered_cols = spread_df.sort_values("spread_score", ascending=False).index.tolist()
    plot_df = plot_df[ordered_cols]

    fig_w = max(12, 0.6 * len(ordered_cols) + 4)
    fig_h = max(6, 0.35 * len(ordered_cols) + 2)

    plt.figure(figsize=(fig_w, fig_h))
    sns.boxplot(data=plot_df, orient="h", showfliers=True, linewidth=1)
    plt.title(f"Boxplot Numeric Columns {title_suffix}".strip())
    plt.xlabel("Value")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.show()


def plot_numeric_heatmap(df: pd.DataFrame, numeric_cols: list[str], title_suffix: str = ""):
    if len(numeric_cols) < 2:
        print("⚠️ Heatmap requires at least 2 numeric columns.")
        return

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    annot = len(numeric_cols) <= 12

    fig_size = max(8, 0.55 * len(numeric_cols) + 4)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annot,
        fmt=".2f" if annot else None,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Heatmap Correlation {title_suffix}".strip())
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plots(df: pd.DataFrame, n_slice):
    if df.empty:
        print("⚠️ No data available for plotting.")
        return

    if n_slice == "all":
        plot_numeric_boxplot(df, list(df.columns), title_suffix="(all features)")
        plot_numeric_heatmap(df, list(df.columns), title_suffix="(all features)")
        return

    n_slice = int(n_slice)
    cols = list(df.columns)

    for start in range(0, len(cols), n_slice):
        chunk_cols = cols[start:start + n_slice]
        chunk_df = df[chunk_cols]
        title_suffix = f"(features {start + 1}-{start + len(chunk_cols)})"
        plot_numeric_boxplot(chunk_df, chunk_cols, title_suffix=title_suffix)
        plot_numeric_heatmap(chunk_df, chunk_cols, title_suffix=title_suffix)


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


def ks_per_feature(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) < 3:
        return 0.0

    std = np.std(vals)
    if std <= EPS:
        return 1.0

    try:
        _, p = stats.kstest(vals, "norm", args=(np.mean(vals), std))
        return float(p)
    except Exception:
        return 0.0


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="SanCheck — data sanity checker"
    )
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument(
        "slice",
        type=parse_slice_arg,
        help="Number of columns per chunk for plotting, or 'all'",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ Failed to load the CSV file: {e}")
        sys.exit(1)

    numeric_cols = get_numeric_valid_columns(df)
    ignored_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        print("❌ No numeric columns found with sufficient valid numeric ratio.")
        print(f"Non-numeric columns / ignored: {', '.join(ignored_cols) if ignored_cols else '-'}")
        sys.exit(1)

    # Column report
    sim_pairs, sim_cols, sim_severity = abnormal_similarity_report(df, numeric_cols)
    nan_inf_df = nan_inf_column_report(df, numeric_cols)
    type_df = inconsistent_type_report(df, numeric_cols)
    row_df, row_scores, row_severity = problematic_row_report(df, numeric_cols)
    dist_df = distribution_report(df, numeric_cols)
    clarity = clarity_breakdown(df, nan_inf_df, type_df, sim_severity, row_severity)

    # plotting
    plots(df[numeric_cols], args.slice)

    # normality
    shapiro = {c: shapiro_per_feature(df[c]) for c in numeric_cols}
    ks = {c: ks_per_feature(df[c]) for c in numeric_cols}

    # ringkasan
    top_entropy = dist_df.sort_values("entropy", ascending=False).head(5)
    top_spread = dist_df.sort_values("spread_score", ascending=False).head(5)
    top_rows = row_df.head(5)

    print("\n📌 Summary of columns")
    print(f"- Valid numeric columns: {len(numeric_cols)}")
    print(f"- Ignored non-numeric columns: {len(ignored_cols)}")

    print("\n📌 Column problems")
    print(f"- Column with NaN/Inf/invalid: {len(nan_inf_df[nan_inf_df['invalid_ratio'] > 0])}")
    for _, r in nan_inf_df.sort_values("invalid_ratio", ascending=False).head(10).iterrows():
        print(
            f"  - {r['column']}: invalid={int(r['invalid_total'])}/{int(r['total'])} "
            f"({r['invalid_ratio']:.3f})"
        )

    print(f"- Type inconsistency column: {int(type_df['flagged'].sum())}")
    for _, r in type_df.sort_values("bad_type_ratio", ascending=False).head(10).iterrows():
        print(
            f"  - {r['column']}: bad_type={int(r['bad_type_total'])} "
            f"({r['bad_type_ratio']:.3f})"
        )

    if len(sim_pairs) > 0:
        print(f"- Similar feature pairs (|corr| >= {DEFAULT_SIM_THRESHOLD}):")
        print(f"  - Severity similarity: {sim_severity:.3f}")
        for _, r in sim_pairs.head(10).iterrows():
            print(f"  - {r['col_a']} <-> {r['col_b']}: |corr|={r['abs_corr']:.3f}")
        if sim_cols:
            print(f"  - Affected columns: {', '.join(sim_cols)}")
    else:
        print("- Similar feature pairs: none")

    print("\n📌 Row problems")
    invalid_row_count = int(row_df["has_invalid_numeric"].sum()) if len(row_df) else 0
    print(f"- Problematic rows (NaN/Inf): {invalid_row_count}/{len(df)}")
    print(f"- Severity row: {row_severity:.3f}")
    for _, r in top_rows.iterrows():
        print(
            f"  - row {r['row_index']}: score={r['row_anomaly_score']:.3f}, "
            f"invalid={bool(r['has_invalid_numeric'])}"
        )

    print("\n📌 Distribution / interpretation")
    print("- High entropy means the distribution is more even/complex; it's not automatically 'noise', it can also be multimodal.")
    print("- High spread score means the data is more dispersed robustly compared to its central tendency.")
    print("\n  Top entropy:")
    for _, r in top_entropy.iterrows():
        print(
            f"  - {r['column']}: entropy={r['entropy']:.3f} "
            f"({r['entropy_label']})"
        )

    print("\n  Top spread:")
    for _, r in top_spread.iterrows():
        print(
            f"  - {r['column']}: spread_score={r['spread_score']:.3f} "
            f"({r['spread_label']}), var={r['variance']:.3f}, iqr={r['iqr']:.3f}"
        )

    print("\n📌 Normality (per fitur)")
    for c in numeric_cols:
        print(f"- {c}: Shapiro={shapiro[c]:.3f} | KS={ks[c]:.3f}")

    print("\n🔨 Final status")
    print(f"- clarity score: {clarity.overall:.3f} / 1.000")
    print(f"- clarity label: {clarity.label}")
    print(f"- missing severity: {clarity.missing_severity:.3f}")
    print(f"- type severity: {clarity.type_severity:.3f}")
    print(f"- similarity severity: {clarity.similarity_severity:.3f}")
    print(f"- row severity: {clarity.row_severity:.3f}")

    print("\n📊 Dataset-level distribution summary")
    print(f"- avg entropy: {dist_df['entropy'].mean():.3f}")
    print(f"- avg spread score: {dist_df['spread_score'].mean():.3f}")


if __name__ == "__main__":
    main()