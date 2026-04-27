from . import _helper as Help
from . import _check_func as Check
from . import _plotting as PLT
from . import _info as Info

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import time

from rich import print

# =============================
# Configuration
# =============================
NUMERIC_VALID_RATIO = 0.95
ENTROPY_BINS = "fd"
DEFAULT_SIM_THRESHOLD = 0.95
EPS = 1e-12

# =============================
# Clarity score
# =============================
@dataclass
class CleanlinessBreakdown:
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
            return "[green]very clean[/green]"
        if score >= 0.70:
            return "[yellow]fairly clean[/yellow]"
        if score >= 0.50:
            return "[orange1]some issues[/orange1]"
        return "[red]dirty[/red]"


def cleanliness_breakdown(
    df: pd.DataFrame,
    nan_inf_df: pd.DataFrame,
    type_df: pd.DataFrame,
    sim_severity: float,
    row_severity: float,
) -> CleanlinessBreakdown:
    if len(nan_inf_df) == 0:
        missing_severity = 0.0
    else:
        missing_severity = float(np.clip(nan_inf_df["invalid_ratio"].mean(), 0.0, 1.0))

    if len(type_df) == 0:
        type_severity = 0.0
    else:
        type_severity = float(np.clip(type_df["bad_type_ratio"].mean(), 0.0, 1.0))

    return CleanlinessBreakdown(
        missing_severity=missing_severity,
        type_severity=type_severity,
        similarity_severity=float(np.clip(sim_severity, 0.0, 1.0)),
        row_severity=float(np.clip(row_severity, 0.0, 1.0)),
    )

# =============================
# CLI
# =============================
def main():
    elapsed = time.time()
    parser = argparse.ArgumentParser(
        description="SanCheck — data sanity checker"
    )
    parser.add_argument("csv", 
                        help="Path to CSV file")
    
    parser.add_argument("target", 
                        help="Target column name (classification only)")
    
    parser.add_argument(
        "slice",
        type=Help.parse_slice_arg,
        help="Number of columns per chunk for plotting, or 'all'",
    )
    
    parser.add_argument(
        "--download-plot",
        action="store_true",
        help="Download plots as PNG files",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (overrides --download-plot)",
    )

    parser.add_argument(
        "--metrics-info",
        action=Help.InfoAction,
        help="Show detailed explanation of metrics",
        nargs=0
    )
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ Failed to load the CSV file: {e}")
        sys.exit(1)

    target = args.target
    if not isinstance(target, str):
        print(f"❌ Failed to process target. 'target' argument must be a string")
        sys.exit(1)

    elif target not in df.columns:
        print(f"❌ Failed to process target. {target} column is not exist in the DataFrame.")
        sys.exit(1)

    numeric_cols = Help.get_numeric_valid_columns(df, thresh=NUMERIC_VALID_RATIO)
    ignored_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        print("❌ No numeric columns found with sufficient valid numeric ratio.")
        print(f"Non-numeric columns / ignored: {', '.join(ignored_cols) if ignored_cols else '-'}")
        sys.exit(1)

    # Reports
    sim_pairs, sim_cols, sim_severity = Check.abnormal_similarity_report(df, numeric_cols, DEFAULT_SIM_THRESHOLD, EPS)
    nan_inf_df = Check.nan_inf_column_report(df, numeric_cols)
    type_df = Check.inconsistent_type_report(df, numeric_cols, thresh=0.05)
    normality = Check.compute_normality(df, numeric_cols)
    row_df, row_scores, row_severity = Check.problematic_row_report(df, numeric_cols, EPS)
    dist_df = Check.distribution_report(df, numeric_cols, EPS, ENTROPY_BINS)
    cleanliness = cleanliness_breakdown(df, nan_inf_df, type_df, sim_severity, row_severity)
    sparsity = Check.sparsity_ratio(df, numeric_cols)
    vif = Check.compute_vif(df, numeric_cols)
    override_rat = Check.class_override_ratio(df, numeric_cols, target)
    imbalance_rat = Check.class_imbalance_ratio(df, target)

    # plotting
    if not args.no_plot:
        PLT.plots(df, n_slice=args.slice, download_plot=args.download_plot)

    # normality
    shapiro = {c: Check.shapiro_per_feature(df[c]) for c in numeric_cols}
    ks = {c: Check.ks_per_feature(df[c], EPS) for c in numeric_cols}

    # summary
    top_entropy = dist_df.sort_values("entropy", ascending=False).head(5)
    top_spread = dist_df.sort_values("spread_score", ascending=False).head(5)
    top_rows = row_df.head(5)
    
    # Colorting
    print("\n[bold cyan]📊 Dataset Summary[/bold cyan]")
    print(f"- Valid numeric columns: {len(numeric_cols)}")
    print(f"- Ignored non-numeric columns: {len(ignored_cols)}")

    print("\n[cyan]📌 Column problems[/cyan]")
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

    print("\n[cyan]📌 Row problems[/cyan]")
    invalid_row_count = int(row_df["has_invalid_numeric"].sum()) if len(row_df) else 0
    print(f"- Problematic rows (NaN/Inf): {invalid_row_count}/{len(df)}")
    print(f"- Severity row: {row_severity:.3f}")
    for _, r in top_rows.iterrows():
        print(
            f"  - row {r['row_index']}: score={r['row_anomaly_score']:.3f}, "
            f"invalid={bool(r['has_invalid_numeric'])}"
        )

    print("\n[cyan]📌 Distribution and interpretation[/cyan]")
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
    
    print(f"\n[cyan]📌 Structure[/cyan]")
    print(f"- VIF mean (normalized): {vif['mean']:.3f}")
    print(f"- VIF per-feature")
    for c in numeric_cols:
        print(f"  - {c}: VIF={vif['per_feature'][c]:.3f}")
    print(f"- sparsity: {sparsity:.3f}")
    print(f"- class imbalance ratio: {imbalance_rat:.3f} | label={Help._label_from_score(imbalance_rat)}")
    print(f"- class override ratio: {override_rat:.3f} | label={Help._label_from_score(override_rat)}")

    print("\n[cyan]📌 Normality[/cyan]")
    print("- Shapiro-wilk and KS test score per-feature:")
    for c in numeric_cols:
        print(f"  - {c}: Shapiro={shapiro[c]:.3f} | KS={ks[c]:.3f}")
    print(f"- normality score (based on skewness and kurtosis): {normality:.3f} | label={Help._label_from_score(normality, toplow=True)}")

    print("\n[cyan]🧼 Cleanineess status[/cyan]")
    print(f"- cleanliness score: {cleanliness.overall:.3f} / 1.000")
    print(f"- cleanliness label: {cleanliness.label}")
    print(f"- missing severity: {cleanliness.missing_severity:.3f}")
    print(f"- type severity: {cleanliness.type_severity:.3f}")
    print(f"- similarity severity: {cleanliness.similarity_severity:.3f}")
    print(f"- row severity: {cleanliness.row_severity:.3f}")

    print("\n[cyan]📊 Dataset-level distribution summary[/cyan]")
    print(f"- avg entropy: {dist_df['entropy'].mean():.3f}")
    print(f"- avg spread score: {dist_df['spread_score'].mean():.3f}")
    
    print(f"\n⏱️ Elapsed time: {time.time() - elapsed:.2f} seconds (including plot visualization)")