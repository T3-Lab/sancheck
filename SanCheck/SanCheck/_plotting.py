import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from . import _check_func as Check
import pandas as pd

def plot_numeric_boxplot(df: pd.DataFrame, numeric_cols: list[str], title_suffix: str = ""):
    if not numeric_cols:
        print("⚠️ No valid numeric columns for boxplot.")
        return

    plot_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    spread_df = Check.distribution_report(df, numeric_cols, 1e-12, "fd").set_index("column")
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
