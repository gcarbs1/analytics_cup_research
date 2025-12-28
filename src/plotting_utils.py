"""
Visualization utilities for PADI analysis.

This module contains plotting functions for creating publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import pearsonr
from typing import Dict


def lighten(hex_color: str, amount: float = 0.55) -> str:
    """
    Lighten a hex color by blending with white.

    Args:
        hex_color: Hex color code (e.g., '#068f41')
        amount: Blending amount (0=original, 1=white)

    Returns:
        Lightened hex color code
    """
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    out = rgb * (1 - amount) + white * amount
    return mcolors.to_hex(out)


def plot_padi_correlation(
    df_analysis: pd.DataFrame,
    output_path: str = "viz/padi_correlation_plot.png",
    color_map: Dict[str, str] = None,
    figsize: tuple = (12.8, 6.2),
    dpi: int = 170
) -> None:
    """
    Create a position-specific correlation plot for PADI vs SofaScore ratings.

    Displays distribution of PADI values by position with scatter points, percentile
    ranges (5th-95th), mean markers, and correlation coefficients.

    Args:
        df_analysis: DataFrame with 'PADI_value_median', 'sofascore_rating',
                     and 'position_group' columns
        output_path: Path to save the figure
        color_map: Optional dictionary mapping position groups to colors
        figsize: Figure size in inches
        dpi: Resolution for saved figure
    """
    if color_map is None:
        color_map = {
            "DEF": "#068f41",
            "MF": "#06478f",
            "FW": "#b01f16",
        }
    default_color = "#06478f"

    def cpos(p):
        return color_map.get(p, default_color)

    # Filter valid data
    df_analysis = df_analysis.loc[
        df_analysis["sofascore_rating"].notna() & df_analysis["PADI_value_median"].notna()
    ].copy()

    # Compute overall correlation
    r_all, p_all = pearsonr(df_analysis["PADI_value_median"], df_analysis["sofascore_rating"])

    # Compute position-specific correlations
    rows = []
    for pos in sorted(df_analysis["position_group"].dropna().unique()):
        dfp = df_analysis[df_analysis["position_group"] == pos]
        if len(dfp) < 3:
            continue
        r, p = pearsonr(dfp["PADI_value_median"], dfp["sofascore_rating"])
        rows.append({"position": pos, "r": r})

    df_corr = pd.DataFrame(rows)

    # Order positions
    preferred_order = ["DEF", "MF", "FW", "ATA", "MEI"]
    pos_list = [p for p in preferred_order if p in set(df_corr["position"])]
    pos_list += [p for p in df_corr["position"].tolist() if p not in pos_list]
    df_corr = df_corr.set_index("position").loc[pos_list].reset_index()

    # Configure matplotlib style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Open Sans", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.titleweight"] = "bold"

    rng = np.random.default_rng(42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = np.arange(len(pos_list))

    all_vals = df_analysis["PADI_value_median"].astype(float).values
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    xrange = max(xmax - xmin, 1e-9)
    dx = 0.03 * xrange

    # Plot each position
    for i, pos in enumerate(pos_list):
        vals = df_analysis.loc[df_analysis["position_group"] == pos, "PADI_value_median"].astype(float).values
        if len(vals) == 0:
            continue

        base = cpos(pos)
        dots = lighten(base, 0.55)

        # Jittered scatter points
        yj = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(
            vals, yj,
            s=18,
            facecolors=dots,
            edgecolors=base,
            linewidths=0.6,
            zorder=2
        )

        # Percentile range line
        p5, p95 = np.percentile(vals, [5, 95])
        m = float(np.mean(vals))
        ax.hlines(i, p5, p95, color=base, linewidth=1.3, zorder=3)

        # Mean marker
        ax.scatter([m], [i], s=130, facecolors=base, edgecolors="#111111", linewidths=0.9, zorder=4)

        # Correlation coefficient annotation
        rho = float(df_corr.loc[df_corr["position"] == pos, "r"].iloc[0])
        x_last = float(np.max(vals))
        ax.text(
            x_last + dx, i,
            f"ρ = {rho:.2f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#111111"
        )

    # Configure axes
    ax.set_yticks(y)
    ax.set_yticklabels(pos_list, fontsize=12)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_xlabel("PADI VALUE MEDIAN", fontsize=12, fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.tick_params(axis="both", colors="#111111")

    # Adjust x-axis to accommodate correlation labels
    xmax_with_text = max(xmax, max(
        float(np.max(df_analysis.loc[df_analysis["position_group"] == pos, "PADI_value_median"].astype(float).values)) + 6*dx
        for pos in pos_list
    ))
    ax.set_xlim(xmin - 0.02*xrange, xmax_with_text)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {output_path}")
