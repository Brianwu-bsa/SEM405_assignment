"""
descriptive_stats.py — Descriptive Statistics for ORB Trade PnL
Author: Mirac

This script computes central tendency, dispersion, and position measures
for the full set of Opening Range Breakout trades, then repeats the
analysis on a random 10 % subsample to verify that the trading edge
is not an artefact of the full dataset.

Metrics computed
────────────────
  Central tendency : mean, median, mode, trimmed mean (1 % and 2.5 %)
  Dispersion       : variance, standard deviation
  Position         : percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99),
                     quartiles (Q1, Q2, Q3), IQR
"""

import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from collections import Counter

from session import Session
from orb import ORB


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_trimmed_mean(data: np.ndarray, trim_pct: float) -> float:
    """
    Computes a trimmed mean by removing the lowest and highest
    *trim_pct* fraction of values before averaging.
    For example trim_pct = 0.01 removes the bottom 1 % and the top 1 %.
    """
    return float(sp_stats.trim_mean(data, trim_pct))


def compute_mode(data: np.ndarray) -> float:
    """
    Finds the most frequently occurring value.
    Because PnL values are continuous floats, we round to 2 decimal
    places first so that nearly identical values are grouped together.
    If there are multiple modes, we return the smallest one.
    """
    rounded = np.round(data, 2)
    counts = Counter(rounded)
    max_count = max(counts.values())
    modes = sorted(val for val, cnt in counts.items() if cnt == max_count)
    return float(modes[0])


def compute_percentiles(data: np.ndarray) -> dict:
    """
    Computes a set of standard percentiles and the inter-quartile range.
    Percentiles are calculated using linear interpolation (NumPy default).
    """
    pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {f"P{p}": float(np.percentile(data, p)) for p in pct_keys}

    # Quartiles are just specific percentiles; we label them explicitly
    percentiles["Q1"] = percentiles["P25"]
    percentiles["Q2"] = percentiles["P50"]
    percentiles["Q3"] = percentiles["P75"]
    percentiles["IQR"] = percentiles["Q3"] - percentiles["Q1"]
    return percentiles


def compute_all_stats(data: np.ndarray, label: str = "Full Data") -> dict:
    """
    Runs all descriptive statistics on a given array of PnL values
    and returns them as a tidy dictionary.
    """
    results = {
        "label": label,
        "n": len(data),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "mode": compute_mode(data),
        "trimmed_mean_1pct": compute_trimmed_mean(data, 0.01),
        "trimmed_mean_2_5pct": compute_trimmed_mean(data, 0.025),
        "variance": float(np.var(data, ddof=1)),       # sample variance (N-1)
        "std_dev": float(np.std(data, ddof=1)),         # sample std dev  (N-1)
    }
    results.update(compute_percentiles(data))
    return results


# ── Pretty-printer ───────────────────────────────────────────────────────────

SEPARATOR = "─" * 60

def print_section(title: str):
    """Prints a clearly visible section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_stats(s: dict):
    """Prints the statistics dictionary in a readable table format."""
    print_section(s["label"])
    print(f"  Number of trades        : {s['n']}")
    print()
    print(f"  Mean                    : {s['mean']:>10.2f} pts")
    print(f"  Median                  : {s['median']:>10.2f} pts")
    print(f"  Mode                    : {s['mode']:>10.2f} pts")
    print(f"  Trimmed Mean (1 %)      : {s['trimmed_mean_1pct']:>10.2f} pts")
    print(f"  Trimmed Mean (2.5 %)    : {s['trimmed_mean_2_5pct']:>10.2f} pts")
    print()
    print(f"  Variance                : {s['variance']:>10.2f}")
    print(f"  Standard Deviation      : {s['std_dev']:>10.2f} pts")
    print()
    print(f"  P1                      : {s['P1']:>10.2f} pts")
    print(f"  P5                      : {s['P5']:>10.2f} pts")
    print(f"  P10                     : {s['P10']:>10.2f} pts")
    print(f"  Q1  (25th percentile)   : {s['Q1']:>10.2f} pts")
    print(f"  Q2  (50th / median)     : {s['Q2']:>10.2f} pts")
    print(f"  Q3  (75th percentile)   : {s['Q3']:>10.2f} pts")
    print(f"  P90                     : {s['P90']:>10.2f} pts")
    print(f"  P95                     : {s['P95']:>10.2f} pts")
    print(f"  P99                     : {s['P99']:>10.2f} pts")
    print(f"  IQR (Q3 − Q1)          : {s['IQR']:>10.2f} pts")


def print_comparison(full: dict, sample: dict):
    """
    Prints a side-by-side comparison between the full dataset and the
    10 % subsample so we can visually evaluate whether the edge persists.
    """
    print_section("Comparison — Full Data vs. 10 % Subsample")

    header = f"  {'Metric':<26} {'Full':>10}   {'Sample':>10}   {'Delta':>10}"
    print(header)
    print("  " + "─" * 56)

    keys = [
        ("Mean",                "mean"),
        ("Median",              "median"),
        ("Mode",                "mode"),
        ("Trimmed Mean 1 %",    "trimmed_mean_1pct"),
        ("Trimmed Mean 2.5 %",  "trimmed_mean_2_5pct"),
        ("Std Dev",             "std_dev"),
        ("Q1",                  "Q1"),
        ("Q3",                  "Q3"),
        ("IQR",                 "IQR"),
    ]
    for label, key in keys:
        f_val = full[key]
        s_val = sample[key]
        delta = s_val - f_val
        print(f"  {label:<26} {f_val:>10.2f}   {s_val:>10.2f}   {delta:>+10.2f}")

    # Check whether the subsample still has a positive mean PnL (i.e. edge holds)
    print()
    if sample["mean"] > 0:
        print("  ✓  Subsample mean is positive — the edge holds in the 10 % draw.")
    else:
        print("  ✗  Subsample mean is non-positive — the edge may not be robust.")

    # Show that Q1 is smaller than Q3 demonstrating asymmetric risk/reward
    if full["Q1"] < full["Q3"] and abs(full["Q3"]) > abs(full["Q1"]):
        print("  ✓  |Q3| > |Q1| — asymmetric risk-to-reward confirmed (more upside).")
    else:
        print("  ⚠  Q1/Q3 relationship does not show clear asymmetric reward.")


# ── Charts ───────────────────────────────────────────────────────────────────

def plot_percentile_bar(stats: dict, filename: str):
    """
    Horizontal bar chart of all nine percentiles.
    This gives a quick visual overview of the PnL distribution's shape —
    if the right side stretches further than the left, the reward
    is asymmetric relative to the risk.
    """
    labels = ["P1", "P5", "P10", "Q1", "Q2 (Median)", "Q3", "P90", "P95", "P99"]
    keys   = ["P1", "P5", "P10", "Q1", "Q2",           "Q3", "P90", "P95", "P99"]
    values = [stats[k] for k in keys]

    colours = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colours, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("PnL (Points)")
    ax.set_title(f"Percentile Breakdown — {stats['label']}")
    ax.grid(axis="x", alpha=0.3)

    # Put value labels on each bar
    for bar, val in zip(bars, values):
        x_pos = val + (3 if val >= 0 else -3)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"  → Saved: {filename}")


def plot_comparison_grouped_bar(full: dict, sample: dict, filename: str):
    """
    Grouped bar chart that puts Full vs. Sample side by side for the
    key summary statistics so the viewer can compare at a glance.
    """
    labels = ["Mean", "Median", "Mode", "Trim 1%", "Trim 2.5%",
              "Std Dev", "Q1", "Q3", "IQR"]
    keys   = ["mean", "median", "mode", "trimmed_mean_1pct",
              "trimmed_mean_2_5pct", "std_dev", "Q1", "Q3", "IQR"]

    full_vals   = [full[k] for k in keys]
    sample_vals = [sample[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, full_vals,   width, label="Full Data",
           color="#3498db", edgecolor="white")
    ax.bar(x + width / 2, sample_vals, width, label="10 % Sample",
           color="#e67e22", edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("PnL (Points)")
    ax.set_title("Full Data vs. 10 % Subsample — Key Statistics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"  → Saved: {filename}")


# ── Main execution ───────────────────────────────────────────────────────────

def main():
    # Load the 1-minute NQ data and generate all ORB trades
    session = Session("../NAS100_1min_RTH.parquet")
    orb = ORB(session)
    trades, df = orb.get_all_trades()

    # Build a DataFrame and extract the PnL column, dropping any None values
    pnl = df["pnl_points"].dropna()

    # ── Full dataset analysis ────────────────────────────────────────────
    full_stats = compute_all_stats(pnl.values, label="Full Data — All Trades")
    print_stats(full_stats)

    # ── 10 % random subsample ───────────────────────────────────────────
    pnl_sample = pnl.sample(frac=0.1, random_state=42)
    sample_stats = compute_all_stats(pnl_sample, label="10 % Subsample")
    print_stats(sample_stats)

    # ── Side-by-side comparison ─────────────────────────────────────────
    print_comparison(full_stats, sample_stats)

    # ── Generate charts ─────────────────────────────────────────────────
    print_section("Charts")
    plot_percentile_bar(full_stats, "percentiles_full.png")
    plot_percentile_bar(sample_stats, "percentiles_sample.png")
    plot_comparison_grouped_bar(full_stats, sample_stats, "comparison_full_vs_sample.png")

    print(f"\n{SEPARATOR}")
    print("  Analysis complete.")
    print(SEPARATOR)

    # ── Save all console output to a text file ──────────────────────────
    # We replay the prints into a StringIO buffer and dump to file
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = buf

    print_stats(full_stats)
    print_stats(sample_stats)
    print_comparison(full_stats, sample_stats)

    sys.stdout = _orig_stdout
    with open("descriptive_stats_output.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  → Saved: descriptive_stats_output.txt")


if __name__ == "__main__":
    main()
