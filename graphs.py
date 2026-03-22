import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from session import Session
from orb import ORB

# Load data from existing ORB code
session = Session("NQ_1min_RTH.parquet")
orb = ORB(session)
trades = orb.get_all_trades()

# Convert trades to DataFrame
df = pd.DataFrame([t.to_dict() for t in trades])

# Main analysis column
pnl = df["pnl_points"].dropna()

print("\n===== MANUAL CHECK =====")

# Convert to list
pnl_list = pnl.tolist()

# Sort for median
sorted_pnl = sorted(pnl_list)

n = len(sorted_pnl)

# Manual median
if n % 2 == 0:
    manual_median = (sorted_pnl[n//2 - 1] + sorted_pnl[n//2]) / 2
else:
    manual_median = sorted_pnl[n//2]

# Manual mean
manual_mean = sum(pnl_list) / n

print("Manual Mean:", round(manual_mean, 2))
print("Pandas Mean:", round(pnl.mean(), 2))

print("Manual Median:", round(manual_median, 2))
print("Pandas Median:", round(pnl.median(), 2))

# ---------------- FULL DATA ----------------
mean_pnl = pnl.mean()
median_pnl = pnl.median()

print("Number of trades used for analysis:", len(pnl))
print("Full Data Mean:", round(mean_pnl, 2))
print("Full Data Median:", round(median_pnl, 2))

# Full data histogram
plt.figure(figsize=(12, 7))
sns.histplot(pnl, bins=30, kde=True, color="skyblue", edgecolor="white", alpha=0.85)
plt.axvline(mean_pnl, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_pnl:.2f}")
plt.axvline(median_pnl, color="red", linestyle="--", linewidth=2, label=f"Median: {median_pnl:.2f}")
plt.axvline(0, color="black", linestyle=":", linewidth=2, label="Break-even: 0")
plt.title("PnL Distribution — Full Data")
plt.xlabel("PnL (Points)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("histogram_full.png", dpi=300)
plt.show()

# Full data box plot
plt.figure(figsize=(8, 8))
plt.boxplot(
    pnl,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="black"),
    medianprops=dict(color="orange", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8),
)
plt.axhline(mean_pnl, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_pnl:.2f}")
plt.axhline(median_pnl, color="red", linestyle="--", linewidth=2, label=f"Median: {median_pnl:.2f}")
plt.axhline(0, color="black", linestyle=":", linewidth=2, label="Break-even: 0")
plt.title("Box Plot of Trade PnL — Full Data")
plt.ylabel("PnL (Points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("boxplot_full.png", dpi=300)
plt.show()

# ---------------- 10% SAMPLE ----------------
sample = pnl.sample(frac=0.1, random_state=42)
sample_mean = sample.mean()
sample_median = sample.median()

print("\n10% Sample size:", len(sample))
print("Sample Mean:", round(sample_mean, 2))
print("Sample Median:", round(sample_median, 2))

# Sample histogram
plt.figure(figsize=(12, 7))
sns.histplot(sample, bins=20, kde=True, color="lightgreen", edgecolor="white", alpha=0.85)
plt.axvline(sample_mean, color="green", linestyle="--", linewidth=2, label=f"Mean: {sample_mean:.2f}")
plt.axvline(sample_median, color="red", linestyle="--", linewidth=2, label=f"Median: {sample_median:.2f}")
plt.axvline(0, color="black", linestyle=":", linewidth=2, label="Break-even: 0")
plt.title("PnL Distribution — 10% Sample")
plt.xlabel("PnL (Points)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("histogram_sample.png", dpi=300)
plt.show()

# Sample box plot
plt.figure(figsize=(8, 8))
plt.boxplot(
    sample,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen", color="black"),
    medianprops=dict(color="orange", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8),
)
plt.axhline(sample_mean, color="green", linestyle="--", linewidth=2, label=f"Mean: {sample_mean:.2f}")
plt.axhline(sample_median, color="red", linestyle="--", linewidth=2, label=f"Median: {sample_median:.2f}")
plt.axhline(0, color="black", linestyle=":", linewidth=2, label="Break-even: 0")
plt.title("Box Plot of Trade PnL — 10% Sample")
plt.ylabel("PnL (Points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("boxplot_sample.png", dpi=300)
plt.show()