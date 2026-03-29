import pandas as pd
import matplotlib.pyplot as plt

from orb import ORB
from session import Session

def plot_equity_curve(trades_df, mode="pnl_points"):
    cumulative_pnl = trades_df[mode].cumsum() # calc the cumulative PNL
    trade_numbers = range(1, len(cumulative_pnl) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the equity curve
    ax.plot(trade_numbers, cumulative_pnl, marker='o', linestyle='-', color='royalblue', markersize=4,
            label='Cumulative PnL')

    # add horizontal line at zero for reference
    ax.axhline(0, color='black', linewidth=1, linestyle='--')

    # Add the area between the line and zero to highlight drawdowns vs profits
    ax.fill_between(trade_numbers, cumulative_pnl, 0, where=(cumulative_pnl >= 0), color='green', alpha=0.1)
    ax.fill_between(trade_numbers, cumulative_pnl, 0, where=(cumulative_pnl < 0), color='red', alpha=0.1)

    title = f"Equity Curve: Cumulative PnL {'points' if mode == 'pnl_points' else ''} over Number of Trades"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Trade Number", fontsize=12)
    ax.set_ylabel(f"Total PnL {'(Points)' if mode == 'pnl_points' else ''}", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"Equity_curve_pnl{'_points' if mode == 'pnl_points' else ''}.png")

if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))

    trades, df = orb.get_all_trades()

    plot_equity_curve(df, mode="pnl_points")
    plot_equity_curve(df, mode="pnl")