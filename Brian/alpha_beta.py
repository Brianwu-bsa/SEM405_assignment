from scipy import stats
import matplotlib.pyplot as plt

from session import Session
from orb import ORB


def compute_alpha_beta(trades_df):
    market_return = trades_df["market_return"]  # X
    strategy_return = trades_df["pnl_points"] / trades_df["entry_price"]  # y
    result = stats.linregress(market_return, strategy_return)
    result = {"slope": result.slope,
              "intercept": result.intercept,
              "r_value": result.rvalue,
              "r_squared": result.rvalue ** 2,
              "p_value": result.pvalue,
              "annualized_alpha": ((1 + result.intercept) ** 252 - 1)}
    return result

def compute_levered_alpha_beta(trades_df, account_value=50000):
    market_return = trades_df["market_return"]  # X
    strategy_return = trades_df["pnl"] / account_value  # y
    result = stats.linregress(market_return, strategy_return)
    result = {"slope": result.slope,
              "intercept": result.intercept,
              "r_value": result.rvalue,
              "r_squared": result.rvalue ** 2,
              "p_value": result.pvalue,
              "annualized_alpha": ((1 + result.intercept) ** 252 - 1)}
    return result

def print_result(result):
    beta = result["slope"]
    alpha = result["intercept"]
    r_value = result["r_value"]  # rvalue is the correlation coefficient (R)
    r_squared = result["r_squared"]
    p_value = result["p_value"]  # p-value for the hypothesis test on Beta
    annualized_alpha = result["annualized_alpha"]

    print(f"Beta (Market Exposure): {round(beta, 4)}")
    print(f"Alpha (Excess Return): {round(alpha, 6)} (Per Trade/Day)")
    print(f"Alpha (Excess Return): {round(annualized_alpha, 6)} (Annualized)")
    print(f"r-value: {round(r_value, 4)}")
    print(f"R-squared (Determination): {round(r_squared, 4)}")
    print(f"P-value (Beta significance): {round(p_value, 4)}")

def plot_returns(trades_df, result):
    market_return = trades_df["market_return"]
    strategy_return = trades_df["pnl_points"] / trades_df["entry_price"]

    regression_line = result["intercept"] + result["slope"] * market_return

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scatter points of the actual daily returns
    ax.scatter(market_return, strategy_return, alpha=0.5, color='cyan', edgecolor='black', label='Daily Returns')

    # Plot the linear regression line
    ax.plot(market_return, regression_line, color='red', linewidth=2,
            label=f'Regression Line ($y = {round(result["slope"], 4)}x + {round(result["intercept"], 6)}$)')

    # Formatting and Labels
    ax.set_title("15m ORB Strategy Return vs. NAS100 Market Return", fontsize=14, fontweight='bold')
    ax.set_xlabel("Market Return (RTH Close - Open) / Open", fontsize=12)
    ax.set_ylabel("Strategy Return (Unlevered)", fontsize=12)

    # Add horizontal and vertical lines at 0 to clearly divide the 4 performance quadrants
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')

    # Create a text box to display the statistical results directly on the chart
    stats_text = (
        f"$\\beta$ (Beta): {round(result["slope"], 4)}\n"
        f"$\\alpha$ (Alpha): {round(result["intercept"], 6)}\n"
        f"$\\alpha$ (Annualized Alpha): {round(result["annualized_alpha"] * 100, 2)}%\n"
        f"$r$: {round(result["r_value"], 4)}\n"
        f"$R^2$: {round(result["r_squared"], 4)}\n"
        f"p-value: {round(result["p_value"], 4)}"
    )

    # Place the text box in the upper left corner
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Add grid and legend
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

    plt.savefig("returns.png")

    plt.tight_layout()
    plt.show()


def plot_levered_returns(trades_df, result, account_value=50000):
    market_return = trades_df["market_return"]
    strategy_return = trades_df["pnl"] / account_value

    regression_line = result["intercept"] + result["slope"] * market_return

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scatter points of the actual daily returns
    ax.scatter(market_return, strategy_return, alpha=0.75, color='deepskyblue', edgecolor='black', label='Daily Returns')

    # Plot the linear regression line
    ax.plot(market_return, regression_line, color='red', linewidth=2,
            label=f'Regression Line ($y = {round(result["slope"], 4)}x + {round(result["intercept"], 6)}$)')

    # Formatting and Labels
    ax.set_title("15m ORB Strategy Return (Levered ~8.5x) vs. NAS100 Market Return", fontsize=14, fontweight='bold')
    ax.set_xlabel("Market Return (RTH Close - Open) / Open", fontsize=12)
    ax.set_ylabel("Strategy Return (Levered)", fontsize=12)

    # Add horizontal and vertical lines at 0 to clearly divide the 4 performance quadrants
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')

    # Create a text box to display the statistical results directly on the chart
    stats_text = (
        f"$\\beta$ (Beta): {round(result["slope"], 4)}\n"
        f"$\\alpha$ (Alpha): {round(result["intercept"], 6)}\n"
        f"$\\alpha$ (Annualized Alpha): {round(result["annualized_alpha"] * 100, 2)}%\n"
        f"$r$: {round(result["r_value"], 4)}\n"
        f"$R^2$: {round(result["r_squared"], 4)}\n"
        f"p-value: {round(result["p_value"], 4)}"
    )

    # Place the text box in the upper left corner
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Add grid and legend
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

    plt.savefig("levered_returns.png")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))
    trades, trades_df = orb.get_all_trades()

    print("--- Linear Regression Results (Strategy vs Market) Unlevered ---")
    result = compute_alpha_beta(trades_df)
    print_result(result)
    plot_returns(trades_df, result)

    print("--- Linear Regression Results (Strategy vs Market) Levered ---")
    result = compute_levered_alpha_beta(trades_df)
    print_result(result)
    plot_levered_returns(trades_df, result)