import pandas as pd
from session import Session

from typing import Optional
from dataclasses import dataclass


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # long or short
    stop_loss: float
    take_profit: float
    orb_high: float
    orb_low: float
    orb_range: float

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP, SL, EOD (end of day)
    pnl_points: Optional[float] = None  # blended per-unit PnL
    pnl: Optional[float] = None # real pnl based on the set number of micro contracts
    result: Optional[str] = None  # "WIN" / "LOSS" (selling for 0 profit (breakeven) is considered a loss)

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "direction": self.direction,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "orb_high": self.orb_high,
            "orb_low": self.orb_low,
            "orb_range": self.orb_range,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_points": self.pnl_points,
            "result": self.result}


class ORB:
    def __init__(self, session: Session, micro_contracts=10):
        self.session = session

        self.RR = 2.50
        self.SL_factor = 1.00  # SL will be in the 100% of the range

        # each micro contract (MNQ) leveraged by 2x, for a fill mini-contract or (NQ) it's 20x leverage
        self.micro_contracts = micro_contracts

    def enter_trade(self, ts, open, low, high, close, orb_low, orb_high, direction: str) -> Trade:
        orb_range = orb_high - orb_low

        sl_distance = orb_range * self.SL_factor  # = 1.0 * range
        tp_distance = self.RR * sl_distance  # = 2.0 * range
        stop_loss = close - sl_distance * (1 if direction == "long" else -1)
        take_profit = close + tp_distance * (1 if direction == "long" else -1)

        return Trade(
            entry_time=ts,
            entry_price=close,
            direction=direction,
            stop_loss=stop_loss,
            orb_low=orb_low,
            orb_high=orb_high,
            orb_range=orb_range,
            take_profit=take_profit)

    def exit_trade(self, trade: Trade, exit_ts, exit_price, exit_reason):
        trade.exit_time = exit_ts
        trade.exit_reason = exit_reason

        if trade.direction == "long":
            trade.pnl_points = exit_price - trade.entry_price
            trade.pnl = trade.pnl_points * self.micro_contracts * 2
        else:
            trade.pnl_points = (trade.entry_price - exit_price)
            trade.pnl = trade.pnl_points * self.micro_contracts * 2

        if trade.pnl_points <= 0:
            trade.result = "LOSS"
        else:
            trade.result = "WIN"
        return trade

    def get_trade(self, df):
        orb_bars = df.between_time("9:30", "9:44")

        orb_high = max(orb_bars["high"])
        orb_low = min(orb_bars["low"])
        orb_range = orb_high - orb_low

        if not (20 <= orb_range <= 150):  # filter out orb ranges that are too small or too big
            return None

        forward_bars = df.between_time("9:45", "16:00")

        trade = None
        for ts, (open, high, low, close) in forward_bars.iterrows():
            if trade is None:  # find entry
                if close < orb_low:  # short
                    trade = self.enter_trade(ts, open, low, high, close, orb_low, orb_high, "short")
                elif close > orb_high:
                    trade = self.enter_trade(ts, open, low, high, close, orb_low, orb_high, "long")
            else:  # EOD exit
                if ts > ts.replace(hour=15, minute=55):
                    trade = self.exit_trade(trade, ts, close, exit_reason="EOD")
                    break

                if trade.direction == "long":
                    if low <= trade.stop_loss:
                        self.exit_trade(trade, ts, trade.stop_loss, "SL")
                        break
                    elif high >= trade.take_profit:
                        self.exit_trade(trade, ts, trade.take_profit, "TP")
                        break
                elif trade.direction == "short":
                    if high >= trade.stop_loss:
                        self.exit_trade(trade, ts, trade.stop_loss, "SL")
                        break
                    elif low <= trade.take_profit:
                        self.exit_trade(trade, ts, trade.take_profit, "TP")
                        break

        if trade is not None and trade.exit_time is None:
            last_ts = forward_bars.index[-1]
            last_close = forward_bars['close'].iloc[-1]
            self.exit_trade(trade, last_ts, last_close, exit_reason="FORCE_CLOSE")
        return trade

    def get_all_trades(self):
        trades = []
        for idx in range(len(self.session)):
            df = self.session.get_session(idx)
            trade = self.get_trade(df)
            if trade is not None:
                trades.append(trade)
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
        return trades, trades_df


if __name__ == '__main__':
    session = Session()
    orb = ORB(session)
    trades, df = orb.get_all_trades()

    print(f"Total Trades:  {len(df)}")
    print(f"Wins:          {len(df[df['result'] == 'WIN'])}")
    print(f"Losses:        {len(df[df['result'] == 'LOSS'])}")
    print(f"Win Rate:      {len(df[df['result'] == 'WIN']) / len(df):.1%}")
    print(f"Total PnL:     {df['pnl_points'].sum():.2f} pts")
    print(f"Avg PnL/Trade: {df['pnl_points'].mean():.2f} pts")
    print(f"Best Trade:    {df['pnl_points'].max():.2f} pts")
    print(f"Worst Trade:   {df['pnl_points'].min():.2f} pts")

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(figsize=(10, 5))
    #
    # ax.hist(df["pnl_points"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    #
    # ax.axvline(df["pnl_points"].mean(), color="green", linestyle="--", label=f"Mean: {df['pnl_points'].mean():.2f}")
    # ax.axvline(df["pnl_points"].median(), color="red", linestyle="--", label=f"Median: {df['pnl_points'].median():.2f}")
    # ax.axvline(0, color="black", linestyle=":", label="Break-even: 0")
    #
    # ax.set_title("PnL Distribution — NQ 15min ORB")
    # ax.set_xlabel("PnL (Points)")
    # ax.set_ylabel("Count")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    from scipy import stats
    from statsmodels.stats.weightstats import ztest
    z_stat, z_p_value = ztest(df["pnl_points"], value=0.0, alternative='larger')
    print(z_stat, z_p_value)

    t_stat, p_value = stats.ttest_1samp(df['pnl_points'].dropna(), popmean=0, alternative="greater")


    # 2. Print the results
    print("\n--- Statistical Significance ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_value:.4f}")

    # 3. Interpret the result
    alpha = 0.05  # Standard significance level
    if p_value < alpha:
        print("Result: Statistically Significant (Reject H0)")
        print(f"The strategy likely has a real edge (p < {alpha}).")
    else:
        print("Result: Not Statistically Significant (Fail to reject H0)")
        print("The results could be due to random chance.")

    # equity curve
    import matplotlib.pyplot as plt

    # 1. Calculate Cumulative PnL
    # We add a 0 at the start so the graph begins at the origin
    cumulative_pnl = df['pnl_points'].cumsum()
    trade_numbers = range(1, len(cumulative_pnl) + 1)

    # 2. Create the Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the equity curve
    ax.plot(trade_numbers, cumulative_pnl, marker='o', linestyle='-', color='royalblue', markersize=4,
            label='Cumulative PnL')

    # Add a horizontal line at zero for reference
    ax.axhline(0, color='black', linewidth=1, linestyle='--')

    # Fill the area between the line and zero to highlight drawdowns vs profits
    ax.fill_between(trade_numbers, cumulative_pnl, 0, where=(cumulative_pnl >= 0), color='green', alpha=0.1)
    ax.fill_between(trade_numbers, cumulative_pnl, 0, where=(cumulative_pnl < 0), color='red', alpha=0.1)

    # Formatting
    ax.set_title("Equity Curve: Cumulative PnL over Number of Trades", fontsize=14, fontweight='bold')
    ax.set_xlabel("Trade Number", fontsize=12)
    ax.set_ylabel("Total PnL (Points)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()
