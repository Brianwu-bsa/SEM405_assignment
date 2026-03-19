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
    result: Optional[str] = None  # "WIN" / "LOSS" (selling for 0 profit (breakeven) is considered a loss)

    @property
    def reward_points(self) -> float:
        return abs(self.take_profit - self.entry_price)

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
    def __init__(self, session: Session):
        self.session = session

        self.RR = 2.0
        self.SL_factor = 1.00  # SL will be in the 100% of the range

    def enter_trade(self, ts, open, low, high, close, orb_low, orb_high, direction: str) -> Trade:

        midpoint = (orb_low + orb_high) / 2
        risk = abs(close - midpoint)

        orb_range = orb_high - orb_low

        return Trade(
            entry_time=ts,
            entry_price=close,
            direction=direction,
            stop_loss=close + (orb_range * self.SL_factor) * (-1 if direction == "long" else 1),
            orb_low=orb_low,
            orb_high=orb_high,
            orb_range=orb_range,
            take_profit=close + self.RR * risk * (-1 if direction == "short" else 1))

    def exit_trade(self, trade: Trade, exit_ts, exit_price, exit_reason):
        trade.exit_time = exit_ts
        trade.exit_reason = exit_reason

        if trade.direction == "long":
            trade.pnl_points = exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - exit_price

        if trade.pnl_points <= 0:
            trade.result = "LOSS"
        else:
            trade.result = "WIN"
        return trade


    def get_trade(self, df):
        orb_bars = df.between_time("9:30", "9:44")
        orb_high = max(orb_bars["high"])
        orb_low = min(orb_bars["low"])

        forward_bars = df.between_time("9:45", "16:00")

        trade = None
        for ts, (open, high, low, close, volume) in forward_bars.iterrows():
            if trade is None:  # find entry
                if close < orb_low:  # short
                    trade = self.enter_trade(ts, open, low, high, close, orb_low, orb_high, "short")
                elif close > orb_high:
                    trade = self.enter_trade(ts, open, low, high, close, orb_low, orb_high, "long")
            else:

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


        if trade is not None:
            return trade

    def get_all_trades(self):
        trades = []
        for idx in range(len(self.session)):
            df = self.session.get_session(idx)
            trade = self.get_trade(df)
            if trade is not None:
                trades.append(trade)
        return trades


if __name__ == '__main__':
    session = Session()
    orb = ORB(session)
    trades = orb.get_all_trades()

    df = pd.DataFrame([t.to_dict() for t in trades])

    print(f"Total Trades:  {len(df)}")
    print(f"Wins:          {len(df[df['result'] == 'WIN'])}")
    print(f"Losses:        {len(df[df['result'] == 'LOSS'])}")
    print(f"Win Rate:      {len(df[df['result'] == 'WIN']) / len(df):.1%}")
    print(f"Total PnL:     {df['pnl_points'].sum():.2f} pts")
    print(f"Avg PnL/Trade: {df['pnl_points'].mean():.2f} pts")
    print(f"Best Trade:    {df['pnl_points'].max():.2f} pts")
    print(f"Worst Trade:   {df['pnl_points'].min():.2f} pts")
