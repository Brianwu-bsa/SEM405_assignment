from session import Session
from orb import ORB
from scipy import stats


def compute_confidence_interval(trades_df, CI=0.95):
    pnl = trades_df["pnl_points"]
    mean = pnl.mean()
    SE = stats.sem(pnl)  # standard error
    return stats.t.interval(CI, df=len(pnl) - 1, loc=mean, scale=SE)  # df here is degrees of freedom


if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))
    _, trades_df = orb.get_all_trades()
    trades_df.to_csv("trade_df.csv")

    for CI in [0.75, 0.85, 0.95, 0.99]:
        CIL, CIU = compute_confidence_interval(trades_df, CI)
        print(CIL, CIU)
