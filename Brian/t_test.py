from session import Session
from orb import ORB

from scipy import stats
from statsmodels.stats.weightstats import ztest

def t_test(trades_df):
    pnl_points = trades_df["pnl_points"]
    t_stat, p_value = stats.ttest_1samp(pnl_points, popmean=0, alternative="greater")

    return t_stat, p_value


if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))
    _, trades_df = orb.get_all_trades()

    print(t_test(trades_df))
