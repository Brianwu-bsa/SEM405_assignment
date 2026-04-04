from session import Session
from orb import ORB

from scipy import stats
from statsmodels.stats.weightstats import ztest
def t_test(trades_df):
    pnl_points = trades_df["pnl_points"]
    t_stat, p_value = stats.ttest_1samp(pnl_points, popmean=0, alternative="greater")

    return t_stat, p_value

def z_test(trades_df):
    pnl_points = trades_df["pnl_points"]
    z_stat, p_value = ztest(pnl_points, value=0, alternative="larger")
    return z_stat, p_value

if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))
    _, trades_df = orb.get_all_trades()

    print(f"t-test, (t-stat, p-value):", t_test(trades_df))
    print(f"z-test, (z-stat, p-value):", z_test(trades_df))
