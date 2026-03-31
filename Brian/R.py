from session import Session
from orb import ORB
from scipy import stats


def compute_r(trades_df):
    r, p_value = stats.pearsonr(trades_df["orb_range"], trades_df["pnl_points"])
    r_squared = r ** 2
    return r, r_squared, p_value

if __name__ == "__main__":
    orb = ORB(Session("../NAS100_1min_RTH.parquet"))
    _, trades_df = orb.get_all_trades()
    r, r_squared, p_value = compute_r(trades_df)
    print(f"Correlation (r):          {r:.4f}")
    print(f"Coefficient of Det (R²):  {r_squared:.4f}")
    print(f"P-value:                  {p_value:.4f}")