"""
Note the time zone for the dataset which is US Eastern Time (EST
Regular Trading Hours are 9:30 to 16:00
Premarket (PM): 18:00 to 9:30
Aftermarket (AM): 16:00 t0 17:00

From 17:00 to 18:00 the stock exchange closes down temporarily for maintenance

"""

import pandas as pd
import time

"""
Author: Brian 
Written: 2026-03-18
"""


def preprocess_data(input_filepath, start_date):
    df = pd.read_csv(input_filepath, sep="\t")
    df.columns = df.columns.str.lower()

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M:%S')
    df['datetime'] = df['datetime'].dt.tz_localize('+03:00').dt.tz_convert('America/New_York')
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    df = df.loc[start_date:]
    df_rth = df.between_time("09:30", "16:00").copy()

    ohlc_cols = ['open', 'high', 'low', 'close']
    df_rth = df_rth[ohlc_cols].astype('float32')

    df_rth.to_csv('NAS100_1min_RTH.csv')
    df_rth.to_parquet('NAS100_1min_RTH.parquet')

    print(f"Processing complete!")
    print(f"Total RTH bars saved: {len(df_rth)}")

    return df_rth


if __name__ == "__main__":
    processed_df = preprocess_data('nas100_1min_ETH.csv', "2022-01-01")

    print(processed_df.head())