import pandas as pd

df = pd.read_parquet("NQ_1min_RTH.parquet")
df.to_csv("NQ_1min_RTH.csv", index=False)