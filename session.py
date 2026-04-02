import pandas as pd
import datetime

class Session:
    def __init__(self, file_path="NAS100_1min_RTH.parquet"):
        if file_path.endswith(".csv"):
            self.rth_df = pd.read_csv(file_path, index_col=0, dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32'})
            self.rth_df.index = pd.to_datetime(self.rth_df.index, utc=True).tz_convert('America/New_York')
        elif file_path.endswith(".parquet"):
            self.rth_df = pd.read_parquet(file_path, engine="pyarrow")
        else:
            raise ValueError("File extension not supported!")
        self.sessions = [group for _, group in self.rth_df.groupby(self.rth_df.index.date)]
        self.dates = {index: date for index, (date, _) in enumerate(self.rth_df.groupby(self.rth_df.index.date))}

    def __len__(self):
        return len(self.sessions)

    def index_to_date(self, idx):
        date = self.dates.get(idx)
        if date is None:
            raise IndexError(f"Index not valid, might be under or over {len(self.sessions)}")
        return date

    def date_to_index(self, date: datetime.date):
        return list(self.dates.values()).index(date)  # O(N), not the time complexity I wanted but would have taken more mem complexity

    def get_dates(self):
        return list(self.dates.values())

    def get_session(self, idx):
        """
        :param idx:
        :return: A pandas dataframe of the open, low, high, close, volume from 9:30 to 16:00 EST
        """
        if idx >= len(self.sessions):
            raise IndexError(f"Index out of bounds, can only index up to less than {len(self.sessions)}")
        return self.sessions[idx]



if __name__ == "__main__":
    s = Session("NAS100_1min_RTH.csv")
    s = Session("NAS100_1min_RTH.parquet")

