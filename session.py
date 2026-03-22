import pandas as pd
import datetime

class Session:
    def __init__(self, file_path="NQ_1min_RTH.parquet"):
        self.rth_df = pd.read_parquet(file_path, engine="pyarrow")
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
    s = Session()
    # print(len(s))
    # print(s.get_session(765))
    date = datetime.date(2025, 12, 11)
    print(s.get_dates())
    # print(s.date_to_index(date))
