class DataProcessing:
    def __init__(self, df):
        self.df = df

    def data_cleaning(self):
        print("Data Cleaning Starting...")
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise Exception("DataFrame is empty.")
        print("Data Cleaning Complete")
        return self.df

