import pandas as pd
 
class PreprocessDataService:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def execute(self):
        self.__validate_data()
        self.__remove_nulls()
        self.__select_columns()

        return self.data

    def __validate_data(self):
        if self.data.empty:
            raise ValueError("DataFrame must not be empty.")

        if len(self.data) <= 200:
            raise ValueError("DataFrame must have more than 200 rows.")

    def __remove_nulls(self):
        self.data.dropna(inplace=True)

        self.__validate_data()

    def __select_columns(self):
        columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        self.data = self.data[columns]   
        