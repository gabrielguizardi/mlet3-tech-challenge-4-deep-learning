import pandas as pd
 
class PreprocessDataService:
    """
    A service class for preprocessing stock market data.

    This class handles basic data preprocessing tasks such as removing null values
    and selecting relevant columns from the input DataFrame.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing stock market data.

    Raises:
        ValueError: If the input DataFrame is empty.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the PreprocessDataService.

        Args:
            data (pd.DataFrame): The input DataFrame to preprocess.
        """
        self.data = data

    def execute(self):
        self.__validate_data()
        self.__remove_nulls()
        self.__select_columns()

        return self.data

    def __validate_data(self):
        if self.data.empty:
            raise ValueError("DataFrame must not be empty.")

    def __remove_nulls(self):
        self.data.dropna(inplace=True)

        self.__validate_data()

    def __select_columns(self):
        columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        self.data = self.data[columns]
