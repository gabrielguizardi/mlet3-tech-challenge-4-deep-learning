from services.preprocess_data_service import PreprocessDataService
from services.yfinance_service import YFinanceService

class PredictPrepareDataService:
    def __init__(self, metadata: dict, scaler: None):
        self.metadata = metadata
        self.scaler = scaler
        self.sequence_length = metadata['request']['sequence_length']
        self.ticker = metadata['request']['ticker']

    def execute(self):
        self.data = self.__get_yfinance_data()
        self.data = self.__preprocess_data()

        self.__validate_data()
    
        self.__sort_and_remove_date()
        self.data = self.__scale_data()

        return self.__create_sequences()

    def __get_yfinance_data(self):
       return YFinanceService(ticker=self.ticker, days=self.sequence_length).execute()
    
    def __preprocess_data(self):
        return PreprocessDataService(data=self.data).execute()

    def __validate_data(self):
        if len(self.data) < self.sequence_length:
            raise ValueError("DataFrame must have at least the same sequence length.")

    def __sort_and_remove_date(self):
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.data = self.data.drop('Date', axis=1)

    def __scale_data(self):
        if self.scaler is None:
            self.data

        return self.scaler.transform(self.data)
    
    def __create_sequences(self):
        sequences = []
        for i in range(len(self.data) - self.sequence_length + 1):
            sequence = self.data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
