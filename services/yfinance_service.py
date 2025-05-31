"""
Serviço para obter e processar dados históricos de ações usando a biblioteca yfinance.

Classes:
    YFinanceService: Classe responsável por buscar e processar dados de ações de um ticker específico em um intervalo de datas.

Exemplo de uso:
    service = YFinanceService('AAPL', '2022-01-01', '2022-12-31')
    resultado = service.execute()
"""

import yfinance as yf
import pandas as pd

class YFinanceService:
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None, days: int = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.days = days

    def execute(self):
        self.__validate_dates()

        stock_data = self.__get_stock_data()
        return self.__process_stock_data(stock_data)

    def __validate_dates(self):
        if (not self.start_date or not self.end_date) and not self.days:
            raise ValueError("Start date and end date must be provided.")

        if self.days:
            if self.days <= 0:
                raise ValueError("Days must be a positive integer.")
        else:
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)

            if start_dt >= end_dt:
                raise ValueError("Start date must be earlier than end date.")
    
    def __get_stock_data(self):
        yf_ticker = yf.Ticker(self.ticker)

        if self.days:
            period = f"{self.days}d"
            return yf_ticker.history(period=period)

        return yf_ticker.history(start=self.start_date, end=self.end_date)

    def __process_stock_data(self, dataframe: pd.DataFrame):
        dataframe.reset_index(inplace=True)
        return dataframe