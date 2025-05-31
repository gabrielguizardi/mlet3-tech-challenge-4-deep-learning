import pandas as pd
import numpy as np
import torch

pd.options.mode.copy_on_write = True

class TrainPrepareDataService:
    def __init__(self, data: pd.DataFrame, train_size: float = 0.8, sequence_length: int = 1, scaler=None, target_column: str = 'Close'):
        self.data = data
        self.train_size = train_size
        self.sequence_length = sequence_length
        self.scaler = scaler() if scaler is not None else None
        self.target_column = target_column

    def execute(self):
        self.__validate_data()
        self.__validate_train_size()
        self.__validate_sequence_length()

        self.__sort_and_remove_date()
        self.__create_target_column()

        train_data, test_data = self.__split_data()
        train_data, test_data = self.__scale_features(train_data, test_data)

        X_train, y_train = self.__create_sequences(train_data)
        X_test, y_test = self.__create_sequences(test_data)

        X_train_tensor, y_train_tensor = self.__prepare_tensors(X_train, y_train)
        X_test_tensor, y_test_tensor = self.__prepare_tensors(X_test, y_test)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, self.scaler

    def __validate_data(self):
        if self.data.empty:
            raise ValueError("DataFrame must not be empty.")

        if len(self.data) <= 200:
            raise ValueError("DataFrame must have more than 200 rows.")

    def __validate_train_size(self):
        if not (0 < self.train_size < 1):
            raise ValueError("Train size must be between 0 and 1.")
        
    def __validate_sequence_length(self):
        if self.sequence_length <= 0:
            raise ValueError("Sequence length must be greater than 0.")
        
        if self.sequence_length > len(self.data):
            raise ValueError("Sequence length cannot be greater than the number of rows in the DataFrame.")
    
    def __sort_and_remove_date(self):
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.data = self.data.drop('Date', axis=1)

    def __create_target_column(self):
        self.data['target'] = self.data[self.target_column].shift(-1)
        self.data = self.data.dropna(subset=['target'])
    
    def __split_data(self):
        train_size = int(len(self.data) * self.train_size)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]

        return train_data, test_data

    def __scale_features(self, train_data, test_data):
        if self.scaler:
            feature_cols = train_data.columns.drop('target')

            train_data[feature_cols] = self.scaler.fit_transform(train_data[feature_cols])
            test_data[feature_cols] = self.scaler.transform(test_data[feature_cols])

        return train_data, test_data

    def __create_sequences(self, data):
        X = []
        y = []

        for i in range(len(data) - self.sequence_length):
            X.append(data.iloc[i:i+self.sequence_length].drop('target', axis=1).values)
            y.append(data.iloc[i+self.sequence_length]['target'])

        return np.array(X), np.array(y)

    def __prepare_tensors(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor