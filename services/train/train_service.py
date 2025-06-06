import numpy as np
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping

from models.lightning_lstm_model import LightningLSTM

class TrainService:
    """
    A service class for training LSTM models using PyTorch Lightning.

    This service handles the training process of LSTM models, including data validation,
    dataloader creation, and model training with early stopping.

    Attributes:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target values.
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait before early stopping.
        features (int): Number of input features.

    Raises:
        ValueError: If training data is invalid or epochs parameter is incorrect.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 10, patience: int = 10):
        """
        Initialize the TrainService.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing target values.
            epochs (int): Number of training epochs (default: 10).
            patience (int): Number of epochs to wait before early stopping (default: 10).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.patience = patience
        self.features = X_train.shape[2] if len(X_train.shape) > 1 else 1

    def execute(self):
        self.__validate_data()
        self.__validate_epochs()

        train_loader, test_loader = self.__create_dataloaders()

        model = LightningLSTM(input_size=self.features, hidden_size=64, output_size=1)
        self.__train_model(model, train_loader, test_loader)

        return model

    def __validate_data(self):
        X_train_shape = self.X_train.shape
        y_train_shape = self.y_train.shape
        X_test_shape = self.X_test.shape
        y_test_shape = self.y_test.shape

        if X_train_shape[0] == 0 or y_train_shape[0] == 0 or X_test_shape[0] == 0 or y_test_shape[0] == 0:
            raise ValueError("Training and testing data must not be empty.")
    
        if X_train_shape[0] != y_train_shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.") 

        if X_test_shape[0] != y_test_shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples.")

        if X_train_shape[1] != X_test_shape[1]:
            raise ValueError("X_train and X_test must have the same number of features (columns).")

    def __validate_epochs(self):
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        
        if self.epochs > 1000:
            raise ValueError("Number of epochs must not exceed 1000.")
        
        if not isinstance(self.epochs, int):
            raise ValueError("Number of epochs must be an integer.")
        
    def __create_dataloaders(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader

    def __train_model(self, model, train_loader, test_loader):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=self.patience,
            verbose=True,
            mode='min'
        )
        trainer = L.Trainer(
            max_epochs=self.epochs,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            callbacks=[early_stop_callback]
        )
        trainer.fit(model, train_loader, test_loader)
