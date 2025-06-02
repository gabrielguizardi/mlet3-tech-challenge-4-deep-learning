import torch.nn as nn
import pytorch_lightning as L
from torch.optim import Adam
from torch import Tensor
from typing import Optional

class LightningLSTM(L.LightningModule):
    """
    A PyTorch Lightning implementation of an LSTM model for time series prediction.

    This model uses a stacked LSTM architecture followed by a fully connected layer
    for time series forecasting tasks.

    Attributes:
        lstm (nn.LSTM): The LSTM layers for sequence processing.
        fc (nn.Linear): The fully connected output layer.
        criterion (nn.MSELoss): Mean squared error loss function.

    Args:
        input_size (int): The number of input features (default: 1).
        hidden_size (int): The number of features in the hidden state (default: 64).
        output_size (int): The size of the output (default: 1).
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 1):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Size of the output.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        # x: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.fc(hidden[-1])  # hidden[-1]: (batch, hidden_size)
        return out.squeeze(-1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
