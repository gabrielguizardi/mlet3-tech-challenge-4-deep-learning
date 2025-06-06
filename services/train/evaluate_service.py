import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

class TrainEvaluateService:
    """
    A service class for evaluating trained models using various metrics.

    This service calculates multiple evaluation metrics for both training and testing datasets,
    including MAE, MAPE, RMSE, and R² scores.

    Attributes:
        model: The trained model to evaluate.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target values.
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize the TrainEvaluateService.

        Args:
            model: The trained model to evaluate.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing target values.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def execute(self):        
        train_metrics = self.__evaluate(self.X_train, self.y_train)
        test_metrics = self.__evaluate(self.X_test, self.y_test)
        
        return train_metrics, test_metrics

    def __evaluate(self, X: np.ndarray, y: np.ndarray):
        self.model.eval()
        
        with torch.no_grad():
            y_pred = self.model(torch.tensor(X, dtype=torch.float32))
        
        y_pred = y_pred.numpy()
        
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "r2": r2
        }