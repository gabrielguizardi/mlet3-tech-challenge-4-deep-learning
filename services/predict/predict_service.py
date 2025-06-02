import torch


class PredictService:
    """
    Service class for making predictions using a trained model.

    This class handles the prediction process using a trained model on new input data.
    It converts input data to the appropriate format and returns model predictions.

    Attributes:
        model: The trained model used for making predictions.
        X_predict: Input data for prediction.
    """

    def __init__(self, model, X_predict):
        """
        Initialize the PredictService.

        Args:
            model: The trained model used for making predictions.
            X_predict: Input data for prediction.
        """
        self.model = model
        self.X_predict = X_predict

    def execute(self):
        predicted_value = self.model(torch.tensor(self.X_predict, dtype=torch.float32))
        return predicted_value.detach().numpy().tolist()