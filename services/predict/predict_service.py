import torch


class PredictService:
    def __init__(self, model, X_predict):
        self.model = model
        self.X_predict = X_predict

    def execute(self):
        predicted_value = self.model(torch.tensor(self.X_predict, dtype=torch.float32))
        return predicted_value.detach().numpy().tolist()