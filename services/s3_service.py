import uuid

class S3Service:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def execute(self):
        train_path = self.__generate_train_path()
        
        print(f"Saving model to {train_path}")

    def __generate_train_path(self):
        return f"models/{uuid.uuid4()}.pt"