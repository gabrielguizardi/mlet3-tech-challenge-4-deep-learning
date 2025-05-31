from pydantic import BaseModel

class TrainModelRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    train_size: float
    sequence_length: int
    target_column: str
    epochs: int
