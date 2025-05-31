from pydantic import BaseModel

class PredictRequest(BaseModel):
    days: int
