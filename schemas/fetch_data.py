from pydantic import BaseModel

class FetchDataRequest(BaseModel):
    ticker: str
    start_date: str = None
    end_date: str = None
    days: int = None
