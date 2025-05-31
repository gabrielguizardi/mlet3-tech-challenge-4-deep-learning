from pydantic import BaseModel

class FetchDataRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
