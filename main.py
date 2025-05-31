from fastapi import FastAPI
from dotenv import load_dotenv

from sklearn.preprocessing import MinMaxScaler

from schemas.fetch_data import FetchDataRequest
from schemas.train import TrainModelRequest

from services.predict.predict_service import PredictService
from services.predict.prepare_data_service import PredictPrepareDataService
from services.yfinance_service import YFinanceService
from services.preprocess_data_service import PreprocessDataService

from services.train.prepare_data_service import TrainPrepareDataService
from services.train.train_service import TrainService
from services.train.evaluate_service import TrainEvaluateService

from services.s3.upload_service import S3UploadService
from services.s3.download_service import S3DownloadService

from error_handlers import http_exception_handler, validation_exception_handler, generic_exception_handler, value_error_handler, file_not_found_error_handler
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

load_dotenv()

app = FastAPI()

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(FileNotFoundError, file_not_found_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

@app.get("/up")
def up():
    return {
        "status": "ok"
    }

@app.post("/models/train")
def train_model(request: TrainModelRequest):
    yfinance_data = YFinanceService(
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date
    ).execute()

    preprocessed_data = PreprocessDataService(data=yfinance_data).execute()

    X_train, y_train, X_test, y_test, scaler = TrainPrepareDataService(
        data=preprocessed_data,
        train_size=request.train_size,
        sequence_length=request.sequence_length,
        scaler=MinMaxScaler,
        target_column=request.target_column
    ).execute()

    model = TrainService(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=request.epochs
    ).execute()

    train_metrics, test_metrics = TrainEvaluateService(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    ).execute()

    metadata = {
        "request": request.model_dump(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "scaler": "MinMaxScaler"
    }

    id, model_s3_path, scaler_s3_path, metadata_s3_path = S3UploadService(
        model=model,
        scaler=scaler,
        metadata=metadata
    ).execute()

    return {
        "message": "Modelo treinado com sucesso",
        "result": {
            "id": id,
            "metrics": {
                "train": train_metrics,
                "test": test_metrics
            },
            "paths": {
                "model_s3_path": model_s3_path,
                "scaler_s3_path": scaler_s3_path,
                "metadata_s3_path": metadata_s3_path
            }
        }
    }

@app.post("/models/{model_id}/predict")
def predict(model_id: str):
    model, scaler, metadata = S3DownloadService(id=model_id).execute()

    X_predict = PredictPrepareDataService(
        metadata=metadata,
        scaler=scaler
    ).execute()

    prediction = PredictService(model=model, X_predict=X_predict).execute()

    return {
        "prediction": prediction,
    }

@app.post("/models/fetch-data")
def fetch_stock_data(request: FetchDataRequest):
    data = YFinanceService(
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        days=request.days
    ).execute()
  
    return {
        "ticker": request.ticker,
        "data": data.to_dict(orient="records")
    }
