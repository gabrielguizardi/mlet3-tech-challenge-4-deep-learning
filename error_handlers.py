from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST


def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)},
    )

def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

def file_not_found_error_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=HTTP_404_NOT_FOUND,
        content={"detail": "File not found", "error": str(exc)},
    )
