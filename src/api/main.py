"""FastAPI application for scoring model predictions."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from ..config import settings
from ..logger import logger
from ..models.model import get_model
from .schemas import ClientFeatures, PredictionResponse, HealthResponse, ErrorResponse


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for credit scoring predictions",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up API...")
    try:
        model = get_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Prêt à Dépenser Scoring API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model.model is not None,
            version=settings.api_version
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version=settings.api_version
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ClientFeatures, request: Request):
    """
    Make a credit scoring prediction.

    Returns a prediction (0 or 1) and probability if available.
    """
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    try:
        # Get model instance
        model = get_model()

        # Convert features to dictionary
        feature_dict = features.model_dump(exclude_unset=True)

        # Log prediction request
        logger.info(
            "Prediction request",
            extra={
                "extra_data": {
                    "features": feature_dict,
                    "timestamp": timestamp
                }
            }
        )

        # Make prediction
        result = model.predict(feature_dict)

        # Calculate inference time
        inference_time = time.time() - start_time

        # Log prediction result
        logger.info(
            "Prediction result",
            extra={
                "extra_data": {
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "timestamp": timestamp
                }
            }
        )

        # Log to production data file for drift analysis
        _log_prediction(feature_dict, result, inference_time)

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            timestamp=timestamp,
            model_version=settings.api_version
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", extra={"extra_data": {"features": feature_dict}})
        raise HTTPException(status_code=500, detail=str(e))


def _log_prediction(features: Dict[str, Any], result: Dict[str, Any], inference_time: float) -> None:
    """Log prediction data to CSV for drift analysis."""
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "inference_time_ms": round(inference_time * 1000, 2),
            "prediction": result["prediction"],
            "probability": result["probability"],
        }
        log_entry.update(features)

        # Append to CSV
        log_path = Path(settings.production_data_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([log_entry])

        if log_path.exists():
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, index=False)

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )