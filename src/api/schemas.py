"""API schemas for request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class ClientFeatures(BaseModel):
    """Input features for scoring prediction."""

    # Example features - adjust based on actual model
    age: Optional[int] = Field(None, ge=18, le=100, description="Client age in years")
    income: Optional[float] = Field(None, ge=0, description="Annual income")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    debt_ratio: Optional[float] = Field(None, ge=0, le=1, description="Debt to income ratio")
    employment_length: Optional[int] = Field(None, ge=0, description="Years of employment")
    loan_amount: Optional[float] = Field(None, ge=0, description="Requested loan amount")
    loan_term: Optional[int] = Field(None, ge=1, le=30, description="Loan term in years")
    home_ownership: Optional[str] = Field(None, description="Home ownership status")
    purpose: Optional[str] = Field(None, description="Purpose of the loan")

    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 18 or v > 100):
            raise ValueError('Age must be between 18 and 100')
        return v

    @validator('credit_score')
    def validate_credit_score(cls, v):
        if v is not None and (v < 300 or v > 850):
            raise ValueError('Credit score must be between 300 and 850')
        return v

    @validator('debt_ratio')
    def validate_debt_ratio(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Debt ratio must be between 0 and 1')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 50000.0,
                "credit_score": 720,
                "debt_ratio": 0.3,
                "employment_length": 5,
                "loan_amount": 15000.0,
                "loan_term": 3,
                "home_ownership": "RENT",
                "purpose": "debt_consolidation"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: Optional[float] = Field(None, description="Probability of positive class")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(default="1.0.0", description="Model version")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "timestamp": "2024-01-01T12:00:00",
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")