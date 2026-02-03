"""
Schémas de validation pour l'API de scoring
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ClientData(BaseModel):
    """Données d'entrée du client pour la prédiction"""
    
    age: int = Field(..., ge=18, le=100, description="Âge du client")
    income: float = Field(..., gt=0, description="Revenu annuel en euros")
    loan_amount: float = Field(..., gt=0, description="Montant du prêt demandé")
    employment_length: int = Field(..., ge=0, le=50, description="Ancienneté professionnelle en années")
    credit_score: int = Field(..., ge=300, le=850, description="Score de crédit")
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 18:
            raise ValueError("Le client doit être majeur (18 ans minimum)")
        if v > 100:
            raise ValueError("Âge invalide")
        return v
    
    @field_validator('income', 'loan_amount')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("La valeur doit être positive")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 50000,
                "loan_amount": 15000,
                "employment_length": 5,
                "credit_score": 720
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    
    client_id: str = Field(..., description="Identifiant unique de la requête")
    score: float = Field(..., ge=0, le=1, description="Score de solvabilité (0-1)")
    decision: str = Field(..., description="Décision: APPROVED ou REJECTED")
    confidence: float = Field(..., ge=0, le=1, description="Confiance de la prédiction")
    inference_time_ms: float = Field(..., description="Temps d'inférence en millisecondes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_id": "req_123456",
                "score": 0.78,
                "decision": "APPROVED",
                "confidence": 0.85,
                "inference_time_ms": 12.5
            }
        }


class HealthResponse(BaseModel):
    """Réponse du health check"""
    
    status: str = Field(..., description="État du service")
    model_loaded: bool = Field(..., description="Modèle chargé")
    version: str = Field(..., description="Version de l'API")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
