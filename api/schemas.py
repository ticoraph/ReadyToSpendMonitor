"""
Schémas de validation pour l'API de scoring
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ClientData(BaseModel):
    """Données d'entrée du client pour la prédiction (features ML)"""

    EXT_SOURCE_3: Optional[float] = Field(None)
    EXT_SOURCE_2: Optional[float] = Field(None)
    EXT_SOURCE_1: Optional[float] = Field(None)

    PAYMENT_RATE: Optional[float] = Field(None)

    DAYS_EMPLOYED: Optional[float] = Field(None)
    DAYS_REGISTRATION: Optional[float] = Field(None)
    DAYS_BIRTH: Optional[int] = Field(None)
    DAYS_ID_PUBLISH: Optional[int] = Field(None)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None)

    DAYS_EMPLOYED_PERC: Optional[float] = Field(None)

    AMT_ANNUITY: Optional[float] = Field(None)
    AMT_GOODS_PRICE: Optional[float] = Field(None)
    AMT_CREDIT: Optional[float] = Field(None)

    REGION_POPULATION_RELATIVE: Optional[float] = Field(None)

    INSTAL_DBD_MEAN: Optional[float] = Field(None)
    INSTAL_DBD_SUM: Optional[float] = Field(None)
    INSTAL_DBD_MAX: Optional[float] = Field(None)

    INSTAL_AMT_PAYMENT_MIN: Optional[float] = Field(None)
    INSTAL_AMT_PAYMENT_MAX: Optional[float] = Field(None)

    INSTAL_DAYS_ENTRY_PAYMENT_MAX: Optional[float] = Field(None)
    INSTAL_DAYS_ENTRY_PAYMENT_SUM: Optional[float] = Field(None)

    ANNUITY_INCOME_PERC: Optional[float] = Field(None)
    INCOME_CREDIT_PERC: Optional[float] = Field(None)
    INCOME_PER_PERSON: Optional[float] = Field(None)

    ACTIVE_DAYS_CREDIT_ENDDATE_MIN: Optional[float] = Field(None)
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: Optional[float] = Field(None)
    ACTIVE_DAYS_CREDIT_MAX: Optional[float] = Field(None)
    ACTIVE_DAYS_CREDIT_MEAN: Optional[float] = Field(None)

    BURO_DAYS_CREDIT_VAR: Optional[float] = Field(None)
    BURO_AMT_CREDIT_SUM_MEAN: Optional[float] = Field(None)

    CLOSED_DAYS_CREDIT_MAX: Optional[float] = Field(None)

    PREV_APP_CREDIT_PERC_VAR: Optional[float] = Field(None)
    PREV_APP_CREDIT_PERC_MEAN: Optional[float] = Field(None)

    APPROVED_DAYS_DECISION_MAX: Optional[float] = Field(None)
    APPROVED_APP_CREDIT_PERC_VAR: Optional[float] = Field(None)

    PREV_DAYS_DECISION_MAX: Optional[float] = Field(None)
    PREV_HOUR_APPR_PROCESS_START_MEAN: Optional[float] = Field(None)

    POS_MONTHS_BALANCE_MEAN: Optional[float] = Field(None)
    POS_NAME_CONTRACT_STATUS_Active_MEAN: Optional[float] = Field(None)
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: Optional[float] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_3": 0.15,
                "EXT_SOURCE_2": 0.78,
                "AMT_CREDIT": 250000,
                "DAYS_BIRTH": -12000
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
