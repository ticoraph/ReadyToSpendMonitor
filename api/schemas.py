"""
Schémas de validation pour l'API de scoring
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ClientData(BaseModel):
    """Données d'entrée du client pour la prédiction (features ML)"""

    #SK_ID_CURR: Optional[int] = Field(None)

    ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN: Optional[float] = Field(None)
    ACTIVE_AMT_CREDIT_SUM_MAX: Optional[float] = Field(None)
    ACTIVE_DAYS_CREDIT_MAX: Optional[float] = Field(None)

    AMT_ANNUITY: Optional[float] = Field(None)
    AMT_CREDIT: Optional[float] = Field(None)
    AMT_GOODS_PRICE: Optional[float] = Field(None)

    ANNUITY_INCOME_PERC: Optional[float] = Field(None)

    APPROVED_AMT_ANNUITY_MEAN: Optional[float] = Field(None)
    APPROVED_CNT_PAYMENT_MEAN: Optional[float] = Field(None)
    APPROVED_DAYS_DECISION_MAX: Optional[float] = Field(None)

    BURO_AMT_CREDIT_MAX_OVERDUE_MEAN: Optional[float] = Field(None)
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: Optional[float] = Field(None)
    BURO_DAYS_CREDIT_MAX: Optional[float] = Field(None)
    BURO_DAYS_CREDIT_MEAN: Optional[float] = Field(None)

    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: Optional[float] = Field(None)

    CLOSED_AMT_CREDIT_SUM_MAX: Optional[float] = Field(None)
    CLOSED_DAYS_CREDIT_ENDDATE_MAX: Optional[float] = Field(None)
    CLOSED_DAYS_CREDIT_MAX: Optional[float] = Field(None)
    CLOSED_DAYS_CREDIT_VAR: Optional[float] = Field(None)

    CODE_GENDER: Optional[float] = Field(None)

    DAYS_BIRTH: Optional[float] = Field(None)
    DAYS_EMPLOYED: Optional[float] = Field(None)
    DAYS_EMPLOYED_PERC: Optional[float] = Field(None)
    DAYS_ID_PUBLISH: Optional[float] = Field(None)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None)
    DAYS_REGISTRATION: Optional[float] = Field(None)

    EXT_SOURCE_1: Optional[float] = Field(None)
    EXT_SOURCE_2: Optional[float] = Field(None)
    EXT_SOURCE_3: Optional[float] = Field(None)

    INSTAL_AMT_PAYMENT_MEAN: Optional[float] = Field(None)
    INSTAL_AMT_PAYMENT_MIN: Optional[float] = Field(None)
    INSTAL_AMT_PAYMENT_SUM: Optional[float] = Field(None)
    INSTAL_DBD_MAX: Optional[float] = Field(None)
    INSTAL_DBD_SUM: Optional[float] = Field(None)
    INSTAL_DPD_MEAN: Optional[float] = Field(None)
    INSTAL_PAYMENT_PERC_MEAN: Optional[float] = Field(None)

    OWN_CAR_AGE: Optional[float] = Field(None)

    PAYMENT_RATE: Optional[float] = Field(None)

    POS_MONTHS_BALANCE_SIZE: Optional[float] = Field(None)

    PREV_CNT_PAYMENT_MEAN: Optional[float] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "DAYS_BIRTH": -14551,
                "AMT_CREDIT": 261648,
                "AMT_ANNUITY": 12856.5,
                "EXT_SOURCE_1": 0.53,
                "PAYMENT_RATE": 0.049
            }
        }

    # -----------------------
    # Validators métier
    # -----------------------

    @field_validator("DAYS_BIRTH")
    @classmethod
    def days_birth_must_be_adult(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > -6570:
            raise ValueError("DAYS_BIRTH doit être ≤ -6570 (âge minimum ≈ 18 ans)")
        return v

    @field_validator(
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
        "INSTAL_PAYMENT_PERC_MEAN",
    )
    @classmethod
    def ratios_must_be_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Les ratios ne peuvent pas être négatifs")
        return v

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
                "client_id": "req_20260205103137658530",
                "score": 0.1883,
                "decision": "REJECTED",
                "confidence": 0.8117,
                "inference_time_ms": 9.67
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
