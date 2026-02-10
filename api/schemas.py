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
"ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5, "ACTIVE_AMT_CREDIT_SUM_MAX": 450000.0, "ACTIVE_DAYS_CREDIT_MAX": -753.0, "AMT_ANNUITY": 10548.0, "AMT_CREDIT": 148365.0, "AMT_GOODS_PRICE": 135000.0, "ANNUITY_INCOME_PERC": 0.1019130434782608, "APPROVED_AMT_ANNUITY_MEAN": 6340.785, "APPROVED_CNT_PAYMENT_MEAN": 14.666666666666666, "APPROVED_DAYS_DECISION_MAX": -348.0, "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5, "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0.0, "BURO_DAYS_CREDIT_MAX": -753.0, "BURO_DAYS_CREDIT_MEAN": -979.6666666666666, "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.2666666666666666, "CLOSED_AMT_CREDIT_SUM_MAX": 38650.5, "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -943.0, "CLOSED_DAYS_CREDIT_MAX": -1065.0, "CLOSED_DAYS_CREDIT_VAR": 256328.0, "CODE_GENDER": 1.0, "DAYS_BIRTH": -11716.0, "DAYS_EMPLOYED": -449.0, "DAYS_EMPLOYED_PERC": 0.0383236599522021, "DAYS_ID_PUBLISH": -3961.0, "DAYS_LAST_PHONE_CHANGE": -1420.0, "DAYS_REGISTRATION": -3997.0, "EXT_SOURCE_1": 0.3608707365728421, "EXT_SOURCE_2": 0.4285392216965799, "EXT_SOURCE_3": 0.7981372313187245, "INSTAL_AMT_PAYMENT_MEAN": 10274.82081081081, "INSTAL_AMT_PAYMENT_MIN": 2.7, "INSTAL_AMT_PAYMENT_SUM": 380168.37, "INSTAL_DBD_MAX": 60.0, "INSTAL_DBD_SUM": 833.0, "INSTAL_DPD_MEAN": 0.4594594594594595, "INSTAL_PAYMENT_PERC_MEAN": 0.945945945945946, "OWN_CAR_AGE": 9.0, "PAYMENT_RATE": 0.0710949347892023, "POS_MONTHS_BALANCE_SIZE": 40.0, "PREV_CNT_PAYMENT_MEAN": 15.142857142857142
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
