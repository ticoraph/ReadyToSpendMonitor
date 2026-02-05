"""
Schémas de validation pour l'API de scoring
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ClientData(BaseModel):
    """Données d'entrée du client pour la prédiction (features ML)"""
    ACTIVE_DAYS_CREDIT_ENDDATE_MIN: float = Field(...)
    ACTIVE_DAYS_CREDIT_MAX: float = Field(...)
    ACTIVE_DAYS_CREDIT_MEAN: float = Field(...)
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: float = Field(...)

    AMT_ANNUITY: float = Field(...)
    AMT_CREDIT: float = Field(...)
    AMT_GOODS_PRICE: float = Field(...)

    ANNUITY_INCOME_PERC: float = Field(...)

    APPROVED_APP_CREDIT_PERC_VAR: float = Field(...)
    APPROVED_DAYS_DECISION_MAX: float = Field(...)

    BURO_AMT_CREDIT_SUM_MEAN: float = Field(...)
    BURO_DAYS_CREDIT_VAR: float = Field(...)

    CLOSED_DAYS_CREDIT_MAX: float = Field(...)

    DAYS_BIRTH: float = Field(...)
    DAYS_EMPLOYED: float = Field(...)
    DAYS_EMPLOYED_PERC: float = Field(...)
    DAYS_ID_PUBLISH: float = Field(...)
    DAYS_LAST_PHONE_CHANGE: float = Field(...)
    DAYS_REGISTRATION: float = Field(...)

    EXT_SOURCE_1: float = Field(...)
    EXT_SOURCE_2: float = Field(...)
    EXT_SOURCE_3: float = Field(...)

    INCOME_CREDIT_PERC: float = Field(...)
    INCOME_PER_PERSON: float = Field(...)

    INSTAL_AMT_PAYMENT_MAX: float = Field(...)
    INSTAL_AMT_PAYMENT_MIN: float = Field(...)
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: float = Field(...)
    INSTAL_DAYS_ENTRY_PAYMENT_SUM: float = Field(...)
    INSTAL_DBD_MAX: float = Field(...)
    INSTAL_DBD_MEAN: float = Field(...)
    INSTAL_DBD_SUM: float = Field(...)

    PAYMENT_RATE: float = Field(...)

    POS_MONTHS_BALANCE_MEAN: float = Field(...)
    POS_NAME_CONTRACT_STATUS_Active_MEAN: float = Field(...)
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: float = Field(...)

    PREV_APP_CREDIT_PERC_MEAN: float = Field(...)
    PREV_APP_CREDIT_PERC_VAR: float = Field(...)
    PREV_DAYS_DECISION_MAX: float = Field(...)
    PREV_HOUR_APPR_PROCESS_START_MEAN: float = Field(...)

    REGION_POPULATION_RELATIVE: float = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_3": 0.26,
                "EXT_SOURCE_2": 0.68,
                "PAYMENT_RATE": 0.07,
                "DAYS_EMPLOYED": -1000,
                "DAYS_REGISTRATION": -5000,
                "EXT_SOURCE_1": 0.34,
                "DAYS_BIRTH": -13000,
                "DAYS_ID_PUBLISH": -4000,
                "DAYS_EMPLOYED_PERC": 0.07,
                "AMT_ANNUITY": 21000.0,
                "REGION_POPULATION_RELATIVE": 0.03,
                "INSTAL_DBD_MEAN": 4,
                "ANNUITY_INCOME_PERC": 0.12,
                "INSTAL_DBD_SUM": 170,
                "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 350,
                "DAYS_LAST_PHONE_CHANGE": -300,
                "INSTAL_AMT_PAYMENT_MIN": 37,
                "INCOME_CREDIT_PERC": 0.60,
                "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -4,
                "BURO_DAYS_CREDIT_VAR": 0,
                "ACTIVE_DAYS_CREDIT_UPDATE_MEAN": -17,
                "APPROVED_DAYS_DECISION_MAX": -320,
                "AMT_GOODS_PRICE": 225000,
                "CLOSED_DAYS_CREDIT_MAX": 0,
                "PREV_APP_CREDIT_PERC_VAR": 0.01,
                "PREV_APP_CREDIT_PERC_MEAN": 0.93,
                "INSTAL_DBD_MAX": 20.0,
                "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -6000,
                "BURO_AMT_CREDIT_SUM_MEAN": 12400,
                "POS_MONTHS_BALANCE_MEAN": -6,
                "INCOME_PER_PERSON": 90000,
                "POS_NAME_CONTRACT_STATUS_Active_MEAN": 0.90,
                "POS_NAME_CONTRACT_STATUS_Completed_MEAN": 0.09,
                "PREV_HOUR_APPR_PROCESS_START_MEAN": 20,
                "ACTIVE_DAYS_CREDIT_MAX": -17,
                "ACTIVE_DAYS_CREDIT_MEAN": -17,
                "APPROVED_APP_CREDIT_PERC_VAR": 0.01,
                "AMT_CREDIT": 296000,
                "INSTAL_AMT_PAYMENT_MAX": 22500,
                "PREV_DAYS_DECISION_MAX": -320
            }
        }

    # -----------------------
    # Validators métier
    # -----------------------

    @field_validator("DAYS_BIRTH")
    @classmethod
    def days_birth_must_be_adult(cls, v: int) -> int:
        if v > -6570:
            raise ValueError("DAYS_BIRTH doit être ≤ -6570 (âge minimum ≈ 18 ans)")
        return v

    @field_validator("INCOME_PER_PERSON")
    @classmethod
    def income_per_person_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("INCOME_PER_PERSON ne peut pas être négatif")
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

