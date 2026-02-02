"""API FastAPI pour le modèle de scoring."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import time
import logging
from datetime import datetime
import json

from src.model import get_model, ScoringModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="ReadyToSpend API",
    description="API de scoring de crédit pour Prêt à Dépenser",
    version="0.1.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schémas Pydantic
class ClientFeatures(BaseModel):
    """Modèle des données d'entrée pour le scoring."""

    age: int = Field(..., ge=18, le=100, description="Âge du client")
    income: float = Field(..., gt=0, description="Revenu annuel")
    employment_length: float = Field(..., ge=0, description="Ancienneté professionnelle (années)")
    debt_ratio: float = Field(..., ge=0, le=1, description="Ratio d'endettement")
    credit_history: int = Field(..., ge=0, description="Historique de crédit (années)")
    num_accounts: int = Field(..., ge=0, description="Nombre de comptes")
    num_late_payments: int = Field(..., ge=0, description="Nombre de retards de paiement")
    home_ownership: int = Field(..., ge=0, le=2, description="Statut de propriété (0: locataire, 1: propriétaire, 2: autre)")
    loan_amount: float = Field(..., gt=0, description="Montant du prêt demandé")
    loan_term: int = Field(..., gt=0, description="Durée du prêt (mois)")

    @validator('income', 'loan_amount')
    def positive_values(cls, v):
        if v <= 0:
            raise ValueError('doit être positif')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 50000.0,
                "employment_length": 5.0,
                "debt_ratio": 0.3,
                "credit_history": 10,
                "num_accounts": 3,
                "num_late_payments": 1,
                "home_ownership": 1,
                "loan_amount": 15000.0,
                "loan_term": 36
            }
        }


class ScoringResponse(BaseModel):
    """Réponse de l'API de scoring."""

    score: Optional[int] = Field(None, description="Score de prédiction (0 ou 1)")
    probability: Optional[float] = Field(None, ge=0, le=1, description="Probabilité de défaut")
    status: str = Field(..., description="Statut de la requête")
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
    inference_time: float = Field(..., description="Temps d'inférence en secondes")
    timestamp: str = Field(..., description="Horodatage de la requête")


# Chargement du modèle au démarrage
model = get_model()


@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'API."""
    logger.info("API démarrée - Modèle chargé avec succès")


@app.get("/", tags=["Root"])
async def root():
    """Point de terminaison racine."""
    return {
        "message": "ReadyToSpend API - Modèle de scoring de crédit",
        "version": "0.1.0",
        "status": "operational"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Vérification de santé de l'API."""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Informations sur le modèle chargé."""
    return {
        "model_path": str(model.model_path),
        "feature_names": model.get_feature_names(),
        "model_type": str(type(model.model).__class__.__name__)
    }


@app.post("/predict", response_model=ScoringResponse, tags=["Prediction"])
async def predict(features: ClientFeatures, request: Request):
    """
    Effectue une prédiction de scoring pour un client.

    Args:
        features: Caractéristiques du client
        request: Objet Request FastAPI

    Returns:
        ScoringResponse: Résultat de la prédiction avec métriques
    """
    start_time = time.time()

    try:
        # Log de la requête
        logger.info(f"Nouvelle requête de prédiction - Client ID: {request.client}")

        # Convertir en dict pour le modèle
        features_dict = features.dict()

        # Prédiction
        result = model.predict(features_dict)

        # Calcul du temps d'inférence
        inference_time = time.time() - start_time

        # Création de la réponse
        response = ScoringResponse(
            score=result.get("score"),
            probability=result.get("probability"),
            status=result.get("status", "success"),
            error=result.get("error"),
            inference_time=inference_time,
            timestamp=datetime.now().isoformat()
        )

        # Log structuré pour monitoring
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": features_dict,
            "output": {"score": response.score, "probability": response.probability},
            "inference_time": inference_time,
            "status": response.status,
            "client_ip": str(request.client.host) if request.client else None
        }
        _log_prediction(log_entry)

        return response

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _log_prediction(log_entry: Dict[str, Any]):
    """Logge la prédiction dans un fichier JSON."""
    import os
    from pathlib import Path

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "predictions.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)