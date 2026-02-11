"""
API FastAPI pour le modèle de scoring de crédit
"""
import os
import json
import time
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from api.schemas import ClientData, PredictionResponse, HealthResponse

MODEL_VERSION = "1.0.0"
LOGS_FILE = "logs/production_logs.json"

def load_model():
    """
    Charge le modèle LightGBM
    """
    model_path = "models/model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    artifact = joblib.load(model_path)

    # Cas recommandé : dict {"model": ..., "features": [...]}
    if isinstance(artifact, dict):
        model = artifact["model"]
        features = artifact.get("features")
    else:
        model = artifact
        features = None

    # Extraction des features
    if features:
        model_features = features
    elif hasattr(model, "feature_name_"):
        model_features = list(model.feature_name_)
    else:
        raise RuntimeError(
            "Impossible de récupérer les features depuis le modèle LightGBM"
        )

    return model, model_features

def log_prediction(client_data: dict, prediction: dict):
    """
    Enregistre les prédictions dans un fichier JSON pour le monitoring
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": client_data,
            "output": prediction,
            "model_version": MODEL_VERSION
        }
        
        with open(LOGS_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        pass

# Création de l'application FastAPI
app = FastAPI(
    title="API de Scoring de Crédit",
    description="API pour prédire la solvabilité des demandes de crédit",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(client_data: ClientData, request: Request):
    """
    Effectue une prédiction de score de crédit
    Retourne un score de solvabilité entre 0 et 1, et une décision.
    """
    try:
        # Charger le modèle
        model, model_features = load_model()

        # Mesurer le temps d'inférence
        start_time = time.time()
        
        # Préparer les données pour la prédiction
        features = pd.DataFrame([client_data.model_dump()])
        
        # Prédiction
        try:
            # Essayer avec predict_proba (pour les classifieurs)
            probas = model.predict_proba(features)
            score = float(probas[0][1])  # Probabilité de la classe positive
        except AttributeError:
            # Si pas de predict_proba, utiliser predict
            prediction = model.predict(features)
            score = float(prediction[0])
        
        # Confidence basée sur le seuil métier
        confidence = abs(score - 0.2) / 0.8
        confidence = min(confidence, 1.0)
        decision = "APPROVED" if score >= 0.2 else "REJECTED"
        
        # Temps d'inférence
        inference_time = (time.time() - start_time) * 1000  # en ms
        
        # Générer un ID unique pour la requête
        client_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Préparer la réponse
        response = PredictionResponse(
            client_id=client_id,
            score=round(score, 4),
            decision=decision,
            confidence=round(confidence, 4),
            inference_time_ms=round(inference_time, 2)
        )
        
        # Logger la prédiction
        log_prediction(
            client_data=client_data.model_dump(),
            prediction=response.model_dump()
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API
    """
    try:
        load_model()
        status = "healthy"
        model_loaded = True
    except:
        status = "unhealthy"
        model_loaded = False

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version=MODEL_VERSION
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions
    """
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)