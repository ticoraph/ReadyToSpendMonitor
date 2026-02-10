"""
API FastAPI pour le modèle de scoring de crédit
"""
import os
import json
import time
import joblib
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from api.schemas import ClientData, PredictionResponse, HealthResponse

import cProfile
import pstats
import io

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Variables globales pour le modèle
MODEL = None
MODEL_VERSION = "1.0.0"
LOGS_FILE = "logs/production_logs.json"


def load_model():
    """
    Charge le modèle LightGBM au démarrage de l'application.
    Le modèle est chargé UNE SEULE FOIS.
    """
    global MODEL, MODEL_FEATURES

    model_path = "models/model.pkl"

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        artifact = joblib.load(model_path)

        # Cas recommandé : dict {"model": ..., "features": [...]}
        if isinstance(artifact, dict):
            MODEL = artifact["model"]
            MODEL_FEATURES = artifact.get("features")

        else:
            MODEL = artifact
            MODEL_FEATURES = None

        logger.info(f"✅ Modèle LightGBM chargé depuis {model_path}")
        logger.info(f"Type du modèle : {type(MODEL)}")

        # -----------------------------
        # 🔍 Extraction des features
        # -----------------------------
        if MODEL_FEATURES:
            logger.info(f"📌 Features (artifact) : {MODEL_FEATURES}")

        elif hasattr(MODEL, "feature_name_"):
            MODEL_FEATURES = list(MODEL.feature_name_)
            logger.info(f"📌 Features (LightGBM) : {MODEL_FEATURES}")

        else:
            raise RuntimeError(
                "Impossible de récupérer les features depuis le modèle LightGBM"
            )

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
        raise

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
        
        # Append au fichier de logs
        with open(LOGS_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture des logs: {e}")


@app.on_event("startup")
async def startup_event():
    """
    Événement exécuté au démarrage de l'API
    """
    logger.info("🚀 Démarrage de l'API de Scoring...")
    load_model()
    logger.info("✅ API prête à recevoir des requêtes")

'''
@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée de l'API
    """
    return {
        "message": "API de Scoring de Crédit - Prêt à Dépenser",
        "version": MODEL_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }
'''

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(client_data: ClientData, request: Request):
    """
    Effectue une prédiction de score de crédit
    Retourne un score de solvabilité entre 0 et 1, et une décision.
    """

    # Vérifier que le modèle est chargé
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:

        pr = cProfile.Profile()
        pr.enable()

        # Mesurer le temps d'inférence
        start_time = time.time()
        
        # Préparer les données pour la prédiction
        features = pd.DataFrame([client_data.model_dump()])
        
        # Prédiction
        try:
            # Essayer avec predict_proba (pour les classifieurs)
            probas = MODEL.predict_proba(features)
            score = float(probas[0][1])  # Probabilité de la classe positive
        except AttributeError:
            # Si pas de predict_proba, utiliser predict
            prediction = MODEL.predict(features)
            score = float(prediction[0])
        
        # Confidence basée sur le seuil métier (simple et utile)
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
        
        logger.info(f"✅ Prédiction réussie: {client_id} - Score: {score:.2f} - Décision: {decision}")
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # top 20 fonctions
        logger.info(s.getvalue())
        pr.dump_stats("output/predict.prof")

        return response
    


    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        version=MODEL_VERSION
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions
    """
    logger.error(f"Erreur non gérée: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
