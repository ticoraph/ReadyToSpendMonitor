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
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cProfile
import pstats
import io

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables globales pour le modèle
MODEL = None
MODEL_VERSION = "1.0.0"
LOGS_FILE = "logs/production_logs.json"
EXECUTOR = ThreadPoolExecutor(max_workers=10)  # 10 workers parallèles

# ========== Configuration du profiling ==========
ENABLE_PROFILING = False  # 👈 Modifier pour activer/désactiver le profiling
PROFILE_OUTPUT_DIR = "output"

# Créer le dossier output si nécessaire
os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)

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

def profile_function(func):
    """
    Décorateur pour profiler une fonction avec cProfile.
    Active/désactive le profiling selon la variable ENABLE_PROFILING.
    Sauvegarde les stats dans un fichier .prof et les affiche dans les logs.
    """
    from functools import wraps
    
    @wraps(func)  # ✅ Important pour préserver les métadonnées de la fonction
    async def wrapper(*args, **kwargs):
        if not ENABLE_PROFILING:
            # Si le profiling est désactivé, exécuter la fonction normalement
            return await func(*args, **kwargs)
        
        # Profiling activé
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            pr.disable()
            
            # Générer les stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats(20)  # top 20 fonctions
            
            # Afficher dans les logs
            logger.info(f"\n📊 Profiling Results for {func.__name__}:\n{s.getvalue()}")
            
            # Sauvegarder le fichier .prof
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_file = os.path.join(PROFILE_OUTPUT_DIR, f"{func.__name__}_{timestamp}.prof")
            pr.dump_stats(profile_file)
            logger.info(f"✅ Profiling stats sauvegardées dans {profile_file}")
    
    return wrapper

def _process_single_prediction(client_data: ClientData, model):
    """
    Fonction interne pour traiter une seule prédiction
    (utilisée par le ThreadPoolExecutor)
    """
    try:
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
        
        return response

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("🚀 Démarrage de l'API de Scoring...")
    load_model()
    logger.info("✅ API prête à recevoir des requêtes")
    app.state.model = MODEL
    yield
    
    # Shutdown propre du executor
    logger.info("🛑 Arrêt de l'API...")
    EXECUTOR.shutdown(wait=True)
    logger.info("✅ API arrêtée")

# Création de l'application FastAPI
app = FastAPI(
    title="API de Scoring de Crédit",
    description="API pour prédire la solvabilité des demandes de crédit",
    version="1.0.0",
    lifespan=lifespan   # 👈 ICI
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
@profile_function
async def predict(client_data: ClientData, request: Request):
    """
    Effectue une prédiction de score de crédit
    Retourne un score de solvabilité entre 0 et 1, et une décision.
    
    🎯 Le profiling cProfile peut être activé/désactivé avec la variable ENABLE_PROFILING
    """
    model = request.app.state.model

    try:
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

        return response
    

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict_batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(clients_data: List[ClientData], request: Request):
    """
    Effectue des prédictions en parallèle pour plusieurs clients.
    Utilise un ThreadPoolExecutor pour paralleliser les requêtes.
    
    Exemple de payload:
    [
        {"feature1": 1.0, "feature2": 2.0, ...},
        {"feature1": 3.0, "feature2": 4.0, ...},
        ...
    ]
    """
    try:
        # Récupérer le modèle depuis l'app state
        model = request.app.state.model
        
        # Vérifier que le modèle est chargé
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Soumettre toutes les tâches au thread pool
        futures = [
            EXECUTOR.submit(_process_single_prediction, client, model)
            for client in clients_data
        ]
        
        # Attendre les résultats
        responses = []
        for future in futures:
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                logger.error(f"Erreur lors du traitement batch: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Erreur batch: {str(e)}")
        
        logger.info(f"✅ Batch traité: {len(responses)} prédictions réussies")
        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction batch: {str(e)}")


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