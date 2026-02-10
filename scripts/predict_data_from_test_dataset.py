#!/usr/bin/env python3
"""
Script de test rapide de l'API
"""
import requests
import json
import time
import pandas as pd
import numpy as np
import logging
import sys
import os

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CSV_PATH = "output/dataset_test_top40.csv"

def test_health():
    """Test du health check"""
    print("🔍 Test Health Check...")
    logger.info("🔍 Test Health Check")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    logger.info(f"Health status={response.status_code} response={response.json()}")

def test_prediction(client_data):
    """Test d'une prédiction"""
    print(f"🎯 Test Prédiction...")
    print(f"   Input: {json.dumps(client_data)}")

    logger.info("🎯 Test Prédiction")
    logger.debug(f"Payload: {json.dumps(client_data)}")
    
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=client_data)
    duration = (time.time() - start) * 1000
    
    print(f"   Status: {response.status_code}")
    print(f"   Duration: {duration:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Score: {result['score']}")
        print(f"   Decision: {result['decision']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Inference Time: {result['inference_time_ms']}ms")
    else:
        print(f"   Error: {response.text}")
    print()

    if response.ok:
        result = response.json()
        logger.info(
            f"Prediction OK | "
            f"score={result['score']} | "
            f"decision={result['decision']} | "
            f"confidence={result['confidence']} | "
            f"time={duration:.2f}ms"
        )
    else:
        logger.error(
            f"Prediction FAILED | "
            f"status={response.status_code} | "
            f"error={response.text}"
        )

def main():
    print("=" * 60)
    print("🚀 TEST DE L'API DE SCORING")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    try:
        test_health()
    except Exception as e:
        print(f"❌ Erreur Health Check: {e}")
        print("⚠️ Assurez-vous que l'API est lancée (uvicorn api.main:app)")
        return
    
    # Test 2: Prédiction avec bon profil
    print("📊 Scénario 1: MAUVAIS PROFIL")
    test_prediction({
"ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 0.0, "ACTIVE_AMT_CREDIT_SUM_MAX": 377545.5, "ACTIVE_DAYS_CREDIT_MAX": -377.0, "AMT_ANNUITY": 23944.5, "AMT_CREDIT": 495000.0, "AMT_GOODS_PRICE": 495000.0, "ANNUITY_INCOME_PERC": 0.1773666666666666, "APPROVED_AMT_ANNUITY_MEAN": 7109.11125, "APPROVED_CNT_PAYMENT_MEAN": 10.5, "APPROVED_DAYS_DECISION_MAX": -278.0, "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 914.56875, "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 75163.5, "BURO_DAYS_CREDIT_MAX": -377.0, "BURO_DAYS_CREDIT_MEAN": -911.4, "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.5, "CLOSED_AMT_CREDIT_SUM_MAX": 55345.5, "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -284.0, "CLOSED_DAYS_CREDIT_MAX": -466.0, "CLOSED_DAYS_CREDIT_VAR": 586734.3333333333, "CODE_GENDER": 1.0, "DAYS_BIRTH": -18574.0, "DAYS_EMPLOYED": -1645.5, "DAYS_EMPLOYED_PERC": 0.1196395813453313, "DAYS_ID_PUBLISH": -2092.0, "DAYS_LAST_PHONE_CHANGE": -1123.0, "DAYS_REGISTRATION": -9002.0, "EXT_SOURCE_1": 0.5084813074095602, "EXT_SOURCE_2": 0.3970079832932257, "EXT_SOURCE_3": 0.6479768603302221, "INSTAL_AMT_PAYMENT_MEAN": 6758.068499999999, "INSTAL_AMT_PAYMENT_MIN": 13.275, "INSTAL_AMT_PAYMENT_SUM": 270322.74, "INSTAL_DBD_MAX": 24.0, "INSTAL_DBD_SUM": 285.0, "INSTAL_DPD_MEAN": 0.125, "INSTAL_PAYMENT_PERC_MEAN": 1.0, "OWN_CAR_AGE": 9.0, "PAYMENT_RATE": 0.0483727272727272, "POS_MONTHS_BALANCE_SIZE": 28.0, "PREV_CNT_PAYMENT_MEAN": 15.0
    })
    
    # Test 3: Prédiction avec profil risqué
    print("📊 Scénario 2: BON PROFIL")
    test_prediction({
    "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 0.0, "ACTIVE_AMT_CREDIT_SUM_MAX": 990000.0, "ACTIVE_DAYS_CREDIT_MAX": -574.0, "AMT_ANNUITY": 23197.5, "AMT_CREDIT": 443088.0, "AMT_GOODS_PRICE": 382500.0, "ANNUITY_INCOME_PERC": 0.128875, "APPROVED_AMT_ANNUITY_MEAN": 9194.94, "APPROVED_CNT_PAYMENT_MEAN": 12.0, "APPROVED_DAYS_DECISION_MAX": -251.0, "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 0.0, "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 261937.125, "BURO_DAYS_CREDIT_MAX": -574.0, "BURO_DAYS_CREDIT_MEAN": -1464.5, "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.2666666666666666, "CLOSED_AMT_CREDIT_SUM_MAX": 405000.0, "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -808.0, "CLOSED_DAYS_CREDIT_MAX": -1448.0, "CLOSED_DAYS_CREDIT_VAR": 102604.5, "CODE_GENDER": 0.0, "DAYS_BIRTH": -17872.0, "DAYS_EMPLOYED": -2038.0, "DAYS_EMPLOYED_PERC": 0.1140331244404655, "DAYS_ID_PUBLISH": -1409.0, "DAYS_LAST_PHONE_CHANGE": -251.0, "DAYS_REGISTRATION": -9341.0, "EXT_SOURCE_1": 0.4846335396141397, "EXT_SOURCE_2": 0.6342419176819181, "EXT_SOURCE_3": 0.2735646775174348, "INSTAL_AMT_PAYMENT_MEAN": 9194.94, "INSTAL_AMT_PAYMENT_MIN": 9194.94, "INSTAL_AMT_PAYMENT_SUM": 73559.52, "INSTAL_DBD_MAX": 27.0, "INSTAL_DBD_SUM": 134.0, "INSTAL_DPD_MEAN": 0.0, "INSTAL_PAYMENT_PERC_MEAN": 1.0, "OWN_CAR_AGE": 10.0, "PAYMENT_RATE": 0.0523541598960026, "POS_MONTHS_BALANCE_SIZE": 8.0, "PREV_CNT_PAYMENT_MEAN": 12.0
    })
    
    #########

    def load_dataset(path: str) -> pd.DataFrame:

        logger.info(f"📄 Chargement du dataset : {CSV_PATH}")
        df = pd.read_csv(path)
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])
        
        logger.info(f"Dataset chargé : {df.shape[0]} lignes / {df.shape[1]} features")

        # Afficher les infos
        print("\n🔍 Diagnostic:")
        print(f"Shape: {df.shape}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"NaN: {df.isna().sum().sum()}")
        print(f"Inf: {(np.isinf(df)).sum().sum()}")

        # Tester la première ligne
        row = df.iloc[0]
        payload = {k: float(v) for k, v in row.to_dict().items()}
        print(f"\n✅ Payload valide (ligne 0): {list(payload.keys())}")




            # ✅ Forcer la conversion en float
        df = df.astype(float, errors='ignore')

        # Remplacer Inf/-Inf par NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Calculer la médiane de chaque colonne (ignorer les NaN)
        medians = df.median()
        logger.info(f"Colonnes avec NaN: {df.isna().sum().sum()}")

        # Remplacer NaN par la médiane
        df = df.fillna(medians)

        return df
    
    def send_row_to_api(row: pd.Series, api_url: str):
        payload = row.to_dict()
        print(f"   Payload: {json.dumps(payload)}")

        start = time.time()
        response = requests.post(
            f"{api_url}/predict",
            json=payload
        )
        elapsed = time.time() - start

        return {
            "status_code": response.status_code,
            "response": response.json() if response.ok else response.text,
            "latency_ms": elapsed * 1000
        }

    df = load_dataset(CSV_PATH)

    print(f"📄 Dataset chargé : {df.shape[0]} lignes / {df.shape[1]} features")

    #sys.exit ()

    results = []

    for idx, row in df.iterrows():
        result = send_row_to_api(row, API_URL)
        results.append(result)

        print(
            f"Ligne {idx} | "
            f"status={result['status_code']} | "
            f"{result['latency_ms']:.2f} ms"
        )

        if result["status_code"] == 200:
            logger.info(
                f"Ligne {idx} | OK | {result['latency_ms']:.2f} ms"
            )
        else:
            logger.warning(
                f"Ligne {idx} | KO | status={result['status_code']}"
            )


    ########

    
    latencies = [r["latency_ms"] for r in results if r["status_code"] == 200]

    print("\n📊 Résumé")
    print(f"Total requêtes : {len(results)}")
    print(f"Succès (200)   : {len(latencies)}")
    print(f"Latence moy.   : {np.mean(latencies):.2f} ms")
    print(f"Latence min    : {np.min(latencies):.2f} ms")
    print(f"Latence max    : {np.max(latencies):.2f} ms")
    print()
    
    print("=" * 60)
    print("✅ Tests terminés!")
    print("=" * 60)

    logger.info("📊 Résumé des tests")
    logger.info(f"Total requêtes : {len(results)}")
    logger.info(f"Succès (200)   : {len(latencies)}")

    if latencies:
        logger.info(f"Latence moy.   : {np.mean(latencies):.2f} ms")
        logger.info(f"Latence min    : {np.min(latencies):.2f} ms")
        logger.info(f"Latence max    : {np.max(latencies):.2f} ms")

    logger.info("✅ Tests terminés")



if __name__ == "__main__":
    main()

