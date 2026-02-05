#!/usr/bin/env python3
"""
Script de test rapide de l'API
"""
import requests
import json
import time
#import random
import pandas as pd
#from datetime import datetime
import numpy as np

API_URL = "http://localhost:8000"
CSV_PATH = "output/dataset_test_top40_clean.csv"

def test_health():
    """Test du health check"""
    print("🔍 Test Health Check...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()

def test_prediction(client_data):
    """Test d'une prédiction"""
    print(f"🎯 Test Prédiction...")
    print(f"   Input: {json.dumps(client_data, indent=2)}")
    
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
    "EXT_SOURCE_3": 0.1595195404777181,
    "EXT_SOURCE_2": 0.7896543511176771,
    "PAYMENT_RATE": 0.03614715189873418,
    "DAYS_EMPLOYED": -2329.0,
    "DAYS_REGISTRATION": -5170.0,
    "EXT_SOURCE_1": 0.7526144906031748,
    "DAYS_BIRTH": -19241,
    "DAYS_ID_PUBLISH": -812,
    "DAYS_EMPLOYED_PERC": 0.1210436048022452,
    "AMT_ANNUITY": 20560.5,
    "REGION_POPULATION_RELATIVE": 0.01885,
    "INSTAL_DBD_MEAN": 8.857142857142858,
    "ANNUITY_INCOME_PERC": 0.1523,
    "INSTAL_DBD_SUM": 62.0,
    "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 411.0,
    "DAYS_LAST_PHONE_CHANGE": -1740.0,
    "INSTAL_AMT_PAYMENT_MIN": 3951.0,
    "INCOME_CREDIT_PERC": 0.23734177215189872,
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -1628.0,
    "BURO_DAYS_CREDIT_VAR": 240043.66666666672,
    "ACTIVE_DAYS_CREDIT_UPDATE_MEAN": -10.666666666666666,
    "APPROVED_DAYS_DECISION_MAX": -1740.0,
    "AMT_GOODS_PRICE": 450000.0,
    "CLOSED_DAYS_CREDIT_MAX": -857.0,
    "PREV_APP_CREDIT_PERC_VAR": 0,
    "PREV_APP_CREDIT_PERC_MEAN": 1.0440786984487325,
    "INSTAL_DBD_MAX": 36.0,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -15365.0,
    "BURO_AMT_CREDIT_SUM_MEAN": 207623.57142857142,
    "POS_MONTHS_BALANCE_MEAN": -72.55555555555556,
    "INCOME_PER_PERSON": 67500.0,
    "POS_NAME_CONTRACT_STATUS_Active_MEAN": 0.7777777777777778,
    "POS_NAME_CONTRACT_STATUS_Completed_MEAN": 0.2222222222222222,
    "PREV_HOUR_APPR_PROCESS_START_MEAN": 13.0,
    "ACTIVE_DAYS_CREDIT_MAX": -49.0,
    "ACTIVE_DAYS_CREDIT_MEAN": -309.3333333333333,
    "APPROVED_APP_CREDIT_PERC_VAR": 0,
    "AMT_CREDIT": 568800.0,
    "INSTAL_AMT_PAYMENT_MAX": 17397.9,
    "PREV_DAYS_DECISION_MAX": -1740.0
    })
    
    # Test 3: Prédiction avec profil risqué
    print("📊 Scénario 2: BON PROFIL")
    test_prediction({
    "EXT_SOURCE_3": 0.2636468134452008,
    "EXT_SOURCE_2": 0.6844067238529257,
    "PAYMENT_RATE": 0.07320777642770353,
    "DAYS_EMPLOYED": -1007.0,
    "DAYS_REGISTRATION": -5719.0,
    "EXT_SOURCE_1": 0.3441652580978948,
    "DAYS_BIRTH": -13563,
    "DAYS_ID_PUBLISH": -4044,
    "DAYS_EMPLOYED_PERC": 0.07424611074246111,
    "AMT_ANNUITY": 21690.0,
    "REGION_POPULATION_RELATIVE": 0.032561,
    "INSTAL_DBD_MEAN": 4.275,
    "ANNUITY_INCOME_PERC": 0.1205,
    "INSTAL_DBD_SUM": 171.0,
    "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 349.0,
    "DAYS_LAST_PHONE_CHANGE": -317.0,
    "INSTAL_AMT_PAYMENT_MIN": 36.99,
    "INCOME_CREDIT_PERC": 0.6075334143377886,
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -4.0,
    "BURO_DAYS_CREDIT_VAR": 0,
    "ACTIVE_DAYS_CREDIT_UPDATE_MEAN": -17.0,
    "APPROVED_DAYS_DECISION_MAX": -318.0,
    "AMT_GOODS_PRICE": 225000.0,
    "CLOSED_DAYS_CREDIT_MAX": 0,
    "PREV_APP_CREDIT_PERC_VAR": 0.009351982573951305,
    "PREV_APP_CREDIT_PERC_MEAN": 0.9316187797200455,
    "INSTAL_DBD_MAX": 20.0,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -6086.0,
    "BURO_AMT_CREDIT_SUM_MEAN": 124195.68,
    "POS_MONTHS_BALANCE_MEAN": -6.0,
    "INCOME_PER_PERSON": 90000.0,
    "POS_NAME_CONTRACT_STATUS_Active_MEAN": 0.9090909090909091,
    "POS_NAME_CONTRACT_STATUS_Completed_MEAN": 0.09090909090909091,
    "PREV_HOUR_APPR_PROCESS_START_MEAN": 19.5,
    "ACTIVE_DAYS_CREDIT_MAX": -17.0,
    "ACTIVE_DAYS_CREDIT_MEAN": -17.0,
    "APPROVED_APP_CREDIT_PERC_VAR": 0.009351982573951305,
    "AMT_CREDIT": 296280.0,
    "INSTAL_AMT_PAYMENT_MAX": 22500.0,
    "PREV_DAYS_DECISION_MAX": -318.0
    })
    
    #########

    def load_dataset(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])

        # Remplacer Inf/-Inf par NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Calculer la médiane de chaque colonne (ignorer les NaN)
        medians = df.median()

        # Remplacer NaN par la médiane
        df = df.fillna(medians)

        return df
    
    def send_row_to_api(row: pd.Series, api_url: str):
        payload = row.to_dict()

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

    results = []

    for idx, row in df.iterrows():
        result = send_row_to_api(row, API_URL)
        results.append(result)

        print(
            f"Ligne {idx} | "
            f"status={result['status_code']} | "
            f"{result['latency_ms']:.2f} ms"
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

if __name__ == "__main__":
    main()



'''

    #####################################

    def randomize_value(value, variation=0.1):
        """
        Génère une valeur aléatoire autour de la valeur d'origine
        en conservant le signe.
        
        variation = 0.1 -> ±10%
        """
        if value == 0:
            return 0

        delta = abs(value) * variation
        randomized = random.uniform(abs(value) - delta, abs(value) + delta)

        return -randomized if value < 0 else randomized
    
    BASE_PAYLOAD ={
        "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 350,
        "ACTIVE_DAYS_CREDIT_MAX": -17,
        "ACTIVE_DAYS_CREDIT_MEAN": -17,
        "ACTIVE_DAYS_CREDIT_UPDATE_MEAN": -17,
        "AMT_ANNUITY": 21000,
        "AMT_CREDIT": 296000,
        "AMT_GOODS_PRICE": 225000,
        "ANNUITY_INCOME_PERC": 0.12,
        "APPROVED_APP_CREDIT_PERC_VAR": 0.01,
        "APPROVED_DAYS_DECISION_MAX": -320,
        "BURO_AMT_CREDIT_SUM_MEAN": 12400,
        "BURO_DAYS_CREDIT_VAR": 0,
        "CLOSED_DAYS_CREDIT_MAX": 0,
        "DAYS_BIRTH": -13000,
        "DAYS_EMPLOYED": -1000,
        "DAYS_EMPLOYED_PERC": 0.07,
        "DAYS_ID_PUBLISH": -4000,
        "DAYS_LAST_PHONE_CHANGE": -300,
        "DAYS_REGISTRATION": -5000,
        "EXT_SOURCE_1": 0.34,
        "EXT_SOURCE_2": 0.68,
        "EXT_SOURCE_3": 0.26,
        "INCOME_CREDIT_PERC": 0.6,
        "INCOME_PER_PERSON": 90000,
        "INSTAL_AMT_PAYMENT_MAX": 22500,
        "INSTAL_AMT_PAYMENT_MIN": 37,
        "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -4,
        "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -6000,
        "INSTAL_DBD_MAX": 20,
        "INSTAL_DBD_MEAN": 4,
        "INSTAL_DBD_SUM": 170,
        "PAYMENT_RATE": 0.07,
        "POS_MONTHS_BALANCE_MEAN": -6,
        "POS_NAME_CONTRACT_STATUS_Active_MEAN": 0.9,
        "POS_NAME_CONTRACT_STATUS_Completed_MEAN": 0.09,
        "PREV_APP_CREDIT_PERC_MEAN": 0.93,
        "PREV_APP_CREDIT_PERC_VAR": 0.01,
        "PREV_DAYS_DECISION_MAX": -320,
        "PREV_HOUR_APPR_PROCESS_START_MEAN": 20,
        "REGION_POPULATION_RELATIVE": 0.03
    }
    
    def generate_random_payload(base_payload, variation=0.1):
        return {
            k: randomize_value(v, variation)
            for k, v in base_payload.items()
        }

    #####################################

    # Test 5: Test de charge (500 requêtes)
    print("🔥 Test de charge (500 requêtes)...")
    times = []
    for i in range(500):
        payload = generate_random_payload(BASE_PAYLOAD, variation=0.99)
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=payload
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Requête {i+1} | status={response.status_code} | {elapsed:.3f}s")

'''