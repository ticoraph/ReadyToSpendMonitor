import pandas as pd
import numpy as np
import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
CSV_PATH = "output/dataset_test_top40.csv"

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

def send_row_to_api(idx_row_tuple: tuple, api_url: str):
    """
    Fonction pour traiter une ligne (utilisée par ThreadPoolExecutor)
    Prend un tuple (index, row) et retourne le résultat
    """
    idx, row = idx_row_tuple
    payload = row.to_dict()

    try:
        start = time.time()
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10  # Timeout de 10 secondes
        )
        elapsed = time.time() - start

        result = {
            "index": idx,
            "status_code": response.status_code,
            "response": response.json() if response.ok else response.text,
            "latency_ms": elapsed * 1000
        }
    except requests.exceptions.Timeout:
        result = {
            "index": idx,
            "status_code": 0,
            "response": "Timeout - API non réactive",
            "latency_ms": 10000
        }
    except Exception as e:
        result = {
            "index": idx,
            "status_code": 0,
            "response": str(e),
            "latency_ms": 0
        }

    return result

def send_all_rows_parallel(df: pd.DataFrame, api_url: str, max_workers: int = 10):
    """
    Envoie toutes les lignes en parallèle avec ThreadPoolExecutor
    """
    results = []
    
    # Créer le ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre toutes les tâches
        futures = {
            executor.submit(send_row_to_api, (idx, row), api_url): idx
            for idx, row in df.iterrows()
        }
        
        # Traiter les résultats à mesure qu'ils arrivent
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            idx = result["index"]
            status = result["status_code"]
            latency = result["latency_ms"]
            
            print(
                f"Ligne {idx:4d} | "
                f"status={status:3d} | "
                f"{latency:8.2f} ms | "
                f"[{completed:4d}/{len(df):4d}]"
            )
            
            if status == 200:
                logger.info(
                    f"Ligne {idx} | OK | {latency:.2f} ms"
                )
            else:
                logger.warning(
                    f"Ligne {idx} | KO | status={status}"
                )
    
    return results

def print_summary(results: list, df: pd.DataFrame):
    """
    Affiche un résumé des résultats
    """
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    
    total = len(results)
    successful = sum(1 for r in results if r["status_code"] == 200)
    failed = total - successful
    
    latencies = [r["latency_ms"] for r in results if r["status_code"] == 200]
    
    print(f"Total requêtes:     {total}")
    print(f"Succès:             {successful} ({successful/total*100:.1f}%)")
    print(f"Erreurs:            {failed} ({failed/total*100:.1f}%)")
    
    if latencies:
        print(f"\nLatence (succès uniquement):")
        print(f"  Min:              {min(latencies):.2f} ms")
        print(f"  Max:              {max(latencies):.2f} ms")
        print(f"  Moyenne:          {np.mean(latencies):.2f} ms")
        print(f"  Médiane:          {np.median(latencies):.2f} ms")
        print(f"  P95:              {np.percentile(latencies, 95):.2f} ms")
    
    total_time = sum(r["latency_ms"] for r in results)
    print(f"\nTemps total:        {total_time/1000:.2f} s")
    print(f"Throughput:         {total / (total_time/1000):.2f} req/s")
    print("="*60 + "\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    
    # Charger le dataset
    df = load_dataset(CSV_PATH)
    print(f"📄 Dataset chargé : {df.shape[0]} lignes / {df.shape[1]} features")

    # Lancer les prédictions en parallèle (10 workers)
    print(f"\n🚀 Lancement des requêtes API en parallèle (10 workers)...\n")
    
    start_global = time.time()
    results = send_all_rows_parallel(df, API_URL, max_workers=10)
    total_time = time.time() - start_global
    
    # Afficher le résumé
    print_summary(results, df)
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    results_df.to_csv("api_predictions_results.csv", index=False)
    print(f"✅ Résultats sauvegardés dans 'api_predictions_results.csv'")