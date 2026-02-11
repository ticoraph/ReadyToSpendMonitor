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

# ========== Configuration ==========
USE_BATCH_MODE = True  # ✅ Activer le mode batch pour meilleure performance
BATCH_SIZE = 100  # Taille de chaque batch
MAX_WORKERS = 5  # Moins de workers en mode batch (chaque batch = 100 requêtes)

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

# ==================== MODE BATCH ====================

def send_batch_to_api(batch_data: list, start_idx: int, api_url: str):
    """
    Envoie un batch de clients à la fois via /predict_batch
    """
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{api_url}/predict_batch",
            json=batch_data,
            timeout=30  # Timeout plus long pour batch
        )
        elapsed = time.time() - start_time

        if response.ok:
            predictions = response.json()
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    "index": start_idx + i,
                    "status_code": 200,
                    "response": pred,
                    "latency_ms": elapsed * 1000 / len(predictions)  # Répartir le temps
                })
            return results
        else:
            # Erreur batch entier
            error_results = []
            for i in range(len(batch_data)):
                error_results.append({
                    "index": start_idx + i,
                    "status_code": response.status_code,
                    "response": response.text,
                    "latency_ms": elapsed * 1000
                })
            return error_results

    except requests.exceptions.Timeout:
        error_results = []
        for i in range(len(batch_data)):
            error_results.append({
                "index": start_idx + i,
                "status_code": 0,
                "response": "Timeout - API non réactive",
                "latency_ms": 30000
            })
        return error_results
    except Exception as e:
        error_results = []
        for i in range(len(batch_data)):
            error_results.append({
                "index": start_idx + i,
                "status_code": 0,
                "response": str(e),
                "latency_ms": 0
            })
        return error_results

def send_all_rows_batch(df: pd.DataFrame, api_url: str, batch_size: int = 100, max_workers: int = 5):
    """
    Envoie les données par batch via /predict_batch avec ThreadPoolExecutor
    Beaucoup plus efficace pour grandes volumes
    """
    results = []
    
    # Créer les batches
    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_data = [row.to_dict() for _, row in batch_df.iterrows()]
        batches.append((i, batch_data))
    
    print(f"📦 {len(batches)} batches créés (taille: {batch_size})\n")
    
    # Créer le ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre toutes les tâches batch
        futures = {
            executor.submit(send_batch_to_api, batch_data, start_idx, api_url): start_idx
            for start_idx, batch_data in batches
        }
        
        # Traiter les résultats à mesure qu'ils arrivent
        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
            completed += 1
            
            successful_in_batch = sum(1 for r in batch_results if r["status_code"] == 200)
            
            print(
                f"Batch {completed:4d}/{len(batches):4d} | "
                f"Succès: {successful_in_batch:3d}/{len(batch_results):3d} | "
                f"[{len(results):6d} requêtes traitées]"
            )
    
    return results

# ==================== MODE SINGLE ====================

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

    # Lancer les prédictions
    print(f"\n🚀 Lancement des requêtes API...\n")
    
    start_global = time.time()
    
    if USE_BATCH_MODE:
        print("📦 Mode BATCH activé (recommandé pour gros volumes)\n")
        results = send_all_rows_batch(df, API_URL, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
    else:
        print("⚡ Mode SINGLE activé\n")
        results = send_all_rows_parallel(df, API_URL, max_workers=10)
    
    total_time = time.time() - start_global
    
    # Afficher le résumé
    print_summary(results, df)
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/api_predictions_results.csv", index=False)
    print(f"✅ Résultats sauvegardés dans 'api_predictions_results.csv'")