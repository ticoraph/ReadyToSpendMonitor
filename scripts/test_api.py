#!/usr/bin/env python3
"""
Script de test rapide de l'API
"""
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8000"

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
    print("📊 Scénario 1: Bon profil (devrait être approuvé)")
    test_prediction({
        "age": 35,
        "income": 60000,
        "loan_amount": 15000,
        "employment_length": 8,
        "credit_score": 750
    })
    
    # Test 3: Prédiction avec profil risqué
    print("📊 Scénario 2: Profil risqué (peut être rejeté)")
    test_prediction({
        "age": 22,
        "income": 25000,
        "loan_amount": 40000,
        "employment_length": 1,
        "credit_score": 580
    })
    
    # Test 4: Prédiction avec profil moyen
    print("📊 Scénario 3: Profil moyen")
    test_prediction({
        "age": 45,
        "income": 50000,
        "loan_amount": 25000,
        "employment_length": 15,
        "credit_score": 680
    })
    
    # Test 5: Test de charge (10 requêtes)
    print("🔥 Test de charge (10 requêtes)...")
    times = []
    for i in range(10):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json={
            "age": 30 + i,
            "income": 50000,
            "loan_amount": 20000,
            "employment_length": 5,
            "credit_score": 700
        })
        duration = (time.time() - start) * 1000
        times.append(duration)
        if i == 0:
            print(f"   Requête {i+1}: {duration:.2f}ms")
    
    print(f"   Moyenne: {sum(times)/len(times):.2f}ms")
    print(f"   Min: {min(times):.2f}ms")
    print(f"   Max: {max(times):.2f}ms")
    print()
    
    print("=" * 60)
    print("✅ Tests terminés!")
    print("=" * 60)

if __name__ == "__main__":
    main()
