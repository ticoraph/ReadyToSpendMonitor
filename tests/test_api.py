"""
Tests unitaires pour l'API de scoring
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app, load_model

# Créer un client de test
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """
    Charge le modèle avant d'exécuter les tests
    """
    load_model()


def test_read_root():
    """
    Test du endpoint racine
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_check():
    """
    Test du health check
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict_valid_input():
    """
    Test de prédiction avec des données valides
    """
    payload = {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "client_id" in data
    assert "score" in data
    assert "decision" in data
    assert "confidence" in data
    assert "inference_time_ms" in data
    
    # Vérifier les types et ranges
    assert 0 <= data["score"] <= 1
    assert data["decision"] in ["APPROVED", "REJECTED"]
    assert 0 <= data["confidence"] <= 1
    assert data["inference_time_ms"] > 0


def test_predict_invalid_age():
    """
    Test avec un âge invalide (trop jeune)
    """
    payload = {
        "age": 15,  # Mineur
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_negative_income():
    """
    Test avec un revenu négatif
    """
    payload = {
        "age": 35,
        "income": -10000,  # Négatif
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_missing_field():
    """
    Test avec un champ manquant
    """
    payload = {
        "age": 35,
        "income": 50000,
        # loan_amount manquant
        "employment_length": 5,
        "credit_score": 720
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_credit_score():
    """
    Test avec un score de crédit hors limites
    """
    payload = {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 1000  # Trop élevé (max 850)
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_multiple_requests():
    """
    Test de plusieurs prédictions successives (charge)
    """
    payload = {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
    
    for _ in range(10):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


def test_predict_edge_cases():
    """
    Test des cas limites
    """
    # Âge minimum
    payload_min = {
        "age": 18,
        "income": 1,
        "loan_amount": 1,
        "employment_length": 0,
        "credit_score": 300
    }
    response = client.post("/predict", json=payload_min)
    assert response.status_code == 200
    
    # Âge maximum
    payload_max = {
        "age": 100,
        "income": 1000000,
        "loan_amount": 500000,
        "employment_length": 50,
        "credit_score": 850
    }
    response = client.post("/predict", json=payload_max)
    assert response.status_code == 200


def test_api_response_time():
    """
    Test du temps de réponse de l'API
    """
    import time
    
    payload = {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "employment_length": 5,
        "credit_score": 720
    }
    
    start = time.time()
    response = client.post("/predict", json=payload)
    duration = time.time() - start
    
    assert response.status_code == 200
    # L'API devrait répondre en moins d'une seconde
    assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
