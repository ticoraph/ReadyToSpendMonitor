"""Tests pour l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Fixture pour le client de test."""
    return TestClient(app)


@pytest.fixture
def sample_client_data():
    """Données de test valides."""
    return {
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


class TestAPIEndpoints:
    """Tests pour les endpoints de l'API."""

    def test_root_endpoint(self, client):
        """Test l'endpoint racine."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"

    def test_health_check(self, client):
        """Test l'endpoint de santé."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_info(self, client):
        """Test l'endpoint d'info modèle."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "feature_names" in data
        assert isinstance(data["feature_names"], list)

    def test_predict_valid_input(self, client, sample_client_data):
        """Test l'endpoint predict avec des données valides."""
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["score"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert "inference_time" in data
        assert "timestamp" in data


class TestAPIValidation:
    """Tests pour la validation des entrées."""

    def test_predict_invalid_age_too_young(self, client, sample_client_data):
        """Test avec un âge invalide (trop jeune)."""
        sample_client_data["age"] = 17
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_age_too_old(self, client, sample_client_data):
        """Test avec un âge invalide (trop vieux)."""
        sample_client_data["age"] = 101
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422

    def test_predict_negative_income(self, client, sample_client_data):
        """Test avec un revenu négatif."""
        sample_client_data["income"] = -1000.0
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422

    def test_predict_zero_income(self, client, sample_client_data):
        """Test avec un revenu nul."""
        sample_client_data["income"] = 0.0
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422

    def test_predict_missing_field(self, client, sample_client_data):
        """Test avec un champ manquant."""
        del sample_client_data["age"]
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422

    def test_predict_wrong_data_type(self, client, sample_client_data):
        """Test avec un type de données incorrect."""
        sample_client_data["age"] = "trente-cinq"
        response = client.post("/predict", json=sample_client_data)
        assert response.status_code == 422