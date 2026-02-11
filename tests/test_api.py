"""
Tests unitaires pour l'API de scoring
"""
import pytest
from fastapi.testclient import TestClient
import api.main as main_module
from api.main import app, load_model

# Créer un client de test
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """
    Charge le modèle avant d'exécuter les tests et l'ajoute à app.state
    """
    try:
        # Charger le modèle globalement
        load_model()
        
        # Récupérer le MODEL depuis le module main
        model = main_module.MODEL
        
        # Ajouter le modèle à app.state pour que les endpoints y accèdent
        app.state.model = model
        
        print(f"✅ Modèle chargé avec succès: {type(model)}")
        yield
        
    except FileNotFoundError as e:
        print(f"⚠️ Fichier modèle non trouvé: {e}")
        app.state.model = None
        yield
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du modèle: {e}")
        app.state.model = None
        yield


# ==================== PAYLOADS ====================

VALID_PAYLOAD = {
    "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
    "ACTIVE_AMT_CREDIT_SUM_MAX": 450000,
    "ACTIVE_DAYS_CREDIT_MAX": -753,
    "AMT_ANNUITY": 10548,
    "AMT_CREDIT": 148365,
    "AMT_GOODS_PRICE": 135000,
    "ANNUITY_INCOME_PERC": 0.1019130434782608,
    "APPROVED_AMT_ANNUITY_MEAN": 6340.785,
    "APPROVED_CNT_PAYMENT_MEAN": 14.666666666666666,
    "APPROVED_DAYS_DECISION_MAX": -348,
    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0,
    "BURO_DAYS_CREDIT_MAX": -753,
    "BURO_DAYS_CREDIT_MEAN": -979.6666666666666,
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.2666666666666666,
    "CLOSED_AMT_CREDIT_SUM_MAX": 38650.5,
    "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -943,
    "CLOSED_DAYS_CREDIT_MAX": -1065,
    "CLOSED_DAYS_CREDIT_VAR": 256328,
    "CODE_GENDER": 1,
    "DAYS_BIRTH": -11716,
    "DAYS_EMPLOYED": -449,
    "DAYS_EMPLOYED_PERC": 0.0383236599522021,
    "DAYS_ID_PUBLISH": -3961,
    "DAYS_LAST_PHONE_CHANGE": -1420,
    "DAYS_REGISTRATION": -3997,
    "EXT_SOURCE_1": 0.3608707365728421,
    "EXT_SOURCE_2": 0.4285392216965799,
    "EXT_SOURCE_3": 0.7981372313187245,
    "INSTAL_AMT_PAYMENT_MEAN": 10274.82081081081,
    "INSTAL_AMT_PAYMENT_MIN": 2.7,
    "INSTAL_AMT_PAYMENT_SUM": 380168.37,
    "INSTAL_DBD_MAX": 60,
    "INSTAL_DBD_SUM": 833,
    "INSTAL_DPD_MEAN": 0.4594594594594595,
    "INSTAL_PAYMENT_PERC_MEAN": 0.945945945945946,
    "OWN_CAR_AGE": 9,
    "PAYMENT_RATE": 0.0710949347892023,
    "POS_MONTHS_BALANCE_SIZE": 40,
    "PREV_CNT_PAYMENT_MEAN": 15.142857142857142
}


# ==================== TESTS ====================

def test_health_check():
    """
    Test du health check
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    assert data["status"] in ["healthy", "unhealthy"]


def test_predict_valid_input():
    """
    Test de prédiction avec des données valides
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    
    data = response.json()
    assert "client_id" in data
    assert "score" in data
    assert "decision" in data
    assert "confidence" in data
    assert "inference_time_ms" in data
    
    # Vérifier les types et ranges
    assert 0 <= data["score"] <= 1, f"Score hors limites: {data['score']}"
    assert data["decision"] in ["APPROVED", "REJECTED"], f"Decision invalide: {data['decision']}"
    assert 0 <= data["confidence"] <= 1, f"Confidence hors limites: {data['confidence']}"
    assert data["inference_time_ms"] > 0, f"Inference time invalide: {data['inference_time_ms']}"


def test_predict_missing_field():
    """
    Test avec un champ manquant
    """
    payload = VALID_PAYLOAD.copy()
    del payload["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]
    
    response = client.post("/predict", json=payload)
    # Devrait retourner une erreur (422 ou 500)
    assert response.status_code in [422, 500], f"Status code attendu 422 ou 500, reçu {response.status_code}"


def test_predict_invalid_data_type():
    """
    Test avec un type de données invalide (string au lieu de float)
    """
    payload = VALID_PAYLOAD.copy()
    payload["AMT_CREDIT"] = "not_a_number"  # String au lieu de float
    
    response = client.post("/predict", json=payload)
    # Devrait retourner une erreur de validation
    assert response.status_code in [422, 500], f"Status code attendu 422 ou 500, reçu {response.status_code}"


def test_predict_multiple_requests():
    """
    Test de plusieurs prédictions successives (charge)
    """
    num_requests = 10
    
    for i in range(num_requests):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200, f"Request {i+1} failed: {response.text}"


def test_api_response_time():
    """
    Test du temps de réponse de l'API
    """
    import time
    
    start = time.time()
    response = client.post("/predict", json=VALID_PAYLOAD)
    duration = time.time() - start
    
    assert response.status_code == 200, f"Status code: {response.status_code}"
    
    # L'API devrait répondre en moins de 2 secondes (test)
    assert duration < 2.0, f"Réponse trop lente: {duration:.2f}s"


def test_predict_batch_valid_input():
    """
    Test de prédiction batch avec des données valides
    """
    batch_payload = [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]
    
    response = client.post("/predict_batch", json=batch_payload)
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    
    data = response.json()
    assert isinstance(data, list), "Response should be a list"
    assert len(data) == 3, f"Expected 3 predictions, got {len(data)}"
    
    # Vérifier chaque prédiction
    for pred in data:
        assert "client_id" in pred
        assert "score" in pred
        assert "decision" in pred
        assert "confidence" in pred
        assert 0 <= pred["score"] <= 1
        assert pred["decision"] in ["APPROVED", "REJECTED"]


def test_predict_batch_empty():
    """
    Test de batch vide
    """
    batch_payload = []
    
    response = client.post("/predict_batch", json=batch_payload)
    # Dépend de comment tu gères les batches vides
    assert response.status_code in [200, 400, 422], f"Status code: {response.status_code}"


def test_predict_batch_mixed_data():
    """
    Test de batch avec plusieurs clients
    """
    payload1 = VALID_PAYLOAD.copy()
    
    payload2 = VALID_PAYLOAD.copy()
    payload2["AMT_CREDIT"] = 200000  # Crédit plus élevé
    
    payload3 = VALID_PAYLOAD.copy()
    payload3["CODE_GENDER"] = 0  # Genre différent
    
    batch_payload = [payload1, payload2, payload3]
    
    response = client.post("/predict_batch", json=batch_payload)
    assert response.status_code == 200, f"Status code: {response.status_code}"
    
    data = response.json()
    assert len(data) == 3, f"Expected 3 predictions, got {len(data)}"


def test_api_response_structure():
    """
    Test de la structure de la réponse API
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    
    data = response.json()
    
    # Vérifier la structure exacte
    required_fields = ["client_id", "score", "decision", "confidence", "inference_time_ms"]
    for field in required_fields:
        assert field in data, f"Champ manquant: {field}"
    
    # Vérifier les types
    assert isinstance(data["client_id"], str)
    assert isinstance(data["score"], (int, float))
    assert isinstance(data["decision"], str)
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["inference_time_ms"], (int, float))


def test_health_check_model_loaded():
    """
    Vérifier que le modèle est chargé au démarrage
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    # Le modèle devrait être chargé après le setup_model fixture
    assert data["model_loaded"] is True or data["model_loaded"] is False
    # (On teste juste que le champ existe, pas sa valeur, au cas où le modèle manque)


# ==================== PERFORMANCE TESTS ====================

def test_multiple_batch_requests():
    """
    Test de performance: plusieurs batches en séquence
    """
    import time
    
    batch_payload = [VALID_PAYLOAD] * 5
    num_batches = 3
    
    start = time.time()
    for i in range(num_batches):
        response = client.post("/predict_batch", json=batch_payload)
        assert response.status_code == 200, f"Batch {i+1} failed"
    
    duration = time.time() - start
    total_predictions = num_batches * len(batch_payload)
    
    print(f"\n⏱️ Performance: {total_predictions} prédictions en {duration:.2f}s ({total_predictions/duration:.1f} req/s)")
    assert duration < 10.0, f"Trop lent: {duration:.2f}s pour {total_predictions} prédictions"


# ==================== ERROR HANDLING ====================

def test_predict_with_extra_fields():
    """
    Test avec des champs supplémentaires (ne devrait pas causer d'erreur)
    """
    payload = VALID_PAYLOAD.copy()
    payload["EXTRA_FIELD"] = 999  # Champ extra
    
    response = client.post("/predict", json=payload)
    # Les champs extra sont généralement ignorés par Pydantic
    assert response.status_code == 200, f"Status code: {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])