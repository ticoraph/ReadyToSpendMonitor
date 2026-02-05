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
  "DAYS_BIRTH": -5400,  # trop jeune
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
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_negative_income():
    """
    Test avec un revenu négatif
    """
    payload = {
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
  "DAYS_BIRTH": -10000,
  "DAYS_EMPLOYED": -1000,
  "DAYS_EMPLOYED_PERC": 0.07,
  "DAYS_ID_PUBLISH": -4000,
  "DAYS_LAST_PHONE_CHANGE": -300,
  "DAYS_REGISTRATION": -5000,
  "EXT_SOURCE_1": 0.34,
  "EXT_SOURCE_2": 0.68,
  "EXT_SOURCE_3": 0.26,
  "INCOME_CREDIT_PERC": 0.6,
  "INCOME_PER_PERSON": -90000,  # revenu négatif
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
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_missing_field():
    """
    Test avec un champ manquant
    """
    payload = {
  "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 350,
  "ACTIVE_DAYS_CREDIT_MAX": -17,
  #"ACTIVE_DAYS_CREDIT_MEAN": -17,
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
  "DAYS_BIRTH": -10000,
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
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_multiple_requests():
    """
    Test de plusieurs prédictions successives (charge)
    """
    payload = {
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
  "DAYS_BIRTH": -10000,
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
    
    for _ in range(10):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


def test_api_response_time():
    """
    Test du temps de réponse de l'API
    """
    import time
    
    payload = {
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
    
    start = time.time()
    response = client.post("/predict", json=payload)
    duration = time.time() - start
    
    assert response.status_code == 200
    # L'API devrait répondre en moins d'une seconde
    assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
