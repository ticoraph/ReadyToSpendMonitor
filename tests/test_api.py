import pytest
from fastapi.testclient import TestClient
from api.main import app
from api import main
from api.schemas import ClientData

# ==========================================================
# 🔧 PAYLOAD VALIDE (repris de ton example)
# ==========================================================

VALID_PAYLOAD = {
    "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
    "ACTIVE_AMT_CREDIT_SUM_MAX": 450000.0,
    "ACTIVE_DAYS_CREDIT_MAX": -753.0,
    "AMT_ANNUITY": 10548.0,
    "AMT_CREDIT": 148365.0,
    "AMT_GOODS_PRICE": 135000.0,
    "ANNUITY_INCOME_PERC": 0.1,
    "APPROVED_AMT_ANNUITY_MEAN": 6340.78,
    "APPROVED_CNT_PAYMENT_MEAN": 14.66,
    "APPROVED_DAYS_DECISION_MAX": -348.0,
    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 7195.5,
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0.0,
    "BURO_DAYS_CREDIT_MAX": -753.0,
    "BURO_DAYS_CREDIT_MEAN": -979.66,
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.26,
    "CLOSED_AMT_CREDIT_SUM_MAX": 38650.5,
    "CLOSED_DAYS_CREDIT_ENDDATE_MAX": -943.0,
    "CLOSED_DAYS_CREDIT_MAX": -1065.0,
    "CLOSED_DAYS_CREDIT_VAR": 256328.0,
    "CODE_GENDER": 1.0,
    "DAYS_BIRTH": -11716.0,
    "DAYS_EMPLOYED": -449.0,
    "DAYS_EMPLOYED_PERC": 0.03,
    "DAYS_ID_PUBLISH": -3961.0,
    "DAYS_LAST_PHONE_CHANGE": -1420.0,
    "DAYS_REGISTRATION": -3997.0,
    "EXT_SOURCE_1": 0.36,
    "EXT_SOURCE_2": 0.42,
    "EXT_SOURCE_3": 0.79,
    "INSTAL_AMT_PAYMENT_MEAN": 10274.82,
    "INSTAL_AMT_PAYMENT_MIN": 2.7,
    "INSTAL_AMT_PAYMENT_SUM": 380168.37,
    "INSTAL_DBD_MAX": 60.0,
    "INSTAL_DBD_SUM": 833.0,
    "INSTAL_DPD_MEAN": 0.45,
    "INSTAL_PAYMENT_PERC_MEAN": 0.94,
    "OWN_CAR_AGE": 9.0,
    "PAYMENT_RATE": 0.07,
    "POS_MONTHS_BALANCE_SIZE": 40.0,
    "PREV_CNT_PAYMENT_MEAN": 15.14
}


# ==========================================================
# 🔧 FIXTURE CLIENT
# ==========================================================

@pytest.fixture
def client():
    return TestClient(app)


# ==========================================================
# 🎯 FAKE MODELS
# ==========================================================

class ProbaModel:
    def predict_proba(self, X):
        return [[0.1, 0.9]]


class PredictOnlyModel:
    def predict(self, X):
        return [0.1]


class BrokenModel:
    def predict(self, X):
        raise Exception("boom")


# ==========================================================
# 🔥 HEALTH
# ==========================================================

def test_health_healthy(client):
    main.MODEL = object()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_unhealthy(client):
    main.MODEL = None
    response = client.get("/health")
    assert response.json()["status"] == "unhealthy"


# ==========================================================
# 🔥 PREDICT
# ==========================================================

def test_predict_proba(client):
    main.MODEL = ProbaModel()
    app.state.model = main.MODEL
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["decision"] == "APPROVED"


def test_predict_fallback_predict(client):
    main.MODEL = PredictOnlyModel()
    app.state.model = main.MODEL
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["decision"] == "REJECTED"


def test_predict_model_error(client):
    main.MODEL = BrokenModel()
    app.state.model = main.MODEL
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500


# ==========================================================
# 🔥 BATCH
# ==========================================================

def test_batch_success(client):
    main.MODEL = ProbaModel()
    app.state.model = main.MODEL
    response = client.post("/predict_batch", json=[VALID_PAYLOAD])
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_batch_model_none(client):
    main.MODEL = None
    app.state.model = None
    response = client.post("/predict_batch", json=[VALID_PAYLOAD])
    assert response.status_code == 503


def test_batch_internal_error(client, monkeypatch):
    main.MODEL = ProbaModel()
    app.state.model = main.MODEL

    def broken(*args, **kwargs):
        raise Exception("boom")

    monkeypatch.setattr(main, "_process_single_prediction", broken)
    response = client.post("/predict_batch", json=[VALID_PAYLOAD])
    assert response.status_code == 500


# ==========================================================
# 🔥 LOAD MODEL
# ==========================================================

def test_load_model_file_not_found(monkeypatch):
    monkeypatch.setattr(main.os.path, "exists", lambda x: False)
    with pytest.raises(FileNotFoundError):
        main.load_model()


def test_load_model_dict(monkeypatch):
    monkeypatch.setattr(main.os.path, "exists", lambda x: True)
    monkeypatch.setattr(main.joblib, "load", lambda x: {
        "model": object(),
        "features": ["f1", "f2"]
    })
    main.load_model()
    assert main.MODEL is not None


def test_load_model_lgbm(monkeypatch):
    class FakeModel:
        feature_name_ = ["f1", "f2"]

    monkeypatch.setattr(main.os.path, "exists", lambda x: True)
    monkeypatch.setattr(main.joblib, "load", lambda x: FakeModel())
    main.load_model()


def test_load_model_runtime_error(monkeypatch):
    class BadModel:
        pass

    monkeypatch.setattr(main.os.path, "exists", lambda x: True)
    monkeypatch.setattr(main.joblib, "load", lambda x: BadModel())

    with pytest.raises(RuntimeError):
        main.load_model()


# ==========================================================
# 🔥 PROFILING BRANCH
# ==========================================================

def test_profile_enabled(client):
    main.MODEL = ProbaModel()
    app.state.model = main.MODEL
    main.ENABLE_PROFILING = True

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    main.ENABLE_PROFILING = False