"""Tests unitaires pour le modèle de scoring."""

import pytest
import numpy as np
from src.models.model import ScoringModel


class TestScoringModel:
    """Tests pour la classe ScoringModel."""

    @pytest.fixture
    def model(self):
        """Fixture pour créer une instance du modèle."""
        return ScoringModel()

    @pytest.fixture
    def sample_features(self):
        """Fixture pour des features de test."""
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

    def test_model_initialization(self, model):
        """Test que le modèle s'initialise correctement."""
        assert model is not None
        assert model.model is not None

    def test_predict_returns_valid_score(self, model, sample_features):
        """Test que la prédiction retourne un score valide (0 ou 1)."""
        result = model.predict(sample_features)
        assert result["status"] == "success"
        assert result["score"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_with_missing_features(self, model):
        """Test que le modèle gère les features manquantes."""
        incomplete_features = {"age": 30, "income": 40000.0}
        result = model.predict(incomplete_features)
        # Le modèle devrait retourner une erreur ou gérer le cas
        assert "status" in result

    def test_get_feature_names(self, model):
        """Test que la liste des features est correctement retournée."""
        features = model.get_feature_names()
        assert isinstance(features, list)
        assert len(features) > 0
        assert "age" in features
        assert "income" in features

    def test_model_singleton(self):
        """Test que le singleton fonctionne correctement."""
        from src.models.model import get_model
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2