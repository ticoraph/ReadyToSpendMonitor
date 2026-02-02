"""Module de chargement et d'utilisation du modèle de scoring."""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ScoringModel:
    """Classe pour gérer le modèle de scoring de crédit."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialise le modèle de scoring.

        Args:
            model_path: Chemin vers le fichier du modèle. Si None, cherche dans models/.
        """
        self.model = None
        self.model_path = model_path or Path("models") / "scoring_model.pkl"
        self._load_model()

    def _load_model(self) -> None:
        """Charge le modèle depuis le disque."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Modèle chargé avec succès depuis {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Modèle non trouvé à {self.model_path}, création d'un modèle dummy")
            self.model = self._create_dummy_model()
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

    def _create_dummy_model(self):
        """Crée un modèle dummy pour le développement."""
        from sklearn.ensemble import RandomForestClassifier

        # Modèle simple pour développement
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Ajuster avec des données dummy
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        logger.info("Modèle dummy créé pour le développement")
        return model

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fait une prédiction sur les données fournies.

        Args:
            features: Dictionnaire des features du client.

        Returns:
            Dictionnaire contenant le score de prédiction et sa probabilité.
        """
        try:
            # Conversion en tableau numpy
            feature_values = self._dict_to_array(features)

            # Prédiction
            score = int(self.model.predict(feature_values)[0])
            probability = float(self.model.predict_proba(feature_values)[0, 1])

            logger.info(f"Prédiction réussie: score={score}, probabilité={probability:.4f}")

            return {
                "score": score,
                "probability": probability,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return {
                "score": None,
                "probability": None,
                "status": "error",
                "error": str(e)
            }

    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convertit un dictionnaire en tableau numpy."""
        # Pour l'instant, conversion simple
        # À adapter selon les features réelles du modèle
        values = list(features.values())
        return np.array([values]).reshape(1, -1)

    def get_feature_names(self) -> list:
        """Retourne la liste des features attendues par le modèle."""
        # À adapter selon le modèle réel
        return [
            "age", "income", "employment_length", "debt_ratio",
            "credit_history", "num_accounts", "num_late_payments",
            "home_ownership", "loan_amount", "loan_term"
        ]


# Instance globale du modèle (chargée une seule fois)
_model_instance: Optional[ScoringModel] = None


def get_model() -> ScoringModel:
    """Retourne l'instance singleton du modèle."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ScoringModel()
    return _model_instance