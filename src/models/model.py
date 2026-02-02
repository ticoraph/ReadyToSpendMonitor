"""Model loading and inference logic."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
from .config import settings
from .logger import logger


class ScoringModel:
    """Scoring model wrapper for inference."""

    def __init__(self, model_path: Path = None):
        """
        Initialize the scoring model.

        Args:
            model_path: Path to the model file. If None, uses default from settings.
        """
        if model_path is None:
            model_path = settings.model_path / settings.model_name

        self.model_path = model_path
        self.model = None
        self.feature_names = None

    def load(self) -> None:
        """Load the model from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            # Try to get feature names from model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            elif hasattr(self.model, 'get_booster'):
                # For XGBoost
                self.feature_names = self.model.get_booster().feature_names

            logger.info(f"Model loaded successfully. Features: {self.feature_names}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a prediction on input data.

        Args:
            data: Input data as dictionary or DataFrame

        Returns:
            Dictionary with prediction and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data

        # Ensure correct feature order
        if self.feature_names:
            # Reorder and select features
            df = df[self.feature_names]

        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = None

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

        return {
            "prediction": int(prediction),
            "probability": probability,
        }


# Global model instance (singleton pattern)
_model_instance: ScoringModel = None


def get_model() -> ScoringModel:
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ScoringModel()
        _model_instance.load()
    return _model_instance