"""Model loader and inference wrapper."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from src.app.core.config import settings


class ModelWrapper:
    def __init__(self, model_path: str | None = None, pipeline_path: str | None = None, threshold: float | None = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline_path = pipeline_path or settings.PIPELINE_PATH
        self.threshold = threshold or settings.THRESHOLD
        self.model = None
        self.pipeline = None
        self.version = settings.MODEL_VERSION

    def load(self):
        # Load pipeline and model (raise helpful errors if missing)
        try:
            self.pipeline = joblib.load(self.pipeline_path)
        except Exception as e:
            raise RuntimeError(f"Unable to load preprocessing pipeline from {self.pipeline_path}: {e}")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Unable to load model from {self.model_path}: {e}")
        return self

    def predict(self, tx: dict) -> dict:
        # tx: raw transaction dict
        if self.pipeline is None or self.model is None:
            raise RuntimeError("Model or pipeline not loaded. Call load() first.")
        # Accept pandas-friendly dicts
        df = pd.DataFrame([tx])
        X = self.pipeline.transform(df)
        # model must support predict_proba
        if not hasattr(self.model, "predict_proba"):
            # if it's anomaly model, we could use decision_function
            raise RuntimeError("Model does not support predict_proba")
        proba = float(self.model.predict_proba(X)[:, 1][0])
        label = int(proba >= self.threshold)
        return {
            "fraud_probability": proba,
            "prediction": "fraud" if label == 1 else "legit",
            "model_version": self.version,
            "threshold": self.threshold,
        }


# singleton accessor for app
_model_wrapper: ModelWrapper | None = None


def get_model() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
        _model_wrapper.load()
    return _model_wrapper
