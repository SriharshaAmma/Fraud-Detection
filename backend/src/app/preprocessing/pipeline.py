"""Preprocessing pipeline for PaySim.

Produces a sklearn-compatible pipeline wrapper that:
- creates derived features
- encodes categorical `type`
- scales numeric features
- exposes fit / transform / save / load
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Base features from dataset
BASE_NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CATEGORICAL = ["type"]

# Derived numeric features we will compute and include
DERIVED_NUMERIC = ["delta_orig", "delta_dest", "amount_to_oldbalance_ratio", "hour_of_day"]

NUMERIC = BASE_NUMERIC + DERIVED_NUMERIC


class PreprocessingPipeline:
    def __init__(self):
        # numeric pipeline: scale numeric features
        numeric_pipeline = Pipeline([("scaler", StandardScaler())])
        # categorical pipeline: one-hot encode `type`
        categorical_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        # ColumnTransformer expects the final columns to exist in df before transform
        self.transformer = ColumnTransformer(
            [("num", numeric_pipeline, NUMERIC), ("cat", categorical_pipeline, CATEGORICAL)],
            remainder="drop",
        )
        self.feature_names: list[str] | None = None

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure required raw columns exist (fill missing with 0 / plausible defaults)
        for col in BASE_NUMERIC + CATEGORICAL:
            if col not in df.columns:
                if col in BASE_NUMERIC:
                    df[col] = 0.0
                else:
                    df[col] = "UNKNOWN"
        return df

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        # Derived features
        # delta_orig: how much money left on sender account after transaction (old - new)
        df["delta_orig"] = df["oldbalanceOrg"].fillna(0) - df["newbalanceOrig"].fillna(0)
        # delta_dest: how much money changed on dest (new - old)
        df["delta_dest"] = df["newbalanceDest"].fillna(0) - df["oldbalanceDest"].fillna(0)
        # ratio amount / oldbalanceOrg (guard tiny denom)
        df["amount_to_oldbalance_ratio"] = df["amount"].fillna(0) / (df["oldbalanceOrg"].fillna(0) + 1e-6)
        # hour_of_day (derived from step if present)
        if "step" in df.columns:
            df["hour_of_day"] = df["step"].astype(float).fillna(0) % 24
        else:
            df["hour_of_day"] = 0.0
        return df

    def fit(self, df: pd.DataFrame):
        df2 = df.copy()
        df2 = self._ensure_columns(df2)
        df2 = self._add_derived(df2)
        # Fit transformer
        self.transformer.fit(df2[NUMERIC + CATEGORICAL])
        # Build feature names
        # numeric names are NUMERIC
        num_names = list(NUMERIC)
        cat_ohe = self.transformer.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(cat_ohe.get_feature_names_out(CATEGORICAL))
        self.feature_names = num_names + cat_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2 = self._ensure_columns(df2)
        df2 = self._add_derived(df2)
        arr = self.transformer.transform(df2[NUMERIC + CATEGORICAL])
        # ColumnTransformer returns numpy array. Construct DF with feature_names
        if self.feature_names is None:
            # In case transform is used before fit (shouldn't happen), build approximate names
            cat_ohe = self.transformer.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(cat_ohe.get_feature_names_out(CATEGORICAL))
            self.feature_names = list(NUMERIC) + cat_names
        return pd.DataFrame(arr, columns=self.feature_names)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        return joblib.load(path)
