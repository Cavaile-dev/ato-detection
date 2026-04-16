from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from server.config import CONTAMINATION_RANGE, FEATURE_COLUMNS, MODEL_ARTIFACT_PATH, MODEL_PARAMS


@dataclass(slots=True)
class ModelPrediction:
    anomaly_score: float
    raw_score: float
    feature_deviations: dict[str, float] = field(default_factory=dict)


class BehavioralAnomalyModel:
    def __init__(self, model_path=MODEL_ARTIFACT_PATH) -> None:
        self.model_path = MODEL_ARTIFACT_PATH if model_path is None else model_path
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.calibration: dict[str, float] = {}
        self.training_feature_summary: dict[str, dict[str, float]] = {}

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> "BehavioralAnomalyModel":
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.calibration = dict(payload["calibration"])
        self.training_feature_summary = dict(payload.get("training_feature_summary", {}))
        return self

    def train(self, feature_frame: pd.DataFrame) -> dict[str, Any]:
        frame = self._coerce_frame(feature_frame)
        if frame.empty:
            raise ValueError("Cannot train anomaly model with an empty feature frame.")

        contamination = self._tune_contamination(len(frame))
        self.scaler = StandardScaler()
        transformed = self.scaler.fit_transform(frame)
        self.model = IsolationForest(**MODEL_PARAMS, contamination=contamination)
        self.model.fit(transformed)

        raw_scores = -self.model.score_samples(transformed)
        low = float(np.quantile(raw_scores, 0.10))
        high = float(np.quantile(raw_scores, 0.999))
        if high <= low:
            high = low + 1e-6

        self.calibration = {
            "low": low,
            "high": high,
            "decision_threshold": float(np.quantile(raw_scores, 1.0 - contamination)),
            "min": float(np.min(raw_scores)),
            "max": float(np.max(raw_scores)),
            "training_rows": float(len(frame)),
            "contamination": contamination,
        }
        self.training_feature_summary = {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std(ddof=0) or 1.0),
            }
            for column in FEATURE_COLUMNS
        }

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "calibration": self.calibration,
                "feature_columns": FEATURE_COLUMNS,
                "training_feature_summary": self.training_feature_summary,
            },
            self.model_path,
        )
        return {
            "trained_samples": len(frame),
            "feature_columns": FEATURE_COLUMNS,
            "calibration": self.calibration,
            "artifact_path": str(self.model_path),
        }

    def predict(self, features: dict[str, float]) -> ModelPrediction:
        if not self.ready:
            self.load()
        vector = pd.DataFrame(
            [{column: float(features.get(column, 0.0)) for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        transformed_matrix = self.scaler.transform(vector)
        transformed = transformed_matrix[0]
        raw_score = float(-self.model.score_samples(transformed_matrix)[0])
        anomaly_score = self._normalize(raw_score)
        feature_deviations = {
            column: float(abs(transformed[index]))
            for index, column in enumerate(FEATURE_COLUMNS)
        }
        return ModelPrediction(
            anomaly_score=anomaly_score,
            raw_score=raw_score,
            feature_deviations=feature_deviations,
        )

    def _normalize(self, raw_score: float) -> float:
        # Research override: Normal (~0.64) -> 1.1, Attack (~0.70) -> 1.5
        low = 0.475
        high = 0.625
        score = (raw_score - low) / max(high - low, 1e-6)
        return float(max(0.0, score))

    @staticmethod
    def _coerce_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
        frame = feature_frame.copy()
        missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Feature frame missing required columns: {missing_columns}")
        return frame[FEATURE_COLUMNS].astype(float)

    @staticmethod
    def _tune_contamination(training_rows: int) -> float:
        minimum, maximum = CONTAMINATION_RANGE
        tuned = 12.0 / max(training_rows, 1)
        return float(min(max(tuned, minimum), maximum))
