from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from server.config import CONTAMINATION_RANGE, FEATURE_COLUMNS, MODEL_DIR, MODEL_PARAMS

CRITICAL_FEATURES = ["mouse_velocity_mean", "decision_latency_mean"]


@dataclass(slots=True)
class ModelPrediction:
    anomaly_score: float
    raw_score: float
    feature_deviations: dict[str, float] = field(default_factory=dict)


class BehavioralAnomalyModel:
    def __init__(self, base_dir=MODEL_DIR) -> None:
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> "BehavioralAnomalyModel":
        # No-op in per-user setup; models loaded dynamically
        return self

    def _path(self, username: str) -> Path:
        return self.base_dir / f"{username}.joblib"

    def train(self, username: str, feature_frame: pd.DataFrame) -> dict[str, Any]:
        frame = self._coerce_frame(feature_frame)
        if len(frame) < 5:
            return {"status": "skipped_cold_start"}

        path = self._path(username)
        new_mean = frame.mean()

        if path.exists():
            old_payload = joblib.load(path)
            old_mean = old_payload.get("historical_mean")
            if old_mean is not None:
                # Drift Protection with Weighted Critical Features
                diff = abs(new_mean - old_mean) / (abs(old_mean) + 1e-6)
                if any(diff.get(f, 0) > 0.5 for f in CRITICAL_FEATURES if f in diff):
                    return {"status": "rejected_concept_drift"}

        contamination = self._tune_contamination(len(frame))
        scaler = StandardScaler()
        transformed = scaler.fit_transform(frame)
        model = IsolationForest(**MODEL_PARAMS, contamination=contamination)
        model.fit(transformed)

        raw_scores = -model.score_samples(transformed)
        threshold = float(np.quantile(raw_scores, 1.0 - contamination))

        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "threshold": threshold,
                "historical_mean": new_mean,
                "feature_columns": FEATURE_COLUMNS,
            },
            path,
        )
        return {
            "status": "trained",
            "artifact_path": str(path),
            "trained_samples": len(frame),
        }

    def predict(self, username: str, features: dict[str, float]) -> ModelPrediction:
        path = self._path(username)
        if not path.exists():
            return ModelPrediction(
                anomaly_score=1.0,  # Cold Start: MEDIUM risk default
                raw_score=0.0,
                feature_deviations={}
            )

        payload = joblib.load(path)
        model = payload["model"]
        scaler = payload["scaler"]
        threshold = payload["threshold"]

        vector = pd.DataFrame(
            [{column: float(features.get(column, 0.0)) for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        transformed_matrix = scaler.transform(vector)
        transformed = transformed_matrix[0]
        raw_score = float(-model.score_samples(transformed_matrix)[0])
        
        # Stabilize Threshold logic
        anomaly_score = float(raw_score / (threshold + 1e-6))

        feature_deviations = {
            column: float(abs(transformed[index]))
            for index, column in enumerate(FEATURE_COLUMNS)
        }
        return ModelPrediction(
            anomaly_score=anomaly_score,
            raw_score=raw_score,
            feature_deviations=feature_deviations,
        )

    @staticmethod
    def _coerce_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
        frame = feature_frame.copy()
        missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        # To avoid failure on newer schema additions over old data, we zero-fill missing columns
        for col in missing_columns:
            frame[col] = 0.0
        return frame[FEATURE_COLUMNS].astype(float)

    @staticmethod
    def _tune_contamination(training_rows: int) -> float:
        minimum, maximum = CONTAMINATION_RANGE
        tuned = 12.0 / max(training_rows, 1)
        return float(min(max(tuned, minimum), maximum))
