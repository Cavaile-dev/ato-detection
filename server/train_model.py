from __future__ import annotations

import pandas as pd

from server.config import FEATURE_COLUMNS, MOCK_FEATURE_PATH
from server.mock_data import generate_mock_dataset
from server.model import BehavioralAnomalyModel


def bootstrap_training_data() -> pd.DataFrame:
    if MOCK_FEATURE_PATH.exists():
        existing = pd.read_csv(MOCK_FEATURE_PATH)
        if len(existing) >= 100 and "label" in existing.columns:
            return existing
    frame, _ = generate_mock_dataset()
    return frame


def train_default_model() -> dict:
    feature_frame = bootstrap_training_data()
    normal_frame = feature_frame[feature_frame["label"] == 0][FEATURE_COLUMNS]
    model_service = BehavioralAnomalyModel()
    return model_service.train(normal_frame)


def main() -> None:
    summary = train_default_model()
    print("Model trained successfully")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
