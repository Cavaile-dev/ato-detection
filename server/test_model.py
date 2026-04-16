from __future__ import annotations

from server.mock_data import generate_mock_dataset
from server.model import BehavioralAnomalyModel
from server.train_model import train_default_model


def main() -> None:
    train_default_model()
    frame, _ = generate_mock_dataset(normal_sessions=2, attacker_sessions=2, seed=99, persist=False)
    model_service = BehavioralAnomalyModel().load()

    print("Smoke test predictions:")
    for _, row in frame.iterrows():
        prediction = model_service.predict(row.to_dict())
        print(
            {
                "session_id": row["session_id"],
                "scenario": row["scenario"],
                "label": int(row["label"]),
                "anomaly_score": round(prediction.anomaly_score, 4),
            }
        )


if __name__ == "__main__":
    main()
