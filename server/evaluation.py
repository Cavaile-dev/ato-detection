from __future__ import annotations

from sklearn.metrics import classification_report, roc_auc_score

from server.config import FEATURE_COLUMNS, MOCK_FEATURE_PATH
from server.model import BehavioralAnomalyModel
from server.train_model import bootstrap_training_data, train_default_model


def evaluate_model(threshold: float = 0.65) -> dict:
    if not MOCK_FEATURE_PATH.exists():
        train_default_model()

    frame = bootstrap_training_data()
    model_service = BehavioralAnomalyModel().load()

    scores = [
        model_service.predict(row[FEATURE_COLUMNS].to_dict()).anomaly_score
        for _, row in frame.iterrows()
    ]
    truth = frame["label"].astype(int).tolist()
    predictions = [1 if score >= threshold else 0 for score in scores]

    report = classification_report(truth, predictions, target_names=["normal", "attacker"], zero_division=0)
    auc = roc_auc_score(truth, scores)
    return {
        "roc_auc": float(auc),
        "report": report,
    }


def main() -> None:
    result = evaluate_model()
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(result["report"])


if __name__ == "__main__":
    main()
