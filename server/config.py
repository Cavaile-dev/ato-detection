from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
WEB_DIR = BASE_DIR / "web"

MODEL_ARTIFACT_PATH = MODEL_DIR / "behavior_model.joblib"
RAW_EVENT_LOG_PATH = DATA_DIR / "stream_events.jsonl"
RISK_DECISION_LOG_PATH = DATA_DIR / "risk_decisions.jsonl"
MOCK_SESSION_PATH = DATA_DIR / "mock_sessions.jsonl"
MOCK_FEATURE_PATH = DATA_DIR / "mock_features.csv"

# Experiment framework paths
FEATURE_TOGGLE_PATH = DATA_DIR / "feature_toggles.json"   # runtime toggle config
EXPERIMENT_LOG_PATH = DATA_DIR / "experiment_log.jsonl"   # per-assessment experiment records

FEATURE_COLUMNS = [
    "mouse_velocity_mean",
    "mouse_velocity_variance",
    "mouse_acceleration_variance",
    "click_interval_std",
    "typing_speed_mean",
    "typing_speed_variance",
    "key_hold_time_mean",
    "navigation_entropy",
    "page_transition_pattern",
    "dwell_time_per_page",
]

MIN_EVENTS_FOR_SCORING = 20
ROLLING_WINDOW_SIZE = 90
ASSESSMENT_HISTORY_LIMIT = 40
MODEL_PARAMS = {
    "n_estimators": 250,
    "random_state": 42,
}
CONTAMINATION_RANGE = (0.02, 0.08)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()
