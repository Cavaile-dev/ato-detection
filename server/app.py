from __future__ import annotations

import atexit

from flask import Flask, jsonify, request

from flask_cors import CORS

from server.config import WEB_DIR
from server.experiment_logger import read_recent
from server.feature_manager import ALL_PARAMS, feature_manager
from server.mock_data import generate_mock_dataset
from server.model import BehavioralAnomalyModel
from server.pipeline import StreamingRiskPipeline
from server.train_model import train_default_model


def bootstrap_model(model_service: BehavioralAnomalyModel) -> None:
    try:
        model_service.load()
    except FileNotFoundError:
        generate_mock_dataset()
        train_default_model()
        model_service.load()


app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

model_service = BehavioralAnomalyModel()
bootstrap_model(model_service)
pipeline = StreamingRiskPipeline(model_service=model_service)
atexit.register(pipeline.shutdown)


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return app.send_static_file("login.html")


@app.get("/dashboard")
def dashboard():
    return app.send_static_file("dashboard.html")


@app.get("/home")
def home():
    return app.send_static_file("home.html")


@app.get("/profile")
def profile():
    return app.send_static_file("profile.html")


@app.get("/settings")
def settings():
    return app.send_static_file("settings.html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health_check():
    return jsonify({"status": "ok", "model_ready": model_service.ready})


# ---------------------------------------------------------------------------
# Data / model management
# ---------------------------------------------------------------------------

@app.post("/api/v1/mock-data")
def generate_mock_data_route():
    payload = request.get_json(silent=True) or {}
    normal_sessions = int(payload.get("normal_sessions", 240))
    attacker_sessions = int(payload.get("attacker_sessions", 120))
    frame, records = generate_mock_dataset(normal_sessions=normal_sessions, attacker_sessions=attacker_sessions)
    return jsonify(
        {
            "status": "generated",
            "normal_sessions": normal_sessions,
            "attacker_sessions": attacker_sessions,
            "feature_rows": len(frame),
            "session_rows": len(records),
        }
    )


@app.post("/api/v1/model/train")
def train_model_route():
    summary = train_default_model()
    model_service.load()
    return jsonify({"status": "trained", "summary": summary})


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

import json
from pathlib import Path

BASELINES_PATH = Path("data/baselines.json")

@app.post("/api/v1/sessions/enroll")
def enroll_baseline():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    events = payload.get("events", [])
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # In a real system, we would process these events through feature_extraction.py
    # to store 'baseline_features'. For prototype ease, we'll store the event count 
    # and extraction snapshot.
    from server.schemas import BehaviorEvent, SessionContext, SessionState
    from server.feature_extraction import extract_features
    from datetime import datetime, timezone
    
    parsed_events = [BehaviorEvent.from_payload(e) for e in events]
    state = SessionState(
        context=SessionContext(session_id="baseline", user_id=user_id, ip_address=""),
        started_at=datetime.now(timezone.utc),
        events=parsed_events
    )
    extracted_features, raw_features = extract_features(state)
    
    # Store to local file
    try:
        baselines = {}
        if BASELINES_PATH.exists():
            baselines = json.loads(BASELINES_PATH.read_text("utf-8"))
        baselines[user_id] = {
            "baseline_features": raw_features,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(parsed_events)
        }
        BASELINES_PATH.write_text(json.dumps(baselines, indent=2), "utf-8")
    except Exception as e:
        app.logger.error(f"Failed to save baseline: {e}")

    return jsonify({"status": "enrolled", "user_id": user_id, "features": raw_features})



@app.post("/api/v1/sessions/start")
def start_session():
    payload = request.get_json(silent=True) or {}
    payload.setdefault("ip_address", request.headers.get("X-Forwarded-For", request.remote_addr or "203.0.113.10"))
    bootstrap_events = payload.pop("bootstrap_events", [])
    state = pipeline.create_session(payload)
    assessment = state.latest_assessment
    if bootstrap_events:
        assessment = pipeline.submit_events(state.session_id, bootstrap_events)
    return jsonify(
        {
            "session_id": state.session_id,
            "assessment": assessment.to_dict(),
            "context": state.context.to_dict(),
        }
    )


@app.post("/api/v1/events")
def ingest_events():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    events = payload.get("events", [])
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not isinstance(events, list):
        return jsonify({"error": "events must be a list"}), 400

    assessment = pipeline.submit_events(session_id=session_id, raw_events=events)
    return jsonify(assessment.to_dict())


@app.get("/api/v1/sessions/<session_id>")
def session_status(session_id: str):
    try:
        return jsonify(pipeline.get_session_snapshot(session_id))
    except KeyError:
        return jsonify({"error": "session not found"}), 404


@app.post("/api/v1/sessions/<session_id>/end")
def end_session(session_id: str):
    try:
        assessment = pipeline.end_session(session_id)
    except KeyError:
        return jsonify({"error": "session not found"}), 404
    return jsonify(assessment.to_dict())


# ---------------------------------------------------------------------------
# Experiment Framework API
# ---------------------------------------------------------------------------

@app.get("/api/v1/experiment/toggles")
def get_toggles():
    """
    GET /api/v1/experiment/toggles

    Returns the current per-parameter feature toggle configuration along with
    metadata about which parameters map to model columns and which are context
    or reserved parameters.

    Response schema:
    {
      "toggles": { "<param>": bool, ... },       // 22 entries
      "active_params": ["dwell_time", ...],       // enabled only
      "disabled_params": ["error_rate", ...],     // disabled only
      "active_count": 18,
      "total_count": 22,
      "active_model_columns": ["mouse_velocity_mean", ...]
    }
    """
    return jsonify({
        "toggles": feature_manager.get_all_toggles(),
        "active_params": feature_manager.active_params(),
        "disabled_params": feature_manager.disabled_params(),
        "active_count": len(feature_manager.active_params()),
        "total_count": len(ALL_PARAMS),
        "active_model_columns": feature_manager.active_model_columns(),
    })


@app.post("/api/v1/experiment/toggles")
def update_toggles():
    """
    POST /api/v1/experiment/toggles
    Body: { "<param>": bool, ... }   (partial or full update)

    Applies toggle changes at runtime — no server restart required.
    Changes take effect on the next event batch processed by the pipeline.
    Writes the new configuration to data/feature_toggles.json.

    Response:
    {
      "status": "ok",
      "errors": {},           // empty if all updates applied
      "toggles": { ... },     // full updated config
      "active_count": 18,
      "total_count": 22
    }
    """
    payload = request.get_json(silent=True) or {}
    errors = feature_manager.update_toggles(payload)
    status = "ok" if not errors else "partial"
    return jsonify({
        "status": status,
        "errors": errors,
        "toggles": feature_manager.get_all_toggles(),
        "active_count": len(feature_manager.active_params()),
        "total_count": len(ALL_PARAMS),
    }), (200 if not errors else 207)


@app.get("/api/v1/experiment/log")
def get_experiment_log():
    """
    GET /api/v1/experiment/log?n=100

    Returns the last *n* experiment log records (default 100, max 1000).
    Each record contains: active_features, disabled_features, extracted_values,
    model_features, scores, risk_level, action, decision, timestamps.

    Use this endpoint to pull experiment results for analysis.
    """
    n = min(int(request.args.get("n", 100)), 1000)
    records = read_recent(n)
    return jsonify({
        "count": len(records),
        "records": records,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
