from __future__ import annotations

import atexit
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from flask import Flask, jsonify, request

from flask_cors import CORS

from server.config import DATA_DIR, WEB_DIR
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

DATABASE_PATH = DATA_DIR / "app.db"
BASELINES_PATH = DATA_DIR / "baselines.json"
active_behavior_sessions: dict[str, dict] = {}


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NOT NULL,
                risk_level TEXT,
                anomaly_score REAL,
                mfa_status TEXT DEFAULT 'NONE'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS behavior_features (
                session_id TEXT PRIMARY KEY,
                mouse_velocity_mean REAL,
                mouse_velocity_variance REAL,
                mouse_velocity_std REAL,
                mouse_acceleration_variance REAL,
                click_interval_mean REAL,
                click_interval_std REAL,
                typing_speed_mean REAL,
                typing_speed_variance REAL,
                typing_speed_std REAL,
                key_hold_time_mean REAL,
                key_hold_time_variance REAL,
                key_hold_time_std REAL,
                navigation_entropy REAL,
                page_transition_pattern REAL,
                dwell_time_per_page_mean REAL,
                dwell_time_per_page_variance REAL,
                decision_latency_mean REAL,
                decision_latency_variance REAL,
                error_rate REAL,
                hesitation_mean REAL,
                hesitation_variance REAL,
                tab_switch_count REAL,
                idle_mean REAL,
                clipboard_count REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                x INTEGER,
                y INTEGER,
                timestamp DATETIME NOT NULL,
                extra_data TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
            [
                ("alice", "hunter2-demo"),
                ("bravo", "hunter2-demo"),
                ("charlie", "hunter2-demo"),
            ],
        )
        connection.commit()


def create_behavior_session(session_id: str, username: str) -> None:
    active_behavior_sessions[session_id] = {
        "username": username,
        "started_at": datetime.now(timezone.utc),
        "mouse_count": 0,
        "click_count": 0,
        "scroll_count": 0,
        "click_intervals": [],
        "mouse_speeds": [],
        "saved": False,
    }


def update_behavior_session(session_id: str, username: str | None, events: list[dict]) -> None:
    session = active_behavior_sessions.get(session_id)
    if session is None:
        create_behavior_session(session_id, username or "unknown")
        session = active_behavior_sessions[session_id]

    if username:
        session["username"] = username

    for event in events:
        event_type = str(event.get("type") or event.get("event_type") or "")
        if event_type == "mouse_move":
            session["mouse_count"] += 1
            velocity = event.get("velocity")
            if velocity is not None:
                session["mouse_speeds"].append(float(velocity))
        elif event_type == "click":
            session["click_count"] += 1
            click_interval = event.get("click_interval")
            if click_interval is not None:
                session["click_intervals"].append(float(click_interval))
        elif event_type == "scroll":
            session["scroll_count"] += 1


def trigger_model_update(username: str) -> None:
    with get_db_connection() as conn:
        records = conn.execute('''
            SELECT f.* FROM sessions s
            JOIN behavior_features f ON s.session_id = f.session_id
            WHERE s.username = ? AND (s.risk_level = 'LOW' OR s.mfa_status = 'SUCCESS')
            ORDER BY s.end_time DESC LIMIT 20
        ''', (username,)).fetchall()
        
    if len(records) >= 5:
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in records])
        # Drop the session_id to isolate features
        if 'session_id' in df.columns:
            df = df.drop(columns=['session_id'])
        model_service.train(username, df)


def persist_research_session(session_id: str, username: str, start_time: datetime, end_time: datetime, risk_level: str, anomaly_score: float, model_features: dict, raw_events: list[dict]) -> None:
    with get_db_connection() as connection:
        # 1. Insert into sessions
        connection.execute(
            "INSERT OR REPLACE INTO sessions (session_id, username, start_time, end_time, risk_level, anomaly_score) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, username, start_time.isoformat(), end_time.isoformat(), risk_level, anomaly_score)
        )
        
        # 2. Insert into behavior_features
        columns = ["session_id"] + list(model_features.keys())
        placeholders = ", ".join(["?"] * len(columns))
        values = [session_id] + [float(model_features.get(k, 0.0)) for k in model_features.keys()]
        
        connection.execute(f"INSERT OR REPLACE INTO behavior_features ({', '.join(columns)}) VALUES ({placeholders})", values)
        
        # 3. Insert raw events
        event_recs = []
        for e in raw_events:
            metadata = {k: v for k, v in e.items() if k not in ("type", "event_type", "x", "y", "timestamp")}
            event_recs.append((
                session_id,
                e.get("type", e.get("event_type", "unknown")),
                e.get("x"),
                e.get("y"),
                datetime.fromtimestamp(e.get("timestamp", end_time.timestamp()), tz=timezone.utc).isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            
        connection.executemany(
            "INSERT INTO raw_events (session_id, event_type, x, y, timestamp, extra_data) VALUES (?, ?, ?, ?, ?, ?)",
            event_recs
        )
        connection.commit()


init_db()


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


@app.get("/transaction")
def transaction():
    return app.send_static_file("transaction.html")


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
# Simple login
# ---------------------------------------------------------------------------

@app.post("/api/v1/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip().lower()
    password = str(payload.get("password") or "")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    with get_db_connection() as connection:
        user = connection.execute(
            "SELECT id, username FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()

    if user is None:
        return jsonify({"error": "invalid username or password"}), 401

    return jsonify({"status": "ok", "username": user["username"]})


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
    username = str(payload.get("username") or payload.get("user_id") or "").strip().lower()
    if not username:
        return jsonify({"error": "username is required"}), 400

    payload["user_id"] = username
    payload.setdefault("ip_address", request.headers.get("X-Forwarded-For", request.remote_addr or "203.0.113.10"))
    bootstrap_events = payload.pop("bootstrap_events", [])
    state = pipeline.create_session(payload)
    create_behavior_session(state.session_id, username)

    assessment = state.latest_assessment
    if bootstrap_events:
        update_behavior_session(state.session_id, username, bootstrap_events)
        assessment = pipeline.submit_events(state.session_id, bootstrap_events)
    return jsonify(
        {
            "session_id": state.session_id,
            "username": username,
            "assessment": assessment.to_dict(),
            "context": state.context.to_dict(),
        }
    )


@app.post("/api/v1/events")
def ingest_events():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    username = str(payload.get("username") or "").strip().lower()
    events = payload.get("events", [])
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not isinstance(events, list):
        return jsonify({"error": "events must be a list"}), 400

    update_behavior_session(str(session_id), username or None, events)
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
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip().lower()
    try:
        assessment = pipeline.end_session(session_id)
        state_snapshot = pipeline.get_session_snapshot(session_id)
    except KeyError:
        return jsonify({"error": "session not found"}), 404

    # Extract raw events dynamically
    raw_events = state_snapshot.get("recent_events", [])
    
    start_time = datetime.fromisoformat(state_snapshot["login_time"]) if "login_time" in state_snapshot else datetime.now(timezone.utc)
    
    # Track Research Session locally
    persist_research_session(
        session_id=session_id,
        username=username or "unknown",
        start_time=start_time,
        end_time=datetime.now(timezone.utc),
        risk_level=assessment.risk_level,
        anomaly_score=assessment.anomaly_score,
        model_features=assessment.features,
        raw_events=raw_events
    )
    
    if assessment.risk_level == "LOW":
        trigger_model_update(username)

    response = assessment.to_dict()
    response["behavior_saved"] = True
    return jsonify(response)


@app.post("/api/v1/transaction/pay")
def pay_transaction():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or "").strip()
    username = str(payload.get("username") or "").strip().lower()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        assessment = pipeline.end_session(session_id)
        state_snapshot = pipeline.get_session_snapshot(session_id)
    except KeyError:
        return jsonify({"error": "session not found"}), 404

    raw_events = state_snapshot.get("recent_events", [])
    start_time = datetime.fromisoformat(state_snapshot["login_time"]) if "login_time" in state_snapshot else datetime.now(timezone.utc)

    persist_research_session(
        session_id=session_id,
        username=username or "unknown",
        start_time=start_time,
        end_time=datetime.now(timezone.utc),
        risk_level=assessment.risk_level,
        anomaly_score=assessment.anomaly_score,
        model_features=assessment.features,
        raw_events=raw_events
    )
    
    if assessment.risk_level == "LOW":
        trigger_model_update(username)

    return jsonify(
        {
            "status": "paid",
            "session_id": session_id,
            "username": username,
            "assessment": assessment.to_dict(),
            "behavior_saved": True,
        }
    )


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
