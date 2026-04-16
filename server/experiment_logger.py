"""
experiment_logger.py — Research-Grade Experiment Logging
=========================================================
Appends one JSONL record per risk assessment to data/experiment_log.jsonl.

Each record captures everything needed for reproducible behavioral biometrics
experiments:
  - which features were active / disabled at the time of assessment
  - the raw extracted value for each active parameter
  - the model feature vector used for scoring
  - the anomaly score, context deviation, combined score
  - the final decision (ALLOW / VERIFY / DENY)
  - the active parameter count for normalisation analysis

EXPERIMENT USE CASES SUPPORTED
--------------------------------
* Single-feature testing      — disable all but one toggle, run session
* Feature ablation            — disable one toggle at a time, compare scores
* Feature combination testing — enable specific subsets, measure detection rate
* Full vs minimal system      — compare all-on vs minimal-on experiments

The log is append-only and never deleted by this module.  Use the
GET /api/v1/experiment/log endpoint to query recent entries.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from server.config import EXPERIMENT_LOG_PATH

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


def _decision_label(action: str) -> str:
    """
    Map the pipeline action string to a 3-way experiment decision label.
      ALLOW_SESSION   → ALLOW
      LOG_AND_MONITOR → VERIFY
      TRIGGER_MFA     → DENY  (treated as 'block for verification')
    """
    if action == "ALLOW_SESSION":
        return "ALLOW"
    if action == "LOG_AND_MONITOR":
        return "VERIFY"
    return "DENY"


def log_experiment(
    session_id: str,
    active_params: list[str],
    disabled_params: list[str],
    raw_values: dict[str, float | None],
    model_features: dict[str, float],
    anomaly_score: float,
    raw_anomaly_score: float,
    context_deviation: float,
    combined_score: float,
    risk_level: str,
    action: str,
    active_param_count: int,
    total_param_count: int,
) -> None:
    """
    Append one experiment record to data/experiment_log.jsonl.

    This function is called from pipeline._process_batch() after every
    completed risk assessment.  It is thread-safe and write-errors are
    soft-logged (they do not crash the pipeline).

    Parameters
    ----------
    session_id        : current session UUID
    active_params     : list of currently-enabled toggle parameter names
    disabled_params   : list of currently-disabled toggle parameter names
    raw_values        : per-parameter extracted values (None if param disabled)
    model_features    : the actual dict[column → float] fed to the model
    anomaly_score     : IsolationForest-derived anomaly score [0, 1]
    context_deviation : context-layer deviation score [0, 1]
    combined_score    : weighted combination of the above [0, 1]
    risk_level        : LOW / MEDIUM / HIGH
    action            : raw pipeline action string
    active_param_count: number of currently-enabled parameters
    total_param_count : total number of known parameters (always 22)
    """
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,

        # Experiment configuration snapshot
        "active_features": active_params,
        "disabled_features": disabled_params,
        "active_param_count": active_param_count,
        "total_param_count": total_param_count,
        "feature_coverage_pct": round(100.0 * active_param_count / max(total_param_count, 1), 1),

        # Raw per-parameter values (None = disabled)
        "extracted_values": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in raw_values.items()
        },

        # Model input vector (zero-filled for disabled params)
        "model_features": {k: round(v, 6) for k, v in model_features.items()},

        # Scoring outputs
        "anomaly_score": round(anomaly_score, 4),
        "raw_anomaly_score": round(raw_anomaly_score, 4),
        "context_deviation": round(context_deviation, 4),
        "combined_score": round(combined_score, 4),

        # Decision
        "risk_level": risk_level,
        "action": action,
        "decision": _decision_label(action),
    }

    _write_record(record)


def _write_record(record: dict) -> None:
    """Thread-safe append to the experiment log JSONL file."""
    line = json.dumps(record, default=str) + "\n"
    try:
        with _write_lock:
            with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write experiment log entry: %s", exc)


def read_recent(n: int = 100) -> list[dict]:
    """
    Return the last *n* experiment log records (most recent last).
    Silently returns an empty list if the file does not exist.
    """
    path: Path = EXPERIMENT_LOG_PATH
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:] if len(lines) > n else lines
        records: list[dict] = []
        for line in recent:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read experiment log: %s", exc)
        return []
