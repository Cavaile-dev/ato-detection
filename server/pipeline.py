from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from server.config import (
    ASSESSMENT_HISTORY_LIMIT,
    MIN_EVENTS_FOR_SCORING,
    RAW_EVENT_LOG_PATH,
    RISK_DECISION_LOG_PATH,
    ROLLING_WINDOW_SIZE,
)
from server.experiment_logger import log_experiment
from server.feature_extraction import extract_features
from server.feature_manager import ALL_PARAMS, feature_manager
from server.model import BehavioralAnomalyModel
from server.risk_engine import build_risk_assessment
from server.schemas import BehaviorEvent, RiskAssessment, SessionContext, SessionState


@dataclass(slots=True)
class PendingBatch:
    session_id: str
    events: list[dict]
    response_queue: queue.Queue


class StreamingRiskPipeline:
    def __init__(self, model_service: BehavioralAnomalyModel) -> None:
        self.model_service = model_service
        self.sessions: dict[str, SessionState] = {}
        self.event_queue: queue.Queue[PendingBatch | None] = queue.Queue()
        self.lock = threading.RLock()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="ato-stream-worker")
        self.worker.start()

    def create_session(self, payload: dict) -> SessionState:
        session_id = str(uuid.uuid4())
        context = SessionContext.from_payload(payload, session_id=session_id)
        state = SessionState(context=context, started_at=context.login_time)
        initial_features, _ = extract_features(state)
        active_count = len(feature_manager.active_params())
        state.latest_assessment = build_risk_assessment(
            state,
            anomaly_score=0.0,
            raw_anomaly_score=0.0,
            features=initial_features,
            feature_deviations={},
            score_ready=False,
            window_event_count=0,
            active_param_count=active_count,
        )
        state.assessment_history.append(state.latest_assessment)

        with self.lock:
            self.sessions[session_id] = state
        return state

    def submit_events(self, session_id: str, raw_events: list[dict], timeout: float = 3.0) -> RiskAssessment:
        if not raw_events:
            return self.get_assessment(session_id)
        with self.lock:
            if session_id not in self.sessions:
                raise KeyError(f"Unknown session_id: {session_id}")

        response_queue: queue.Queue = queue.Queue(maxsize=1)
        self.event_queue.put(PendingBatch(session_id=session_id, events=list(raw_events), response_queue=response_queue))
        return response_queue.get(timeout=timeout)

    def end_session(self, session_id: str) -> RiskAssessment:
        with self.lock:
            state = self.sessions[session_id]
            state.ended_at = datetime.now(timezone.utc)
            return state.latest_assessment

    def get_assessment(self, session_id: str) -> RiskAssessment:
        with self.lock:
            state = self.sessions[session_id]
            return state.latest_assessment

    def get_session_snapshot(self, session_id: str) -> dict:
        with self.lock:
            state = self.sessions[session_id]
            assessment = state.latest_assessment.to_dict() if state.latest_assessment else {}
            return {
                "session_id": state.session_id,
                "user_id": state.context.user_id,
                "event_count": state.event_count,
                "login_time": state.context.login_time.isoformat(),
                "ip_address": state.context.ip_address,
                "device": state.context.device.to_dict(),
                "assessment": assessment,
                "assessment_history": [entry.to_dict() for entry in state.assessment_history[-ASSESSMENT_HISTORY_LIMIT:]],
                "recent_events": [event.to_dict() for event in state.events[-20:]],
            }

    def shutdown(self) -> None:
        self.event_queue.put(None)
        self.worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while True:
            batch = self.event_queue.get()
            if batch is None:
                self.event_queue.task_done()
                break
            try:
                assessment = self._process_batch(batch.session_id, batch.events)
                batch.response_queue.put(assessment, timeout=1.0)
            except Exception as exc:  # pragma: no cover - operational safety path
                batch.response_queue.put(
                    RiskAssessment(
                        session_id=batch.session_id,
                        anomaly_score=0.0,
                        raw_anomaly_score=0.0,
                        risk_level="MEDIUM",
                        action="LOG_AND_MONITOR",
                        detected_state="SUSPICIOUS",
                        context_deviation=0.0,
                        combined_score=0.0,
                        reasons=[f"Pipeline exception: {exc}"],
                        features={},
                    )
                )
            finally:
                self.event_queue.task_done()

    def _process_batch(self, session_id: str, raw_events: list[dict]) -> RiskAssessment:
        with self.lock:
            state = self.sessions[session_id]

        parsed_events = [BehaviorEvent.from_payload(event) for event in raw_events]
        parsed_events.sort(key=lambda event: event.timestamp)

        with self.lock:
            state.events.extend(parsed_events)
            for event in parsed_events:
                if event.page and event.event_type == "navigation":
                    if not state.navigation_sequence or state.navigation_sequence[-1] != event.page:
                        state.navigation_sequence.append(event.page)

        window_state = self._build_window_state(state)

        # extract_features now returns (model_features, raw_values)
        features, raw_values = extract_features(window_state)
        window_event_count = len(window_state.events)

        # Snapshot active/disabled params at this instant (may change between assessments)
        active_params = feature_manager.active_params()
        disabled_params = feature_manager.disabled_params()
        active_count = len(active_params)

        if window_event_count >= MIN_EVENTS_FOR_SCORING:
            prediction = self.model_service.predict(features)
            anomaly_score = prediction.anomaly_score
            raw_anomaly_score = prediction.raw_score
            feature_deviations = prediction.feature_deviations
            score_ready = True
        else:
            anomaly_score = 0.0
            raw_anomaly_score = 0.0
            feature_deviations = {}
            score_ready = False

        assessment = build_risk_assessment(
            state,
            anomaly_score=anomaly_score,
            raw_anomaly_score=raw_anomaly_score,
            features=features,
            feature_deviations=feature_deviations,
            score_ready=score_ready,
            window_event_count=window_event_count,
            active_param_count=active_count,
        )

        with self.lock:
            state.latest_assessment = assessment
            state.assessment_history.append(assessment)
            if len(state.assessment_history) > ASSESSMENT_HISTORY_LIMIT:
                state.assessment_history = state.assessment_history[-ASSESSMENT_HISTORY_LIMIT:]

        # --- Experiment log (one record per assessment) ---
        log_experiment(
            session_id=session_id,
            active_params=active_params,
            disabled_params=disabled_params,
            raw_values=raw_values,
            model_features=features,
            anomaly_score=anomaly_score,
            raw_anomaly_score=raw_anomaly_score,
            context_deviation=assessment.context_deviation,
            combined_score=assessment.combined_score,
            risk_level=assessment.risk_level,
            action=assessment.action,
            active_param_count=active_count,
            total_param_count=len(ALL_PARAMS),
        )

        self._append_jsonl(
            RAW_EVENT_LOG_PATH,
            {
                "session_id": session_id,
                "user_id": state.context.user_id,
                "batch_size": len(parsed_events),
                "events": [event.to_dict() for event in parsed_events],
            },
        )
        self._append_jsonl(RISK_DECISION_LOG_PATH, assessment.to_dict())
        return assessment

    @staticmethod
    def _build_window_state(state: SessionState) -> SessionState:
        recent_events = list(state.events[-ROLLING_WINDOW_SIZE:])
        recent_navigation = [
            event.page
            for event in recent_events
            if event.page and event.event_type in {"navigation", "page_enter"}
        ]
        started_at = (
            datetime.fromtimestamp(recent_events[0].timestamp, tz=timezone.utc)
            if recent_events
            else state.started_at
        )
        return SessionState(
            context=state.context,
            started_at=started_at,
            events=recent_events,
            navigation_sequence=recent_navigation,
            latest_assessment=state.latest_assessment,
        )

    @staticmethod
    def _append_jsonl(path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
