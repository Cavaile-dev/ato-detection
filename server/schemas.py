from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def parse_datetime(value: Any | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value in (None, ""):
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class DeviceInfo:
    device_id: str = "unknown-device"
    device_type: str = "desktop"
    os: str = "unknown-os"
    browser: str = "unknown-browser"
    user_agent: str = ""

    @property
    def fingerprint(self) -> str:
        core = self.device_id.strip() or f"{self.device_type}:{self.os}:{self.browser}"
        return core.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "os": self.os,
            "browser": self.browser,
            "user_agent": self.user_agent,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Any | None) -> "DeviceInfo":
        if isinstance(payload, DeviceInfo):
            return payload
        if isinstance(payload, str):
            return cls(device_id=payload, user_agent=payload)

        payload = payload or {}
        return cls(
            device_id=str(payload.get("device_id") or payload.get("fingerprint") or "unknown-device"),
            device_type=str(payload.get("device_type") or "desktop"),
            os=str(payload.get("os") or "unknown-os"),
            browser=str(payload.get("browser") or "unknown-browser"),
            user_agent=str(payload.get("user_agent") or payload.get("userAgent") or ""),
        )


@dataclass(slots=True)
class UserProfile:
    typical_login_hour: float = 9.0
    trusted_ips: list[str] = field(default_factory=list)
    trusted_devices: list[str] = field(default_factory=list)
    preferred_pages: list[str] = field(default_factory=lambda: ["/home", "/profile", "/settings"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "typical_login_hour": self.typical_login_hour,
            "trusted_ips": list(self.trusted_ips),
            "trusted_devices": list(self.trusted_devices),
            "preferred_pages": list(self.preferred_pages),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Any | None,
        fallback_ip: str = "",
        fallback_device: str = "",
        fallback_hour: float = 9.0,
    ) -> "UserProfile":
        if isinstance(payload, UserProfile):
            return payload

        payload = payload or {}
        trusted_ips = [str(ip) for ip in payload.get("trusted_ips") or ([] if not fallback_ip else [fallback_ip])]
        trusted_devices = [
            str(device)
            for device in payload.get("trusted_devices") or ([] if not fallback_device else [fallback_device])
        ]
        typical_login_hour = float(payload.get("typical_login_hour", fallback_hour))
        return cls(
            typical_login_hour=typical_login_hour,
            trusted_ips=trusted_ips,
            trusted_devices=[device.lower() for device in trusted_devices],
            preferred_pages=[str(page) for page in payload.get("preferred_pages") or ["/home", "/profile", "/settings"]],
        )


@dataclass(slots=True)
class BehaviorEvent:
    event_type: str
    timestamp: float
    page: str = ""
    x: float | None = None
    y: float | None = None
    velocity: float | None = None
    acceleration: float | None = None
    click_interval: float | None = None
    key_interval: float | None = None
    hold_time: float | None = None
    hesitation: float | None = None
    dwell_time: float | None = None
    scroll_delta: float | None = None
    field_name: str = ""
    trajectory_angle: float | None = None
    direction_change: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.event_type,
            "timestamp": self.timestamp,
            "page": self.page,
            "x": self.x,
            "y": self.y,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "click_interval": self.click_interval,
            "key_interval": self.key_interval,
            "hold_time": self.hold_time,
            "hesitation": self.hesitation,
            "dwell_time": self.dwell_time,
            "scroll_delta": self.scroll_delta,
            "field_name": self.field_name,
            "trajectory_angle": self.trajectory_angle,
            "direction_change": self.direction_change,
        }
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BehaviorEvent":
        raw_timestamp = payload.get("timestamp", payload.get("time", datetime.now(timezone.utc).timestamp()))
        timestamp = float(raw_timestamp)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return cls(
            event_type=str(payload.get("type") or payload.get("event_type") or "unknown"),
            timestamp=timestamp,
            page=str(payload.get("page") or payload.get("route") or ""),
            x=float(payload["x"]) if payload.get("x") is not None else None,
            y=float(payload["y"]) if payload.get("y") is not None else None,
            velocity=float(payload["velocity"]) if payload.get("velocity") is not None else None,
            acceleration=float(payload["acceleration"]) if payload.get("acceleration") is not None else None,
            click_interval=float(payload["click_interval"]) if payload.get("click_interval") is not None else None,
            key_interval=float(payload["key_interval"]) if payload.get("key_interval") is not None else None,
            hold_time=float(payload["hold_time"]) if payload.get("hold_time") is not None else None,
            hesitation=float(payload["hesitation"]) if payload.get("hesitation") is not None else None,
            dwell_time=float(payload["dwell_time"]) if payload.get("dwell_time") is not None else None,
            scroll_delta=float(payload["scroll_delta"]) if payload.get("scroll_delta") is not None else None,
            field_name=str(payload.get("field_name") or payload.get("field") or ""),
            trajectory_angle=float(payload["trajectory_angle"]) if payload.get("trajectory_angle") is not None else None,
            direction_change=float(payload["direction_change"]) if payload.get("direction_change") is not None else None,
            metadata={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "type",
                    "event_type",
                    "timestamp",
                    "time",
                    "page",
                    "route",
                    "x",
                    "y",
                    "velocity",
                    "acceleration",
                    "click_interval",
                    "key_interval",
                    "hold_time",
                    "hesitation",
                    "dwell_time",
                    "scroll_delta",
                    "field_name",
                    "field",
                    "trajectory_angle",
                    "direction_change",
                }
            },
        )


@dataclass(slots=True)
class SessionContext:
    session_id: str
    user_id: str
    ip_address: str
    device: DeviceInfo
    login_time: datetime
    profile: UserProfile

    def login_hour(self) -> float:
        return self.login_time.hour + (self.login_time.minute / 60.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "device": self.device.to_dict(),
            "login_time": self.login_time.isoformat(),
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], session_id: str) -> "SessionContext":
        ip_address = str(payload.get("ip_address") or payload.get("ip") or "203.0.113.10")
        device = DeviceInfo.from_payload(payload.get("device") or payload.get("device_info"))
        login_time = parse_datetime(payload.get("login_time"))
        profile = UserProfile.from_payload(
            payload.get("profile"),
            fallback_ip=ip_address,
            fallback_device=device.fingerprint,
            fallback_hour=login_time.hour + (login_time.minute / 60.0),
        )
        return cls(
            session_id=session_id,
            user_id=str(payload.get("user_id") or "demo-user"),
            ip_address=ip_address,
            device=device,
            login_time=login_time,
            profile=profile,
        )


@dataclass(slots=True)
class RiskAssessment:
    session_id: str
    anomaly_score: float
    raw_anomaly_score: float
    risk_level: str
    action: str
    detected_state: str
    context_deviation: float
    combined_score: float
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    top_deviation_features: list[str] = field(default_factory=list)
    processed_events: int = 0
    score_ready: bool = False
    window_event_count: int = 0
    # Experiment framework fields — number of active / total toggle parameters
    active_param_count: int = 0
    total_param_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "anomaly_score": round(self.anomaly_score, 4),
            "raw_anomaly_score": round(self.raw_anomaly_score, 4),
            "risk_level": self.risk_level,
            "action": self.action,
            "detected_state": self.detected_state,
            "context_deviation": round(self.context_deviation, 4),
            "combined_score": round(self.combined_score, 4),
            "processed_events": self.processed_events,
            "score_ready": self.score_ready,
            "window_event_count": self.window_event_count,
            "reasons": list(self.reasons),
            "top_deviation_features": list(self.top_deviation_features),
            "features": {key: round(value, 6) for key, value in self.features.items()},
            # Experiment framework
            "active_param_count": self.active_param_count,
            "total_param_count": self.total_param_count,
        }


@dataclass(slots=True)
class SessionState:
    context: SessionContext
    started_at: datetime
    events: list[BehaviorEvent] = field(default_factory=list)
    navigation_sequence: list[str] = field(default_factory=list)
    latest_assessment: RiskAssessment | None = None
    assessment_history: list[RiskAssessment] = field(default_factory=list)
    ended_at: datetime | None = None

    @property
    def session_id(self) -> str:
        return self.context.session_id

    @property
    def event_count(self) -> int:
        return len(self.events)
