from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from server.config import DATA_DIR, FEATURE_COLUMNS
STREAM_EVENTS_PATH = DATA_DIR / "stream_events.jsonl"
REAL_FEATURES_PATH = DATA_DIR / "real_features.csv"
@dataclass
class SessionSummary:
    session_id: str
    user_id: str
    event_count: int
    first_timestamp: float
    last_timestamp: float
def get_session_summaries() -> list[SessionSummary]:
    """Read all sessions from stream_events.jsonl"""
    if not STREAM_EVENTS_PATH.exists():
        return []
    
    sessions: dict[str, dict] = {}
    
    for line in STREAM_EVENTS_PATH.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            sid = record.get("session_id", "unknown")
            uid = record.get("user_id", "unknown")
            events = record.get("events", [])
            
            if sid not in sessions:
                sessions[sid] = {
                    "user_id": uid,
                    "events": [],
                }
            
            sessions[sid]["events"].extend(events)
        except json.JSONDecodeError:
            continue
    
    result = []
    for sid, data in sessions.items():
        events = data["events"]
        if not events:
            continue
        
        timestamps = [e.get("timestamp", 0) for e in events]
        
        result.append(SessionSummary(
            session_id=sid,
            user_id=data["user_id"],
            event_count=len(events),
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps),
        ))
    
    return result
def extract_features_from_session(session_id: str) -> dict | None:
    """Extract 10 features from a single session"""
    from server.schemas import BehaviorEvent, SessionContext, SessionState, UserProfile, DeviceInfo
    from server.feature_extraction import extract_features
    
    if not STREAM_EVENTS_PATH.exists():
        return None
    
    events = []
    user_id = "unknown"
    ip_address = "real-client"
    
    for line in STREAM_EVENTS_PATH.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("session_id") == session_id:
                events.extend(record.get("events", []))
                user_id = record.get("user_id", user_id)
                ip_address = record.get("ip_address", ip_address)
        except json.JSONDecodeError:
            continue
    
    if len(events) < 20:
        return None
    
    parsed_events = [BehaviorEvent.from_payload(e) for e in events]
    parsed_events.sort(key=lambda e: e.timestamp)
    
    first_event = parsed_events[0]
    
    default_profile = UserProfile(
        typical_login_hour=first_event.timestamp % 24,
        trusted_ips=[ip_address],
        trusted_devices=["real-device"],
        preferred_pages=["/home", "/profile", "/settings", "/home"],
    )
    
    default_device = DeviceInfo(
        device_id="real-device",
        device_type="desktop",
        os="real-os",
        browser="real-browser",
    )
    
    context = SessionContext(
        session_id=session_id,
        user_id=user_id,
        ip_address=ip_address,
        device=default_device,
        login_time=datetime.fromtimestamp(first_event.timestamp, tz=timezone.utc),
        profile=default_profile,
    )
    
    state = SessionState(
        context=context,
        started_at=context.login_time,
        events=parsed_events,
    )
    
    features, _ = extract_features(state)
    return features
def collect_real_training_data(min_events: int = 20, min_sessions: int = 10) -> pd.DataFrame | None:
    """
    Convert all collected sessions to training-ready DataFrame.
    
    All sessions are labeled as 0 (normal) - Isolation Forest is unsupervised,
    so labels are NOT used during training.
    
    Returns:
        DataFrame with session_id, user_id, label, and 10 feature columns
        or None if not enough sessions.
    """
    summaries = get_session_summaries()
    
    if len(summaries) < min_sessions:
        print(f"[RealDataCollector] Only {len(summaries)} sessions found, need at least {min_sessions}")
        return None
    
    rows = []
    skipped = 0
    
    for summary in summaries:
        if summary.event_count < min_events:
            skipped += 1
            continue
        
        features = extract_features_from_session(summary.session_id)
        if features is None:
            skipped += 1
            continue
        
        row = {
            "session_id": summary.session_id,
            "user_id": summary.user_id,
            "label": 0,  # All normal - no labeling needed
            "scenario": "real_normal",
            **features,
        }
        rows.append(row)
    
    if not rows:
        print(f"[RealDataCollector] No valid sessions (skipped {skipped})")
        return None
    
    df = pd.DataFrame(rows)
    
    cols = ["session_id", "user_id", "label", "scenario"] + FEATURE_COLUMNS
    df = df[[c for c in cols if c in df.columns]]
    
    df.to_csv(REAL_FEATURES_PATH, index=False)
    
    print(f"[RealDataCollector] Collected {len(df)} sessions (skipped {skipped})")
    return df
def get_stats() -> dict:
    """Get collection statistics"""
    summaries = get_session_summaries()
    total_events = sum(s.event_count for s in summaries)
    
    return {
        "total_sessions": len(summaries),
        "total_events": total_events,
        "avg_events_per_session": total_events / len(summaries) if summaries else 0,
    }
def main():
    print("=== Real Data Collector ===")
    
    stats = get_stats()
    print(f"Sessions in stream_events.jsonl: {stats['total_sessions']}")
    print(f"Total events: {stats['total_events']}")
    print(f"Avg events/session: {stats['avg_events_per_session']:.1f}")
    
    if stats['total_sessions'] > 0:
        print("\nConverting to training data...")
        df = collect_real_training_data()
        if df is not None:
            print(f"Success! Saved to {REAL_FEATURES_PATH}")
        else:
            print("Failed - not enough valid sessions")
if __name__ == "__main__":
    main()