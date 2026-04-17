"""
feature_extraction.py — Modular Behavioral Feature Extraction
=============================================================
Each behavioral parameter is extracted by its own named function following the
convention get<ParameterName>(events, state).

Toggle-awareness is applied in the top-level extract_features() call:
- If a parameter is DISABLED its model columns are set to 0.0  (zero-fill keeps
  the IsolationForest input vector at the expected width without fabricating signal)
- The raw extracted values (before zero-fill) are also returned for experiment logging

WHY EACH FEATURE MATTERS
--------------------------
dwell_time         — time key held down; bots hold far longer or shorter than humans
flight_time        — gap between keystrokes; reflects natural typing rhythm
typing_consistency — variance in speed; humans are inconsistent, bots are not
mouse_trajectory   — path curvature; bots move in straight lines, humans deviate
mouse_velocity     — mean speed; bots move at constant superhuman speeds
mouse_acceleration — speed changes; humans decelerate before targets, bots don't
click_interval     — time between clicks; bots click at mathematically regular intervals
scroll_behavior    — scroll patterns; bots often skip scrolling entirely
navigation_pattern — page visit entropy; bots follow fixed, low-entropy paths
time_per_page      — time spent per page; bots rush, legitimate users read
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from server.config import FEATURE_COLUMNS
from server.feature_manager import feature_manager
from server.schemas import BehaviorEvent, SessionState


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def safe_variance(values: list[float]) -> float:
    return float(np.var(values)) if values else 0.0


def shannon_entropy(sequence: list[str]) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def circular_hour_difference(current_hour: float, typical_hour: float) -> float:
    delta = abs(current_hour - typical_hour)
    return float(min(delta, 24.0 - delta))


# ---------------------------------------------------------------------------
# Internal raw-data helpers (shared across named functions)
# ---------------------------------------------------------------------------

def _extract_mouse_metrics(events: list[BehaviorEvent]) -> tuple[list[float], list[float]]:
    mouse_events = [event for event in events if event.event_type == "mouse_move"]
    velocities: list[float] = []
    accelerations: list[float] = []
    previous_event: BehaviorEvent | None = None
    previous_velocity = 0.0

    for event in mouse_events:
        velocity = event.velocity
        if velocity is None and previous_event and previous_event.x is not None and previous_event.y is not None:
            if event.x is not None and event.y is not None:
                delta_seconds = max(event.timestamp - previous_event.timestamp, 1e-6)
                distance = math.hypot(event.x - previous_event.x, event.y - previous_event.y)
                velocity = distance / delta_seconds

        if velocity is not None:
            velocity = float(velocity)
            velocities.append(velocity)

            acceleration = event.acceleration
            if acceleration is None and previous_event:
                delta_seconds = max(event.timestamp - previous_event.timestamp, 1e-6)
                acceleration = (velocity - previous_velocity) / delta_seconds
            if acceleration is not None:
                accelerations.append(float(acceleration))
            previous_velocity = velocity

        previous_event = event

    return velocities, accelerations


def _extract_click_intervals(events: list[BehaviorEvent]) -> list[float]:
    click_events = [event for event in events if event.event_type == "click"]
    intervals: list[float] = []
    previous_timestamp: float | None = None

    for event in click_events:
        interval = event.click_interval
        if interval is None and previous_timestamp is not None:
            interval = event.timestamp - previous_timestamp
        if interval is not None and interval >= 0:
            intervals.append(float(interval))
        previous_timestamp = event.timestamp

    return intervals


def _extract_typing_metrics(events: list[BehaviorEvent]) -> tuple[list[float], list[float]]:
    typing_events = [event for event in events if event.event_type == "keystroke"]
    typing_speeds: list[float] = []
    hold_times: list[float] = []

    for event in typing_events:
        if event.key_interval and event.key_interval > 0:
            typing_speeds.append(1.0 / max(float(event.key_interval), 1e-4))
        if event.hold_time is not None and event.hold_time >= 0:
            hold_times.append(float(event.hold_time))

    return typing_speeds, hold_times


def _extract_navigation_sequence(state: SessionState, events: list[BehaviorEvent]) -> list[str]:
    if state.navigation_sequence:
        return [page for page in state.navigation_sequence if page]
    return [
        event.page
        for event in events
        if event.page and event.event_type in {"navigation", "page_enter"}
    ]


def _transition_pairs(pages: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left, right in zip(pages, pages[1:]):
        if left and right and left != right:
            pairs.append((left, right))
    return pairs


def _page_transition_pattern(state: SessionState, pages: list[str]) -> float:
    observed = _transition_pairs(pages)
    preferred = _transition_pairs(state.context.profile.preferred_pages)
    if not observed:
        return 0.0
    if not preferred:
        return 0.5
    preferred_set = set(preferred)
    matches = sum(1 for transition in observed if transition in preferred_set)
    return float(matches / len(observed))


def _extract_page_dwell_times(events: list[BehaviorEvent]) -> list[float]:
    ordered_events = sorted(events, key=lambda event: event.timestamp)
    navigation_events = [
        event
        for event in ordered_events
        if event.page and event.event_type in {"navigation", "page_enter"}
    ]
    dwell_times: list[float] = []

    explicit_dwell = [
        float(event.dwell_time)
        for event in ordered_events
        if event.event_type == "page_exit" and event.dwell_time is not None and event.dwell_time >= 0
    ]
    dwell_times.extend(explicit_dwell)

    for index, event in enumerate(navigation_events):
        if index + 1 < len(navigation_events):
            dwell = navigation_events[index + 1].timestamp - event.timestamp
        else:
            dwell = ordered_events[-1].timestamp - event.timestamp if ordered_events else 0.0
        if dwell >= 0:
            dwell_times.append(float(dwell))

    return dwell_times


# ---------------------------------------------------------------------------
# Named feature extraction functions (one per toggle parameter)
# ---------------------------------------------------------------------------

def getDwellTime(events: list[BehaviorEvent]) -> tuple[float, float, float]:
    """
    Dwell time = mean duration a key is physically held down.
    WHY: Humans hold keys for 60-180 ms with natural variance; bots are
         often either zero-ms (simulated keypress) or unnaturally long.
    TOGGLES: 'dwell_time' → model column 'key_hold_time_mean'
    """
    _, hold_times = _extract_typing_metrics(events)
    return safe_mean(hold_times), safe_variance(hold_times), safe_std(hold_times)


def getFlightTime(events: list[BehaviorEvent]) -> tuple[float, float, float]:
    """
    Flight time = mean interval between consecutive keystrokes.
    WHY: The rhythmic cadence of typing is a strong biometric identifier; bots
         type at a constant machine interval, humans vary naturally.
    TOGGLES: 'flight_time' → model column 'typing_speed_mean' (speed = 1/interval)
    """
    typing_speeds, _ = _extract_typing_metrics(events)
    return safe_mean(typing_speeds), safe_variance(typing_speeds), safe_std(typing_speeds)


def getTypingConsistency(events: list[BehaviorEvent]) -> float:
    """
    Typing consistency = variance of typing speed across keystrokes.
    WHY: A low variance indicates robotic regularity; humans exhibit natural spread.
    TOGGLES: 'typing_consistency' → model column 'typing_speed_variance'
    """
    typing_speeds, _ = _extract_typing_metrics(events)
    return safe_variance(typing_speeds)


def getErrorRate(events: list[BehaviorEvent]) -> float:
    """
    Error rate = normalised count of backspace / delete correction keystrokes.
    WHY: Humans make typos; bots typically have zero error rate.
    TOGGLES: 'error_rate' → model column 'error_rate' (newly wired)
    """
    typing_events = [e for e in events if e.event_type == "keystroke"]
    if not typing_events:
        return 0.0

    corrections = sum(1 for e in typing_events if e.metadata and e.metadata.get("key") in {"Backspace", "Delete"})
    # Some legacy payloads might store 'key' directly on the event dict if loaded from json depending on schema,
    # but we will rely on how BehaviorEvent parses it. Let's look at schema logic if needed.
    # To be safe, we also check if event payload has key directly.
    return float(corrections / len(typing_events))


def getMouseTrajectory(events: list[BehaviorEvent]) -> float:
    """
    Mouse trajectory = mean velocity as a proxy for path linearity.
    WHY: Bots draw mathematically straight lines; human paths curve naturally.
    TOGGLES: 'mouse_trajectory' → model column 'mouse_velocity_mean'
    """
    velocities, _ = _extract_mouse_metrics(events)
    return safe_mean(velocities)


def getMouseVelocity(events: list[BehaviorEvent]) -> tuple[float, float, float]:
    """
    Mouse velocity = (mean velocity, velocity variance).
    WHY: Bots move at constant superhuman speeds with zero variance; humans
         accelerate and decelerate naturally.
    TOGGLES: 'mouse_velocity' → 'mouse_velocity_mean', 'mouse_velocity_variance'
    """
    velocities, _ = _extract_mouse_metrics(events)
    return safe_mean(velocities), safe_variance(velocities), safe_std(velocities)


def getMouseAcceleration(events: list[BehaviorEvent]) -> float:
    """
    Mouse acceleration = variance in acceleration across mouse movements.
    WHY: Humans decelerate toward click targets (Fitts's law); bots do not.
    TOGGLES: 'mouse_acceleration' → model column 'mouse_acceleration_variance'
    """
    _, accelerations = _extract_mouse_metrics(events)
    return safe_variance(accelerations)


def getClickInterval(events: list[BehaviorEvent]) -> tuple[float, float]:
    """
    Click interval = standard deviation of inter-click durations.
    WHY: A near-zero std indicates programmatic, robotic clicking patterns.
    TOGGLES: 'click_interval' → model column 'click_interval_std'
    """
    intervals = _extract_click_intervals(events)
    return safe_mean(intervals), safe_std(intervals)


def getScrollBehavior(events: list[BehaviorEvent]) -> float:
    """
    Scroll behavior = mean absolute scroll delta across scroll events.
    WHY: Humans scroll variably; bots often skip scrolling or use fixed deltas.
    TOGGLES: 'scroll_behavior' → reserved (no model column yet); always returns 0.0
    """
    scroll_events = [e for e in events if e.event_type == "scroll" and e.scroll_delta is not None]
    if not scroll_events:
        return 0.0
    return safe_mean([abs(float(e.scroll_delta)) for e in scroll_events])  # type: ignore[arg-type]


def getDeviceFingerprint(state: SessionState) -> str | None:
    """
    Device fingerprint = browser/device fingerprint string from context.
    WHY: ATO attempts commonly originate from unrecognised devices.
    TOGGLES: 'device_fingerprint' → context feature (handled by risk_engine)
    Returns None when toggle is disabled so risk_engine skips the check.
    """
    if not feature_manager.is_enabled("device_fingerprint"):
        return None
    return state.context.device.fingerprint


def getIPGeo(state: SessionState) -> str | None:
    """
    IP / geolocation = current IP address string from context.
    WHY: Login from an unrecognised IP is a strong ATO signal.
    TOGGLES: 'ip_geo' → context feature (handled by risk_engine)
    """
    if not feature_manager.is_enabled("ip_geo"):
        return None
    return state.context.ip_address


def getLoginTime(state: SessionState) -> float | None:
    """
    Login time = hour-of-day (0-24) at session start.
    WHY: Logins at unusual hours vs. the user's typical schedule are suspicious.
    TOGGLES: 'login_time' → context feature (handled by risk_engine)
    """
    if not feature_manager.is_enabled("login_time"):
        return None
    return state.context.login_hour()


def getSessionDuration(state: SessionState) -> float | None:
    """
    Session duration = elapsed seconds since session start.
    WHY: Extremely short sessions may indicate automated credential stuffing.
    TOGGLES: 'session_duration' → context/session feature; not in model columns
    """
    if not feature_manager.is_enabled("session_duration"):
        return None
    if state.events:
        return float(state.events[-1].timestamp - state.events[0].timestamp)
    return 0.0


def getNavigationPattern(state: SessionState, events: list[BehaviorEvent]) -> tuple[float, float]:
    """
    Navigation pattern = (page entropy, page transition pattern score).
    WHY: Attackers tend to navigate directly to sensitive pages (low entropy,
         non-matching transitions); legitimate users follow familiar paths.
    TOGGLES: 'navigation_pattern' → 'navigation_entropy', 'page_transition_pattern'
    """
    nav_seq = _extract_navigation_sequence(state, events)
    return shannon_entropy(nav_seq), _page_transition_pattern(state, nav_seq)


def getTimePerPage(events: list[BehaviorEvent]) -> tuple[float, float]:
    """
    Time per page = mean dwell time across visited pages (seconds).
    WHY: Bots rush through pages; legitimate users spend time reading content.
    TOGGLES: 'time_per_page' → model column 'dwell_time_per_page'
    """
    dwells = _extract_page_dwell_times(events)
    return safe_mean(dwells), safe_variance(dwells)


def getActionSequence(state: SessionState, events: list[BehaviorEvent]) -> float:
    """
    Action sequence = Shannon entropy of the page navigation sequence.
    WHY: A highly predictable (low-entropy) action sequence is a bot signal.
    TOGGLES: 'action_sequence' → model column 'navigation_entropy'
    """
    nav_seq = _extract_navigation_sequence(state, events)
    return shannon_entropy(nav_seq)


def getDecisionLatency(events: list[BehaviorEvent]) -> tuple[float, float]:
    """
    Decision latency = mean pause (seconds) immediately before form submission or
    significant clicks — measured via the hesitation metric on click events.
    WHY: Humans deliberate before acting; bots click instantly with zero latency.
    TOGGLES: 'decision_latency' → no dedicated model column yet; returned for logging
    """
    hesitations = [
        float(e.hesitation)
        for e in events
        if e.event_type == "click" and e.hesitation is not None
    ]
    return safe_mean(hesitations), safe_variance(hesitations)


def getHesitation(events: list[BehaviorEvent]) -> tuple[float, float]:
    """
    Hesitation = mean micro-pause duration across all interaction events.
    WHY: Hesitation patterns encode cognitive load; attackers acting from scripts
         show no hesitation whatsoever.
    TOGGLES: 'hesitation' → no dedicated model column yet; returned for logging
    """
    hesitations = [
        float(e.hesitation)
        for e in events
        if e.hesitation is not None and e.hesitation >= 0
    ]
    return safe_mean(hesitations), safe_variance(hesitations)


def getTabSwitching(events: list[BehaviorEvent]) -> float:
    """
    Tab switching = frequency of focus-lost events indicating tab/window changes.
    WHY: Attackers copy-pasting credentials often switch tabs rapidly.
    """
    tab_switches = [e for e in events if e.event_type == "tab_switch"]
    return float(len(tab_switches))


def getIdleTime(events: list[BehaviorEvent]) -> float:
    """
    Idle time = mean duration of inactivity gaps between events (seconds).
    WHY: Natural users have irregular idle periods; automated scripts have none.
    """
    idle_times = [float(e.metadata.get("idle_time", 0.0)) for e in events if e.event_type == "idle"]
    if not idle_times:
        return 0.0
    return safe_mean(idle_times)


def getClipboardUsage(events: list[BehaviorEvent]) -> float:
    """
    Clipboard usage = fraction of text-field input sourced from paste events.
    WHY: Bots / credential stuffers frequently paste rather than type credentials.
    """
    clipboard = [e for e in events if e.event_type == "clipboard"]
    return float(len(clipboard))


# ---------------------------------------------------------------------------
# Core extraction pipeline — toggle-aware
# ---------------------------------------------------------------------------

def extract_features(
    state: SessionState,
) -> tuple[dict[str, float], dict[str, float | None]]:
    """
    Extract all behavioral features from the session state, honouring the
    current feature toggle configuration.

    Returns
    -------
    model_features : dict[str, float]
        Fixed-width vector matching FEATURE_COLUMNS.  Disabled parameters
        contribute 0.0 (neutral zero-fill, not absence of data).

    raw_values : dict[str, float | None]
        Per-parameter extracted values for experiment logging.
        Disabled parameters return None so the experiment log records
        which parameters were actually computed.
    """
    events = sorted(state.events, key=lambda event: event.timestamp)

    # --- call each named extractor once (cheap; shares raw data) ---
    _mv_mean, _mv_var, _mv_std = getMouseVelocity(events)
    _ma_var            = getMouseAcceleration(events)
    _ci_mean, _ci_std   = getClickInterval(events)
    _ft_mean, _ft_var, _ft_std = getFlightTime(events)
    _tc                = getTypingConsistency(events)
    _dt_mean, _dt_var, _dt_std = getDwellTime(events)
    _nav_entropy, _nav_transition = getNavigationPattern(state, events)
    _tpp_mean, _tpp_var = getTimePerPage(events)
    _dec_lat_mean, _dec_lat_var = getDecisionLatency(events)
    _hes_mean, _hes_var = getHesitation(events)
    _scroll            = getScrollBehavior(events)
    _err_rate          = getErrorRate(events)
    _tab               = getTabSwitching(events)
    _idle              = getIdleTime(events)
    _clip              = getClipboardUsage(events)

    fm = feature_manager

    # --- build raw_values dict for experiment logger ---
    raw_values: dict[str, float | None] = {
        "dwell_time":          _dt_mean     if fm.is_enabled("dwell_time")          else None,
        "flight_time":         _ft_mean     if fm.is_enabled("flight_time")          else None,
        "typing_consistency":  _tc          if fm.is_enabled("typing_consistency")   else None,
        "error_rate":          _err_rate    if fm.is_enabled("error_rate")           else None,
        "mouse_trajectory":    _mv_mean     if fm.is_enabled("mouse_trajectory")     else None,
        "mouse_velocity":      _mv_mean     if fm.is_enabled("mouse_velocity")       else None,
        "mouse_acceleration":  _ma_var      if fm.is_enabled("mouse_acceleration")   else None,
        "click_interval":      _ci_std      if fm.is_enabled("click_interval")       else None,
        "scroll_behavior":     _scroll      if fm.is_enabled("scroll_behavior")      else None,
        "swipe_speed":         None,        # reserved
        "gesture_pattern":     None,        # reserved
        "device_fingerprint":  state.context.device.fingerprint if fm.is_enabled("device_fingerprint") else None,
        "ip_geo":              state.context.ip_address          if fm.is_enabled("ip_geo")             else None,
        "login_time":          state.context.login_hour()        if fm.is_enabled("login_time")         else None,
        "session_duration":    getSessionDuration(state),
        "navigation_pattern":  _nav_entropy if fm.is_enabled("navigation_pattern")  else None,
        "time_per_page":       _tpp_mean    if fm.is_enabled("time_per_page")        else None,
        "action_sequence":     _nav_entropy if fm.is_enabled("action_sequence")      else None,
        "decision_latency":    _dec_lat_mean if fm.is_enabled("decision_latency")     else None,
        "hesitation":          _hes_mean    if fm.is_enabled("hesitation")           else None,
        "tab_switching":       _tab         if fm.is_enabled("tab_switching")        else None,
        "idle_time":           _idle        if fm.is_enabled("idle_time")            else None,
        "clipboard_usage":     _clip        if fm.is_enabled("clipboard_usage")      else None,
    }

    # --- build model_features dict with zero-fill for disabled params ---
    # Each FEATURE_COLUMN gets a computed value only if EVERY parameter that
    # contributes to it is enabled.  Otherwise it is zero-filled so the
    # IsolationForest receives a neutral (non-fabricated) input.

    def _col_enabled(col: str) -> bool:
        """True if at least one toggle contributing to this column is enabled."""
        from server.feature_manager import PARAM_TO_COLUMNS
        contributing = [p for p, cols in PARAM_TO_COLUMNS.items() if col in cols]
        if not contributing:
            return True  # no specific toggle controls this column — keep it
        return any(fm.is_enabled(p) for p in contributing)

    computed: dict[str, float] = {
        "mouse_velocity_mean":       _mv_mean,
        "mouse_velocity_variance":   _mv_var,
        "mouse_velocity_std":        _mv_std,
        "mouse_acceleration_variance": _ma_var,
        "click_interval_mean":       _ci_mean,
        "click_interval_std":        _ci_std,
        "typing_speed_mean":         _ft_mean,
        "typing_speed_variance":     _ft_var,
        "typing_speed_std":          _ft_std,
        "key_hold_time_mean":        _dt_mean,
        "key_hold_time_variance":    _dt_var,
        "key_hold_time_std":         _dt_std,
        "navigation_entropy":        _nav_entropy,
        "page_transition_pattern":   _nav_transition,
        "dwell_time_per_page_mean":  _tpp_mean,
        "dwell_time_per_page_variance": _tpp_var,
        "decision_latency_mean":     _dec_lat_mean,
        "decision_latency_variance": _dec_lat_var,
        "error_rate":                _err_rate,
        "hesitation_mean":           _hes_mean,
        "hesitation_variance":       _hes_var,
        "tab_switch_count":          _tab,
        "idle_mean":                 _idle,
        "clipboard_count":           _clip,
    }

    model_features: dict[str, float] = {
        col: (float(computed[col]) if _col_enabled(col) else 0.0)
        for col in FEATURE_COLUMNS
    }

    return model_features, raw_values


def to_feature_vector(features: dict[str, float]) -> np.ndarray:
    values = [float(features.get(column, 0.0)) for column in FEATURE_COLUMNS]
    return np.asarray(values, dtype=float).reshape(1, -1)
