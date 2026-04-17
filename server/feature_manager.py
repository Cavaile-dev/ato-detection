"""
feature_manager.py — Behavioral Biometrics Feature Toggle Manager
=================================================================
Reads the per-parameter toggle configuration (data/feature_toggles.json) and
exposes helpers that the rest of the pipeline uses to decide which features to
collect, extract, and include in scoring.

TOGGLE KEYS → MODEL COLUMNS MAPPING
-------------------------------------
Each high-level toggle (e.g. "mouse_velocity") may map onto one or more of the
10 FEATURE_COLUMNS used by the IsolationForest model.  When a toggle is disabled
its model columns are zero-filled during feature extraction (neutral inference
input — does not fabricate signal) and the toggle name is recorded in the
experiment log so ablation studies can post-process results correctly.

HONORING FEATURE REMOVALS
--------------------------
Context features (device_fingerprint, ip_geo, login_time) are passed to the
risk engine which checks is_context_enabled() before computing deviations.
Model features (keystroke, mouse, navigation) are zero-filled in
feature_extraction.py when their toggle is off.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from server.config import FEATURE_TOGGLE_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ALL 22 PARAMETER NAMES (canonical order)
# ---------------------------------------------------------------------------
ALL_PARAMS: list[str] = [
    # Keystroke dynamics
    "dwell_time",         # time a key is physically held down (key hold duration)
    "flight_time",        # gap between consecutive keystrokes (inter-key interval)
    "typing_consistency", # variance in typing speed — regularity of rhythm
    "error_rate",         # frequency of backspace / correction keys

    # Mouse / pointer dynamics
    "mouse_trajectory",   # linearity and curvature of mouse paths
    "mouse_velocity",     # speed of mouse movement — bots move uniformly fast/slow
    "mouse_acceleration", # acceleration changes — humans decelerate before clicks
    "click_interval",     # time between successive mouse clicks
    "scroll_behavior",    # scroll velocity and pattern

    # Touch / gesture (mobile; reserved for future collection)
    "swipe_speed",        # speed of swipe gestures on touch screens
    "gesture_pattern",    # fingerprint of common gesture sequences

    # Context / environment signals
    "device_fingerprint", # browser & device fingerprint similarity to baseline
    "ip_geo",             # IP address / geolocation vs trusted set
    "login_time",         # time-of-day compared to typical login hour
    "session_duration",   # overall session length vs baseline

    # Navigation behavior
    "navigation_pattern", # entropy and transition patterns across pages
    "time_per_page",      # dwell time per visited page
    "action_sequence",    # sequence entropy of page navigation actions

    # Cognitive / decision signals
    "decision_latency",   # pause before submitting a form / clicking a CTA
    "hesitation",         # micro-pauses and hesitation before actions

    # Browser-level signals (reserved; not yet collected)
    "tab_switching",      # frequency of tab/window switching during session
    "idle_time",          # idle gap durations within the session
    "clipboard_usage",    # copy-paste activity that bypasses typed input
]

# ---------------------------------------------------------------------------
# MAPPING: toggle key → FEATURE_COLUMNS contributed to the model vector
# (a parameter may map to zero columns if it is context-only or reserved)
# ---------------------------------------------------------------------------
PARAM_TO_COLUMNS: dict[str, list[str]] = {
    # Keystroke
    "dwell_time":          ["key_hold_time_mean", "key_hold_time_variance", "key_hold_time_std"],
    "flight_time":         ["typing_speed_mean", "typing_speed_std"],       # inter-key gap → speed
    "typing_consistency":  ["typing_speed_variance"],
    "error_rate":          ["error_rate"],                          # added model column

    # Mouse
    "mouse_trajectory":    ["mouse_velocity_mean", "mouse_velocity_std"],     # linearity proxy
    "mouse_velocity":      ["mouse_velocity_mean", "mouse_velocity_variance", "mouse_velocity_std"],
    "mouse_acceleration":  ["mouse_acceleration_variance"],
    "click_interval":      ["click_interval_mean", "click_interval_std"],
    "scroll_behavior":     [],                          # collected, no model column yet

    # Touch (mobile reserved)
    "swipe_speed":         [],
    "gesture_pattern":     [],

    # Context — handled by risk_engine, not model columns
    "device_fingerprint":  [],
    "ip_geo":              [],
    "login_time":          [],
    "session_duration":    [],

    # Navigation
    "navigation_pattern":  ["navigation_entropy", "page_transition_pattern"],
    "time_per_page":       ["dwell_time_per_page_mean", "dwell_time_per_page_variance"],
    "action_sequence":     ["navigation_entropy"],

    # Cognitive
    "decision_latency":    ["decision_latency_mean", "decision_latency_variance"],
    "hesitation":          ["hesitation_mean", "hesitation_variance"],

    # Browser (reserved)
    "tab_switching":       ["tab_switch_count"],
    "idle_time":           ["idle_mean"],
    "clipboard_usage":     ["clipboard_count"],
}

# Context-only parameters — checked by risk_engine, NOT zero-filled in features
CONTEXT_PARAMS: frozenset[str] = frozenset({
    "device_fingerprint",
    "ip_geo",
    "login_time",
    "session_duration",
})

# Default — all params enabled (preserves original system behaviour)
_DEFAULT_TOGGLES: dict[str, bool] = {param: True for param in ALL_PARAMS}

# ---------------------------------------------------------------------------
# GROUP CONFIGURATIONS
# ---------------------------------------------------------------------------
GROUP_TO_PARAMS: dict[str, list[str]] = {
    "use_keystroke": [
        "dwell_time", "flight_time", "typing_consistency", "error_rate"
    ],
    "use_mouse": [
        "mouse_trajectory", "mouse_velocity", "mouse_acceleration", "click_interval", "scroll_behavior"
    ],
    "use_touch": [
        "swipe_speed", "gesture_pattern"
    ],
    "use_context": [
        "device_fingerprint", "ip_geo", "login_time", "session_duration"
    ],
    "use_session": [
        "navigation_pattern", "time_per_page", "action_sequence"
    ],
    "use_cognitive": [
        "decision_latency", "hesitation"
    ],
    "use_browser": [
        "tab_switching", "idle_time", "clipboard_usage"
    ]
}

def selectFeatures(allFeatures: dict[str, Any], config: dict[str, bool]) -> dict[str, Any]:
    """
    Requested Group Selection utility:
    Filters an existing dictionary of extracted features down to ONLY the
    groups specified in `config`. (e.g. `{"use_keystroke": True}`)
    NOTE: In this framework, we also zero-fill or block at the point of 
    extraction via feature_manager. This utility serves as a post-filter
    or group-to-feature mask.
    """
    allowed_params = set()
    for group_key, is_enabled in config.items():
        if is_enabled and group_key in GROUP_TO_PARAMS:
            allowed_params.update(GROUP_TO_PARAMS[group_key])
            
    # Map allowed internal params back to their derived Model Columns
    allowed_columns = set()
    for param in allowed_params:
        allowed_columns.update(PARAM_TO_COLUMNS.get(param, []))
        
    return {k: v for k, v in allFeatures.items() if k in allowed_columns}


# ---------------------------------------------------------------------------
# FeatureManager
# ---------------------------------------------------------------------------

class FeatureManager:
    """
    Thread-safe singleton that owns the current toggle configuration.

    Usage::

        from server.feature_manager import feature_manager

        if feature_manager.is_enabled("mouse_velocity"):
            ...

        active = feature_manager.active_params()
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._toggles: dict[str, bool] = dict(_DEFAULT_TOGGLES)
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self, param: str) -> bool:
        """Return True if the named parameter is currently active."""
        with self._lock:
            return bool(self._toggles.get(param, False))

    def is_context_enabled(self, param: str) -> bool:
        """Convenience: is this context parameter (ip_geo, device_fingerprint, …) on?"""
        return self.is_enabled(param)

    def active_params(self) -> list[str]:
        """Ordered list of currently-enabled parameter names."""
        with self._lock:
            return [p for p in ALL_PARAMS if self._toggles.get(p, False)]

    def disabled_params(self) -> list[str]:
        """Ordered list of currently-disabled parameter names."""
        with self._lock:
            return [p for p in ALL_PARAMS if not self._toggles.get(p, False)]

    def active_model_columns(self) -> list[str]:
        """
        Flat, deduplicated list of FEATURE_COLUMNS that at least one active
        parameter contributes to.  Used for scoring analysis.
        """
        seen: set[str] = set()
        result: list[str] = []
        for param in self.active_params():
            for col in PARAM_TO_COLUMNS.get(param, []):
                if col not in seen:
                    seen.add(col)
                    result.append(col)
        return result

    def get_all_toggles(self) -> dict[str, bool]:
        """Snapshot of the full toggle map."""
        with self._lock:
            return dict(self._toggles)

    def update_toggles(self, updates: dict[str, Any]) -> dict[str, str]:
        """
        Apply partial or full toggle updates and persist to disk.

        Returns a dict of validation errors (empty if all OK).
        Toggling affects the *next* assessment batch — no restart required.
        """
        errors: dict[str, str] = {}
        cleaned: dict[str, bool] = {}

        for key, value in updates.items():
            if key.startswith("_"):               # skip comment/version fields
                continue
            if key not in ALL_PARAMS:
                errors[key] = "unknown parameter"
                continue
            if not isinstance(value, bool):
                errors[key] = f"expected bool, got {type(value).__name__}"
                continue
            cleaned[key] = value

        if not errors:
            with self._lock:
                self._toggles.update(cleaned)
            self._save_to_disk()
            logger.info("Feature toggles updated: %s", cleaned)

        return errors

    def update_from_group_config(self, group_config: dict[str, bool]) -> dict[str, str]:
        """
        Translates a group config mask (e.g. `{"use_keystroke": true, "use_mouse": false}`)
        into granular toggles and applies them. Leftover groups are implicitly false
        if at least one group config is supplied.
        """
        granular_updates: dict[str, bool] = {}
        for group_key, params in GROUP_TO_PARAMS.items():
            is_active = bool(group_config.get(group_key, False))
            for param in params:
                granular_updates[param] = is_active
        return self.update_toggles(granular_updates)

    def reload_from_disk(self) -> None:
        """Force a reload from the JSON file (e.g. after external edit)."""
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        if not FEATURE_TOGGLE_PATH.exists():
            logger.info("No feature_toggles.json found — using defaults (all ON)")
            return
        try:
            raw: dict[str, Any] = json.loads(FEATURE_TOGGLE_PATH.read_text(encoding="utf-8"))
            loaded: dict[str, bool] = {}
            for param in ALL_PARAMS:
                if param in raw:
                    loaded[param] = bool(raw[param])
            with self._lock:
                self._toggles.update(loaded)
            active_count = sum(1 for v in self._toggles.values() if v)
            logger.info("Feature toggles loaded: %d/%d active", active_count, len(ALL_PARAMS))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load feature_toggles.json: %s — keeping defaults", exc)

    def _save_to_disk(self) -> None:
        try:
            with self._lock:
                payload: dict[str, Any] = {
                    "_comment": "Per-parameter behavioral biometrics feature toggles.",
                    "_version": 1,
                }
                payload.update(self._toggles)
            FEATURE_TOGGLE_PATH.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not save feature_toggles.json: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
feature_manager = FeatureManager()
