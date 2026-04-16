"""
risk_engine.py — Toggle-Aware Risk Scoring
==========================================
Extends the original scoring logic to honour context-feature toggles
(device_fingerprint, ip_geo, login_time) from the feature manager.

When a context feature is disabled via its toggle, the corresponding deviation
sub-score is zeroed out so the experiment captures a clean ablation result.

The active_param_count parameter is surfaced in the final RiskAssessment so
experiment logs can normalise results across different feature configurations.
"""
from __future__ import annotations

from server.feature_extraction import circular_hour_difference
from server.feature_manager import ALL_PARAMS, feature_manager
from server.schemas import RiskAssessment, SessionState


def calculate_context_deviation(state: SessionState) -> tuple[float, list[str]]:
    """
    Compute the context deviation score from IP, device, and login-time signals.

    Each sub-score is gated by its corresponding feature toggle:
      ip_geo            → IP address check
      device_fingerprint → device fingerprint check
      login_time        → time-of-day check

    When a context toggle is OFF its contribution is zeroed (treated as no
    deviation observed) so the experiment isolates only the enabled signals.
    """
    profile = state.context.profile
    reasons: list[str] = []

    # --- IP geolocation check (toggle: ip_geo) ---
    ip_score = 0.0
    if feature_manager.is_enabled("ip_geo"):
        if profile.trusted_ips and state.context.ip_address not in profile.trusted_ips:
            ip_score = 1.0
            reasons.append("IP address differs from trusted profile")

    # --- Device fingerprint check (toggle: device_fingerprint) ---
    device_score = 0.0
    if feature_manager.is_enabled("device_fingerprint"):
        if profile.trusted_devices and state.context.device.fingerprint not in profile.trusted_devices:
            device_score = 1.0
            reasons.append("Device fingerprint is unfamiliar")

    # --- Login time check (toggle: login_time) ---
    time_score = 0.0
    if feature_manager.is_enabled("login_time"):
        time_delta = circular_hour_difference(state.context.login_hour(), profile.typical_login_hour)
        time_score = min(time_delta / 5.0, 1.0)
        if time_score >= 0.35:
            reasons.append("Login time deviates from typical behavior")

    context_deviation = min((0.35 * ip_score) + (0.35 * device_score) + (0.30 * time_score), 1.0)
    return context_deviation, reasons


def _feature_reason_map(feature_name: str) -> str:
    if feature_name.startswith("mouse_"):
        return "mouse movement unnatural"
    if feature_name.startswith("typing_") or feature_name == "key_hold_time_mean":
        return "typing pattern deviation"
    if feature_name in {"navigation_entropy", "page_transition_pattern", "dwell_time_per_page"}:
        return "navigation anomaly"
    if feature_name == "click_interval_std":
        return "click timing anomaly"
    return "behavioral biometrics differ from baseline"


def _extract_behavioral_reasons(feature_deviations: dict[str, float], anomaly_score: float) -> tuple[list[str], list[str]]:
    if not feature_deviations or anomaly_score < 0.35:
        return [], []

    ranked = sorted(feature_deviations.items(), key=lambda item: item[1], reverse=True)
    top_features = [name for name, _ in ranked[:4]]
    reasons: list[str] = []

    for feature_name, deviation in ranked[:4]:
        if deviation < 1.15:
            continue
        reason = _feature_reason_map(feature_name)
        if reason not in reasons:
            reasons.append(reason)

    return reasons, top_features


def _detected_state(risk_level: str, score_ready: bool) -> str:
    if not score_ready:
        return "OBSERVING"
    if risk_level == "HIGH":
        return "ATTACKER-LIKE"
    if risk_level == "MEDIUM":
        return "SUSPICIOUS"
    return "NORMAL-LIKE"


def build_risk_assessment(
    state: SessionState,
    anomaly_score: float,
    raw_anomaly_score: float,
    features: dict[str, float],
    feature_deviations: dict[str, float] | None = None,
    score_ready: bool = False,
    window_event_count: int = 0,
    active_param_count: int | None = None,
) -> RiskAssessment:
    """
    Build a RiskAssessment from current model outputs and context signals.

    Parameters
    ----------
    active_param_count : int | None
        Number of currently-enabled feature toggle parameters.
        If None, defaults to the total number of ALL_PARAMS (all-enabled baseline).
        Surfaced in the assessment and experiment log for normalisation analysis.

    NOTE ON SCORING FAIRNESS WITH FEWER FEATURES
    ---------------------------------------------
    The IsolationForest was trained on all 10 model columns.  When parameters
    are disabled their columns are zero-filled (neutral signal), which means the
    model score reflects less information.  We do NOT re-weight the combined
    score formula because the zero-fill already biases toward "normal" — the
    experiment log records active_param_count so post-hoc analysis can account
    for this.  If you want per-config model retraining, that is a future
    extension outside the scope of this framework.
    """
    if active_param_count is None:
        active_param_count = len(ALL_PARAMS)

    context_deviation, reasons = calculate_context_deviation(state)
    behavioral_reasons, top_features = _extract_behavioral_reasons(feature_deviations or {}, anomaly_score)
    combined_score = ((0.72 * anomaly_score) + (0.28 * context_deviation)) if score_ready else context_deviation * 0.35

    if not score_ready:
        risk_level = "LOW" if context_deviation < 0.65 else "MEDIUM"
        action = "ALLOW_SESSION" if risk_level == "LOW" else "LOG_AND_MONITOR"
        reasons.insert(0, "warming up: collecting enough interaction events before behavioral scoring")
    elif combined_score >= 1.55 or anomaly_score >= 1.55:
        risk_level = "HIGH"
        action = "TRIGGER_MFA"
    elif combined_score >= 1.25 or anomaly_score >= 1.25:
        risk_level = "MEDIUM"
        action = "LOG_AND_MONITOR"
    else:
        risk_level = "LOW"
        action = "ALLOW_SESSION"

    for reason in behavioral_reasons:
        if reason not in reasons:
            reasons.append(reason)

    if score_ready and risk_level == "LOW" and not behavioral_reasons:
        reasons.append("behavior aligns with the learned baseline")
    elif score_ready and not behavioral_reasons and anomaly_score >= 0.40:
        reasons.append("behavioral biometrics differ from baseline")

    # Surface reduced-feature context for analysts
    total_params = len(ALL_PARAMS)
    if active_param_count < total_params:
        reasons.append(
            f"experiment mode: {active_param_count}/{total_params} features active — "
            "score reflects partial feature set"
        )

    return RiskAssessment(
        session_id=state.session_id,
        anomaly_score=float(anomaly_score),
        raw_anomaly_score=float(raw_anomaly_score),
        risk_level=risk_level,
        action=action,
        detected_state=_detected_state(risk_level, score_ready),
        context_deviation=float(context_deviation),
        combined_score=float(combined_score),
        features=features,
        reasons=reasons,
        top_deviation_features=top_features,
        processed_events=state.event_count,
        score_ready=score_ready,
        window_event_count=window_event_count,
        active_param_count=active_param_count,
        total_param_count=total_params,
    )
