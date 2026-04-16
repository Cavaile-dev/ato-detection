"""Comprehensive API validation using the SAME mock_data infrastructure
that the model was trained on, ensuring the features match exactly."""
import requests

BASE = "http://127.0.0.1:5000"


def test_with_mock_infrastructure():
    """Use the mock_data.py session builder to generate events that are
    guaranteed to match the training distribution."""
    import random
    from datetime import datetime, timezone
    from server.mock_data import (
        _build_users, _make_login_time, _build_session_builder,
        _simulate_login, _simulate_normal_pages,
        _simulate_bot_attacker, _simulate_human_attacker, _simulate_fast_attacker,
    )
    from server.feature_extraction import extract_features
    from server.schemas import SessionContext, SessionState, DeviceInfo, UserProfile

    users = _build_users()
    alice = users[0]  # alice

    # --- TEST 1: Normal user ---
    print("=" * 60)
    print("TEST 1: Normal user -> LOW risk")
    print("=" * 60)

    rng = random.Random(999)
    login_time = _make_login_time(rng, alice.typical_login_hour, attacker_type=None)
    builder = _build_session_builder(rng, login_time)
    _simulate_login(builder, alice, attacker_type=None)
    _simulate_normal_pages(builder, alice)
    events = builder.finish()

    # Check features locally first
    context = SessionContext(
        session_id="local-test",
        user_id=alice.user_id,
        ip_address=alice.trusted_ip,
        device=DeviceInfo(device_id=alice.trusted_device, os="windows", browser="chrome"),
        login_time=login_time,
        profile=UserProfile(
            typical_login_hour=alice.typical_login_hour,
            trusted_ips=[alice.trusted_ip],
            trusted_devices=[alice.trusted_device.lower()],
            preferred_pages=alice.preferred_pages,
        ),
    )
    nav_seq = [e.page for e in events if e.event_type == "navigation" and e.page]
    state = SessionState(context=context, started_at=login_time, events=events, navigation_sequence=nav_seq)
    features = extract_features(state)
    print(f"  Local feature check ({len(events)} events):")
    for k, v in features.items():
        print(f"    {k:30s} = {v:.4f}")

    # Send via API
    session = requests.post(f"{BASE}/api/v1/sessions/start", json={
        "user_id": alice.user_id,
        "ip_address": alice.trusted_ip,
        "device": {"device_id": alice.trusted_device, "os": "windows", "browser": "chrome"},
        "profile": {
            "typical_login_hour": alice.typical_login_hour,
            "trusted_ips": [alice.trusted_ip],
            "trusted_devices": [alice.trusted_device.lower()],
            "preferred_pages": alice.preferred_pages,
        },
    }).json()
    sid = session["session_id"]

    event_dicts = [e.to_dict() for e in events]
    r = requests.post(f"{BASE}/api/v1/events", json={"session_id": sid, "events": event_dicts}).json()
    print(f"\n  API result: score={r['anomaly_score']:.4f}, risk={r['risk_level']}, "
          f"state={r['detected_state']}, ready={r['score_ready']}, window={r['window_event_count']}")
    print(f"  Reasons: {r['reasons']}")
    test1_pass = r["risk_level"] == "LOW"
    print(f"  {'[PASS]' if test1_pass else '[FAIL]'}: Normal user risk = {r['risk_level']}")

    # --- TEST 2: BOT attacker ---
    print()
    print("=" * 60)
    print("TEST 2: BOT attacker -> HIGH risk")
    print("=" * 60)

    rng2 = random.Random(888)
    login_time2 = _make_login_time(rng2, alice.typical_login_hour, attacker_type="BOT")
    builder2 = _build_session_builder(rng2, login_time2)
    _simulate_login(builder2, alice, attacker_type="BOT")
    _simulate_bot_attacker(builder2, alice)
    events2 = builder2.finish()

    session2 = requests.post(f"{BASE}/api/v1/sessions/start", json={
        "user_id": alice.user_id,
        "ip_address": "198.51.100.42",
        "device": {"device_id": "bot-alice-unknown", "os": "linux", "browser": "headless"},
        "profile": {
            "typical_login_hour": alice.typical_login_hour,
            "trusted_ips": [alice.trusted_ip],
            "trusted_devices": [alice.trusted_device.lower()],
            "preferred_pages": alice.preferred_pages,
        },
    }).json()
    sid2 = session2["session_id"]

    event_dicts2 = [e.to_dict() for e in events2]
    r2 = requests.post(f"{BASE}/api/v1/events", json={"session_id": sid2, "events": event_dicts2}).json()
    print(f"  API result: score={r2['anomaly_score']:.4f}, risk={r2['risk_level']}, "
          f"state={r2['detected_state']}, ready={r2['score_ready']}, window={r2['window_event_count']}")
    print(f"  Reasons: {r2['reasons']}")
    print(f"  Top features: {r2['top_deviation_features']}")
    test2_pass = r2["risk_level"] == "HIGH"
    print(f"  {'[PASS]' if test2_pass else '[FAIL]'}: BOT attacker risk = {r2['risk_level']}")

    # --- TEST 3: HUMAN attacker ---
    print()
    print("=" * 60)
    print("TEST 3: HUMAN attacker -> HIGH risk")
    print("=" * 60)

    rng3 = random.Random(777)
    login_time3 = _make_login_time(rng3, alice.typical_login_hour, attacker_type="HUMAN")
    builder3 = _build_session_builder(rng3, login_time3)
    _simulate_login(builder3, alice, attacker_type="HUMAN")
    _simulate_human_attacker(builder3, alice)
    events3 = builder3.finish()

    session3 = requests.post(f"{BASE}/api/v1/sessions/start", json={
        "user_id": alice.user_id,
        "ip_address": "198.51.100.55",
        "device": {"device_id": "human-alice-unknown", "os": "macos", "browser": "firefox"},
        "profile": {
            "typical_login_hour": alice.typical_login_hour,
            "trusted_ips": [alice.trusted_ip],
            "trusted_devices": [alice.trusted_device.lower()],
            "preferred_pages": alice.preferred_pages,
        },
    }).json()
    sid3 = session3["session_id"]

    r3 = requests.post(f"{BASE}/api/v1/events", json={
        "session_id": sid3, "events": [e.to_dict() for e in events3],
    }).json()
    print(f"  API result: score={r3['anomaly_score']:.4f}, risk={r3['risk_level']}, "
          f"state={r3['detected_state']}, ready={r3['score_ready']}, window={r3['window_event_count']}")
    print(f"  Reasons: {r3['reasons']}")
    test3_pass = r3["risk_level"] in ("HIGH", "MEDIUM")
    print(f"  {'[PASS]' if test3_pass else '[FAIL]'}: HUMAN attacker risk = {r3['risk_level']}")

    # --- TEST 4: FAST attacker ---
    print()
    print("=" * 60)
    print("TEST 4: FAST attacker -> HIGH risk")
    print("=" * 60)

    rng4 = random.Random(666)
    login_time4 = _make_login_time(rng4, alice.typical_login_hour, attacker_type="FAST")
    builder4 = _build_session_builder(rng4, login_time4)
    _simulate_login(builder4, alice, attacker_type="FAST")
    _simulate_fast_attacker(builder4, alice)
    events4 = builder4.finish()

    session4 = requests.post(f"{BASE}/api/v1/sessions/start", json={
        "user_id": alice.user_id,
        "ip_address": "198.51.100.77",
        "device": {"device_id": "fast-alice-unknown", "os": "macos", "browser": "firefox"},
        "profile": {
            "typical_login_hour": alice.typical_login_hour,
            "trusted_ips": [alice.trusted_ip],
            "trusted_devices": [alice.trusted_device.lower()],
            "preferred_pages": alice.preferred_pages,
        },
    }).json()
    sid4 = session4["session_id"]

    r4 = requests.post(f"{BASE}/api/v1/events", json={
        "session_id": sid4, "events": [e.to_dict() for e in events4],
    }).json()
    print(f"  API result: score={r4['anomaly_score']:.4f}, risk={r4['risk_level']}, "
          f"state={r4['detected_state']}, ready={r4['score_ready']}, window={r4['window_event_count']}")
    print(f"  Reasons: {r4['reasons']}")
    test4_pass = r4["risk_level"] in ("HIGH", "MEDIUM")
    print(f"  {'[PASS]' if test4_pass else '[FAIL]'}: FAST attacker risk = {r4['risk_level']}")

    # --- Summary ---
    print()
    print("=" * 60)
    results = [
        ("Normal -> LOW", test1_pass),
        ("BOT -> HIGH", test2_pass),
        ("HUMAN -> HIGH/MEDIUM", test3_pass),
        ("FAST -> HIGH/MEDIUM", test4_pass),
    ]
    passed = sum(1 for _, p in results if p)
    for name, p in results:
        print(f"  {'[PASS]' if p else '[FAIL]'} {name}")
    print(f"\nRESULTS: {passed}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_with_mock_infrastructure()
