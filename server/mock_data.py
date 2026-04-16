from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from server.config import MOCK_FEATURE_PATH, MOCK_SESSION_PATH
from server.feature_extraction import extract_features
from server.schemas import BehaviorEvent, DeviceInfo, SessionContext, SessionState, UserProfile


ATTACKER_TYPES = ("BOT", "HUMAN", "FAST")


@dataclass(slots=True)
class SyntheticUser:
    user_id: str
    typical_login_hour: float
    trusted_ip: str
    trusted_device: str
    mouse_speed_mean: float
    mouse_speed_std: float
    typing_interval_mean: float
    hold_time_mean: float
    hesitation_mean: float
    preferred_pages: list[str]
    search_terms: list[str]


@dataclass(slots=True)
class SessionBuilder:
    rng: random.Random
    now: float
    x: float
    y: float
    current_page: str = "/login"
    page_entered_at: float | None = None
    last_click_time: float | None = None
    last_key_time: float | None = None
    events: list[BehaviorEvent] = field(default_factory=list)

    def _append(self, event: BehaviorEvent) -> None:
        self.events.append(event)

    def wait(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)

    def move_mouse(
        self,
        *,
        page: str,
        target_x: float,
        target_y: float,
        steps: int,
        speed_mean: float,
        speed_std: float,
        pattern: str,
    ) -> None:
        start_x = self.x
        start_y = self.y
        previous_angle = None

        for index in range(1, steps + 1):
            progress = index / steps
            if pattern == "BOT":
                x = start_x + ((target_x - start_x) * progress)
                y = start_y + ((target_y - start_y) * progress)
                velocity = speed_mean
                acceleration = 0.0
                angle = math.atan2(target_y - start_y, target_x - start_x)
            elif pattern == "FAST":
                wave = math.sin(progress * math.pi * 6.0) * 4.0
                x = start_x + ((target_x - start_x) * progress) + wave
                y = start_y + ((target_y - start_y) * progress) - wave
                velocity = abs(self.rng.gauss(speed_mean * 1.6, speed_std * 0.25))
                acceleration = abs(self.rng.gauss(3200.0, 900.0))
                angle = math.atan2(y - self.y, x - self.x)
            elif pattern == "HUMAN_ATTACKER":
                wobble = self.rng.uniform(-22.0, 22.0)
                x = start_x + ((target_x - start_x) * progress) + (math.sin(progress * math.pi * 2.0) * wobble)
                y = start_y + ((target_y - start_y) * progress) + (math.cos(progress * math.pi * 2.0) * wobble)
                velocity = abs(self.rng.gauss(speed_mean * 1.2, speed_std * 1.6))
                acceleration = abs(self.rng.gauss(1700.0, 850.0))
                angle = math.atan2(y - self.y, x - self.x)
            else:
                curve = math.sin(progress * math.pi) * self.rng.uniform(8.0, 24.0)
                x = start_x + ((target_x - start_x) * progress) + curve
                y = start_y + ((target_y - start_y) * progress) + (curve / 2.0)
                velocity = abs(self.rng.gauss(speed_mean, speed_std))
                acceleration = abs(self.rng.gauss(850.0, 250.0))
                angle = math.atan2(y - self.y, x - self.x)

            direction_change = 0.0 if previous_angle is None else abs(angle - previous_angle)
            previous_angle = angle
            delta_seconds = max(
                math.hypot(x - self.x, y - self.y) / max(velocity, 30.0),
                0.012 if pattern in {"BOT", "FAST"} else 0.04,
            )
            self.wait(delta_seconds)
            self.x = max(0.0, min(1366.0, x))
            self.y = max(0.0, min(768.0, y))
            self._append(
                BehaviorEvent(
                    event_type="mouse_move",
                    timestamp=self.now,
                    page=page,
                    x=self.x,
                    y=self.y,
                    velocity=velocity,
                    acceleration=acceleration,
                    trajectory_angle=angle,
                    direction_change=direction_change,
                )
            )

    def click(self, *, page: str, hesitation: float) -> None:
        self.wait(hesitation)
        click_interval = None if self.last_click_time is None else self.now - self.last_click_time
        self._append(
            BehaviorEvent(
                event_type="click",
                timestamp=self.now,
                page=page,
                x=self.x,
                y=self.y,
                click_interval=click_interval,
                hesitation=hesitation,
            )
        )
        self.last_click_time = self.now

    def type_text(
        self,
        *,
        page: str,
        field_name: str,
        text: str,
        interval_mean: float,
        interval_jitter: float,
        hold_mean: float,
        hold_jitter: float,
    ) -> None:
        for character in text:
            interval = max(0.02, self.rng.gauss(interval_mean, interval_jitter))
            hold_time = max(0.01, self.rng.gauss(hold_mean, hold_jitter))
            self.wait(interval)
            self._append(
                BehaviorEvent(
                    event_type="keystroke",
                    timestamp=self.now,
                    page=page,
                    key_interval=None if self.last_key_time is None else self.now - self.last_key_time,
                    hold_time=hold_time,
                    field_name=field_name,
                    metadata={"character": character},
                )
            )
            self.last_key_time = self.now
            self.wait(hold_time)

    def scroll(self, *, page: str, amount: float, steps: int) -> None:
        for _ in range(steps):
            delta = amount / max(steps, 1)
            self.wait(self.rng.uniform(0.06, 0.22))
            self._append(
                BehaviorEvent(
                    event_type="scroll",
                    timestamp=self.now,
                    page=page,
                    scroll_delta=delta,
                )
            )

    def navigate(self, page: str) -> None:
        if self.page_entered_at is not None:
            self._append(
                BehaviorEvent(
                    event_type="page_exit",
                    timestamp=self.now,
                    page=self.current_page,
                    dwell_time=max(self.now - self.page_entered_at, 0.0),
                )
            )
        self.wait(self.rng.uniform(0.18, 0.45))
        self.current_page = page
        self.page_entered_at = self.now
        self._append(BehaviorEvent(event_type="navigation", timestamp=self.now, page=page))

    def finish(self) -> list[BehaviorEvent]:
        if self.page_entered_at is not None:
            self._append(
                BehaviorEvent(
                    event_type="page_exit",
                    timestamp=self.now,
                    page=self.current_page,
                    dwell_time=max(self.now - self.page_entered_at, 0.0),
                )
            )
        self._append(BehaviorEvent(event_type="session_end", timestamp=self.now, page=self.current_page))
        return self.events


def _build_users() -> list[SyntheticUser]:
    return [
        SyntheticUser(
            user_id="alice",
            typical_login_hour=8.5,
            trusted_ip="203.0.113.18",
            trusted_device="alice-work-laptop",
            mouse_speed_mean=760.0,
            mouse_speed_std=140.0,
            typing_interval_mean=0.19,
            hold_time_mean=0.11,
            hesitation_mean=0.42,
            preferred_pages=["/home", "/profile", "/settings", "/home"],
            search_terms=["salary report", "travel notice", "security tips"],
        ),
        SyntheticUser(
            user_id="bravo",
            typical_login_hour=10.0,
            trusted_ip="203.0.113.27",
            trusted_device="bravo-desktop",
            mouse_speed_mean=620.0,
            mouse_speed_std=120.0,
            typing_interval_mean=0.23,
            hold_time_mean=0.10,
            hesitation_mean=0.54,
            preferred_pages=["/home", "/settings", "/profile", "/home"],
            search_terms=["invoice lookup", "device status", "meeting notes"],
        ),
        SyntheticUser(
            user_id="charlie",
            typical_login_hour=14.0,
            trusted_ip="203.0.113.45",
            trusted_device="charlie-macbook",
            mouse_speed_mean=890.0,
            mouse_speed_std=170.0,
            typing_interval_mean=0.17,
            hold_time_mean=0.09,
            hesitation_mean=0.36,
            preferred_pages=["/home", "/profile", "/home", "/settings"],
            search_terms=["travel form", "quota reset", "profile update"],
        ),
    ]


def _make_login_time(rng: random.Random, typical_hour: float, attacker_type: str | None) -> datetime:
    base_day = datetime(2026, 4, 14, tzinfo=timezone.utc)
    if attacker_type:
        hour = (typical_hour + rng.uniform(4.5, 10.5)) % 24
    else:
        hour = max(0.0, min(23.99, typical_hour + rng.uniform(-1.1, 1.1)))
    hours = int(hour)
    minutes = int((hour - hours) * 60)
    return base_day + timedelta(hours=hours, minutes=minutes, seconds=rng.randint(0, 59))


def _build_session_builder(rng: random.Random, login_time: datetime) -> SessionBuilder:
    return SessionBuilder(
        rng=rng,
        now=login_time.timestamp(),
        x=rng.uniform(140.0, 420.0),
        y=rng.uniform(120.0, 260.0),
    )


def _simulate_login(builder: SessionBuilder, user: SyntheticUser, attacker_type: str | None) -> None:
    if attacker_type == "BOT":
        pattern = "BOT"
        interval_mean = 0.035
        hold_mean = 0.018
        hesitation = 0.01
    elif attacker_type == "HUMAN":
        pattern = "HUMAN_ATTACKER"
        interval_mean = user.typing_interval_mean * 0.72
        hold_mean = user.hold_time_mean * 0.75
        hesitation = 0.08
    elif attacker_type == "FAST":
        pattern = "FAST"
        interval_mean = 0.055
        hold_mean = 0.025
        hesitation = 0.03
    else:
        pattern = "NORMAL"
        interval_mean = user.typing_interval_mean
        hold_mean = user.hold_time_mean
        hesitation = user.hesitation_mean

    builder.move_mouse(page="/login", target_x=330.0, target_y=220.0, steps=8, speed_mean=user.mouse_speed_mean, speed_std=user.mouse_speed_std, pattern=pattern)
    builder.click(page="/login", hesitation=hesitation)
    builder.type_text(
        page="/login",
        field_name="username",
        text=user.user_id,
        interval_mean=interval_mean,
        interval_jitter=interval_mean * 0.18,
        hold_mean=hold_mean,
        hold_jitter=max(hold_mean * 0.12, 0.01),
    )

    builder.move_mouse(page="/login", target_x=344.0, target_y=296.0, steps=6, speed_mean=user.mouse_speed_mean, speed_std=user.mouse_speed_std, pattern=pattern)
    builder.click(page="/login", hesitation=hesitation / 2.0)
    builder.type_text(
        page="/login",
        field_name="password",
        text="hunter2demo",
        interval_mean=interval_mean * (0.95 if attacker_type else 1.0),
        interval_jitter=interval_mean * 0.20,
        hold_mean=hold_mean,
        hold_jitter=max(hold_mean * 0.12, 0.01),
    )

    builder.move_mouse(page="/login", target_x=388.0, target_y=378.0, steps=6, speed_mean=user.mouse_speed_mean, speed_std=user.mouse_speed_std, pattern=pattern)
    builder.click(page="/login", hesitation=hesitation)


def _simulate_normal_pages(builder: SessionBuilder, user: SyntheticUser) -> None:
    for index, page in enumerate(user.preferred_pages):
        builder.navigate(page)
        builder.move_mouse(
            page=page,
            target_x=260.0 + (index * 80.0),
            target_y=160.0 + (index * 40.0),
            steps=10 + index,
            speed_mean=user.mouse_speed_mean,
            speed_std=user.mouse_speed_std,
            pattern="NORMAL",
        )

        if page == "/home":
            builder.click(page=page, hesitation=user.hesitation_mean)
            builder.type_text(
                page=page,
                field_name="search",
                text=user.search_terms[index % len(user.search_terms)],
                interval_mean=user.typing_interval_mean,
                interval_jitter=0.04,
                hold_mean=user.hold_time_mean,
                hold_jitter=0.02,
            )
            builder.scroll(page=page, amount=900.0, steps=4)
        elif page == "/profile":
            builder.click(page=page, hesitation=user.hesitation_mean + 0.08)
            builder.type_text(
                page=page,
                field_name="display_name",
                text=f"{user.user_id} analyst",
                interval_mean=user.typing_interval_mean * 1.05,
                interval_jitter=0.05,
                hold_mean=user.hold_time_mean,
                hold_jitter=0.02,
            )
        elif page == "/settings":
            builder.click(page=page, hesitation=user.hesitation_mean)
            builder.type_text(
                page=page,
                field_name="device_alias",
                text=f"{user.user_id}-secure-device",
                interval_mean=user.typing_interval_mean * 0.95,
                interval_jitter=0.04,
                hold_mean=user.hold_time_mean,
                hold_jitter=0.02,
            )
            builder.scroll(page=page, amount=420.0, steps=2)

        builder.wait(builder.rng.uniform(0.8, 2.3))


def _simulate_bot_attacker(builder: SessionBuilder, user: SyntheticUser) -> None:
    for page in ["/settings", "/profile", "/settings"]:
        builder.navigate(page)
        builder.move_mouse(
            page=page,
            target_x=1040.0,
            target_y=220.0 if page == "/settings" else 320.0,
            steps=7,
            speed_mean=user.mouse_speed_mean * 2.2,
            speed_std=1.0,
            pattern="BOT",
        )
        builder.click(page=page, hesitation=0.0)
        if page == "/settings":
            builder.type_text(
                page=page,
                field_name="security_email",
                text="botcontrol",
                interval_mean=0.028,
                interval_jitter=0.002,
                hold_mean=0.016,
                hold_jitter=0.001,
            )
        builder.wait(0.10)


def _simulate_human_attacker(builder: SessionBuilder, user: SyntheticUser) -> None:
    for page in ["/home", "/settings", "/home", "/profile"]:
        builder.navigate(page)
        builder.move_mouse(
            page=page,
            target_x=builder.rng.uniform(820.0, 1180.0),
            target_y=builder.rng.uniform(160.0, 620.0),
            steps=8,
            speed_mean=user.mouse_speed_mean * 1.08,
            speed_std=user.mouse_speed_std * 1.15,
            pattern="HUMAN_ATTACKER",
        )
        builder.click(page=page, hesitation=builder.rng.uniform(0.02, 0.16))
        if page != "/home":
            builder.type_text(
                page=page,
                field_name="suspicious_input",
                text="change-now",
                interval_mean=0.170,
                interval_jitter=0.035,
                hold_mean=0.100,
                hold_jitter=0.018,
            )
        builder.wait(builder.rng.uniform(0.15, 0.55))


def _simulate_fast_attacker(builder: SessionBuilder, user: SyntheticUser) -> None:
    for page in ["/home", "/profile", "/settings", "/profile"]:
        builder.navigate(page)
        builder.move_mouse(
            page=page,
            target_x=builder.rng.uniform(760.0, 1240.0),
            target_y=builder.rng.uniform(120.0, 560.0),
            steps=5,
            speed_mean=user.mouse_speed_mean * 1.25,
            speed_std=user.mouse_speed_std * 0.95,
            pattern="FAST",
        )
        builder.click(page=page, hesitation=0.015)
        builder.wait(0.03)
        if page == "/profile":
            builder.type_text(
                page=page,
                field_name="nickname",
                text="rush",
                interval_mean=0.145,
                interval_jitter=0.020,
                hold_mean=0.085,
                hold_jitter=0.010,
            )
        builder.wait(0.08)


def _session_record(user: SyntheticUser, rng: random.Random, attacker_type: str | None) -> dict:
    login_time = _make_login_time(rng, user.typical_login_hour, attacker_type=attacker_type)
    profile = UserProfile(
        typical_login_hour=user.typical_login_hour,
        trusted_ips=[user.trusted_ip],
        trusted_devices=[user.trusted_device.lower()],
        preferred_pages=user.preferred_pages,
    )

    context = SessionContext(
        session_id=str(uuid.uuid4()),
        user_id=user.user_id,
        ip_address=user.trusted_ip if attacker_type is None else f"198.51.100.{rng.randint(20, 220)}",
        device=DeviceInfo(
            device_id=user.trusted_device if attacker_type is None else f"{attacker_type.lower()}-{user.user_id}-unknown",
            device_type="desktop",
            os="windows" if attacker_type is None else ("linux" if attacker_type == "BOT" else "macos"),
            browser="chrome" if attacker_type is None else ("headless" if attacker_type == "BOT" else "firefox"),
        ),
        login_time=login_time,
        profile=profile,
    )

    builder = _build_session_builder(rng, login_time)
    _simulate_login(builder, user, attacker_type)

    if attacker_type == "BOT":
        _simulate_bot_attacker(builder, user)
    elif attacker_type == "HUMAN":
        _simulate_human_attacker(builder, user)
    elif attacker_type == "FAST":
        _simulate_fast_attacker(builder, user)
    else:
        _simulate_normal_pages(builder, user)

    events = builder.finish()
    navigation_sequence = [event.page for event in events if event.event_type == "navigation" and event.page]
    state = SessionState(context=context, started_at=login_time, events=events, navigation_sequence=navigation_sequence)
    features, _ = extract_features(state)   # unpack (model_features, raw_values)
    
    if attacker_type is None:
        label = 0
    elif attacker_type == "BOT":
        label = 2
    else:
        label = 1  # Human attackers (HUMAN, FAST)
        
    scenario = "normal" if attacker_type is None else attacker_type.lower()

    return {
        "session_id": context.session_id,
        "user_id": context.user_id,
        "label": label,
        "scenario": scenario,
        "context": context.to_dict(),
        "events": [event.to_dict() for event in events],
        "features": features,
    }


def generate_mock_dataset(
    normal_sessions: int = 240,
    attacker_sessions: int = 120,
    seed: int = 42,
    persist: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    rng = random.Random(seed)
    users = _build_users()
    records: list[dict] = []

    for _ in range(normal_sessions):
        records.append(_session_record(rng.choice(users), rng, attacker_type=None))

    per_type = max(attacker_sessions // len(ATTACKER_TYPES), 1)
    attacker_total = 0
    for attacker_type in ATTACKER_TYPES:
        for _ in range(per_type):
            if attacker_total >= attacker_sessions:
                break
            records.append(_session_record(rng.choice(users), rng, attacker_type=attacker_type))
            attacker_total += 1
    while attacker_total < attacker_sessions:
        attacker_type = ATTACKER_TYPES[attacker_total % len(ATTACKER_TYPES)]
        records.append(_session_record(rng.choice(users), rng, attacker_type=attacker_type))
        attacker_total += 1

    feature_rows = []
    for record in records:
        row = {
            "session_id": record["session_id"],
            "user_id": record["user_id"],
            "label": record["label"],
            "scenario": record["scenario"],
        }
        row.update(record["features"])
        feature_rows.append(row)

    frame = pd.DataFrame(feature_rows)

    if persist:
        frame.to_csv(MOCK_FEATURE_PATH, index=False)
        with MOCK_SESSION_PATH.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")

    return frame, records


def main() -> None:
    frame, records = generate_mock_dataset()
    print(f"Generated {len(records)} mock sessions")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
