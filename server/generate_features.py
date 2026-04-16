from __future__ import annotations

import json

import pandas as pd

from server.config import MOCK_FEATURE_PATH, MOCK_SESSION_PATH
from server.feature_extraction import extract_features
from server.schemas import BehaviorEvent, SessionContext, SessionState


def regenerate_feature_table(session_path=MOCK_SESSION_PATH, output_path=MOCK_FEATURE_PATH) -> pd.DataFrame:
    rows: list[dict] = []
    with session_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            context = SessionContext.from_payload(record["context"], session_id=record["session_id"])
            events = [BehaviorEvent.from_payload(event) for event in record["events"]]
            state = SessionState(
                context=context,
                started_at=context.login_time,
                events=events,
                navigation_sequence=[event.page for event in events if event.event_type == "navigation" and event.page],
            )
            features, _ = extract_features(state)   # unpack (model_features, raw_values)
            row = {
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "label": record["label"],
                "scenario": record["scenario"],
            }
            row.update(features)
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    frame = regenerate_feature_table()
    print(frame.head().to_string(index=False))
    print(f"\nSaved {len(frame)} feature rows to {output_path_or_default()}")


def output_path_or_default() -> str:
    return str(MOCK_FEATURE_PATH)


if __name__ == "__main__":
    main()
