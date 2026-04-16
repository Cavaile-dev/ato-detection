from __future__ import annotations

import asyncio
import json

from server.mock_data import generate_mock_dataset
from server.model import BehavioralAnomalyModel
from server.pipeline import StreamingRiskPipeline
from server.train_model import train_default_model


def chunk_list(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


async def stream_record(pipeline: StreamingRiskPipeline, record: dict) -> dict:
    state = pipeline.create_session(record["context"])
    print(f"\nStreaming {record['scenario']} session for {record['user_id']} -> {state.session_id}")

    latest = None
    for batch in chunk_list(record["events"], 10):
        latest = await asyncio.to_thread(pipeline.submit_events, state.session_id, batch)
        print(json.dumps(latest.to_dict(), indent=2))
        await asyncio.sleep(0.05)
    return latest.to_dict() if latest else {}


async def main() -> None:
    train_default_model()
    _, records = generate_mock_dataset(normal_sessions=1, attacker_sessions=1, seed=7, persist=False)
    model_service = BehavioralAnomalyModel().load()
    pipeline = StreamingRiskPipeline(model_service=model_service)

    try:
        for record in records:
            await stream_record(pipeline, record)
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
