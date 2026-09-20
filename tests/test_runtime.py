"""Tests for runtime audio, session configuration, and pipeline context."""

import asyncio
import contextlib
from types import SimpleNamespace

from gemini_live.live import LiveConfig
from gemini_live.runtime import (
    AudioStream,
    LiveSessionManager,
    active_pipeline_context,
)
from homeassistant.components.assist_pipeline.pipeline import (
    KEY_ASSIST_PIPELINE,
    PipelineEventType,
)
from homeassistant.core import Context


def _make_config(**overrides) -> LiveConfig:
    defaults = {"model": "m", "voice": "v", "system_instruction": "s"}
    defaults.update(overrides)
    return LiveConfig(**defaults)


def _pipeline_run(
    run_id: str,
    *,
    conversation_id: str,
    started_at: str,
    context: object,
    device_id: str = "device-1",
    stt_entity_id: str = "stt.live_model",
    final_event: PipelineEventType | None = None,
):
    events = [
        SimpleNamespace(
            type=PipelineEventType.RUN_START,
            data={"conversation_id": conversation_id},
            timestamp="0",
        ),
        SimpleNamespace(
            type=PipelineEventType.STT_START,
            data=None,
            timestamp=started_at,
        ),
    ]
    if final_event is not None:
        events.append(
            SimpleNamespace(type=final_event, data=None, timestamp="9")
        )
    return SimpleNamespace(
        id=run_id,
        pipeline=SimpleNamespace(id="pipeline-1"),
        stt_provider=SimpleNamespace(entity_id=stt_entity_id),
        _device_id=device_id,
        context=context,
        events=events,
    )


def _hass_with_pipeline_runs(*runs):
    return SimpleNamespace(
        data={
            KEY_ASSIST_PIPELINE: SimpleNamespace(
                pipeline_runs=SimpleNamespace(
                    _pipeline_runs={
                        "pipeline-1": {run.id: run for run in runs},
                    }
                ),
                pipeline_debug={
                    "pipeline-1": {
                        run.id: SimpleNamespace(events=run.events) for run in runs
                    }
                },
            )
        }
    )


def test_active_pipeline_context_preserves_the_run_context():
    source_context = Context(user_id="voice-user", parent_id="parent-context")
    run = _pipeline_run(
        "run-1",
        conversation_id="conversation-1",
        started_at="1",
        context=source_context,
    )

    conversation_id, device_id, pipeline_context = active_pipeline_context(
        _hass_with_pipeline_runs(run),
        "stt.live_model",
    )

    assert conversation_id == "conversation-1"
    assert device_id == "device-1"
    assert pipeline_context is source_context
    assert pipeline_context.user_id == "voice-user"
    assert pipeline_context.parent_id == "parent-context"


def test_active_pipeline_context_selects_the_most_recent_concurrent_run():
    older = _pipeline_run(
        "run-1",
        conversation_id="conversation-1",
        started_at="1",
        context=Context(user_id="first-user"),
        device_id="device-1",
    )
    newer = _pipeline_run(
        "run-2",
        conversation_id="conversation-2",
        started_at="2",
        context=Context(user_id="second-user"),
        device_id="device-2",
    )

    conversation_id, device_id, pipeline_context = active_pipeline_context(
        _hass_with_pipeline_runs(older, newer),
        "stt.live_model",
    )

    assert conversation_id == "conversation-2"
    assert device_id == "device-2"
    assert pipeline_context is newer.context


def test_active_pipeline_context_ignores_finished_and_error_runs(monkeypatch):
    finished = _pipeline_run(
        "run-1",
        conversation_id="conversation-1",
        started_at="1",
        context=Context(),
        final_event=PipelineEventType.RUN_END,
    )
    failed = _pipeline_run(
        "run-2",
        conversation_id="conversation-2",
        started_at="2",
        context=Context(),
        final_event=PipelineEventType.ERROR,
    )
    monkeypatch.setattr(
        "gemini_live.runtime.new_conversation_id",
        lambda: "temporary-conversation",
    )

    assert active_pipeline_context(
        _hass_with_pipeline_runs(finished, failed),
        "stt.live_model",
    ) == ("temporary-conversation", None, None)


def test_active_pipeline_context_rejects_an_invalid_context():
    run = _pipeline_run(
        "run-1",
        conversation_id="conversation-1",
        started_at="1",
        context=object(),
    )

    assert active_pipeline_context(
        _hass_with_pipeline_runs(run),
        "stt.live_model",
    ) == ("conversation-1", "device-1", None)


async def test_interrupt_discards_queued_audio_and_keeps_stream_open():
    stream = AudioStream()
    await stream.add_chunk(b"old-a")
    await stream.add_chunk(b"old-b")

    stream.interrupt()

    consumer = asyncio.create_task(anext(stream.async_chunks()))
    await stream.add_chunk(b"new")
    assert await asyncio.wait_for(consumer, 1) == b"new"


async def test_interrupted_stream_delivers_only_replacement_audio():
    stream = AudioStream()
    await stream.add_chunk(b"a1")
    await stream.add_chunk(b"a2")
    stream.interrupt()
    await stream.add_chunk(b"b1")
    stream.finish()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == [b"b1"]


async def test_interrupt_is_not_finish():
    stream = AudioStream()
    await stream.add_chunk(b"a")
    stream.interrupt()
    await stream.add_chunk(b"b")

    consumer = asyncio.create_task(anext(stream.async_chunks()))
    assert await asyncio.wait_for(consumer, 1) == b"b"

    # The stream is still open: a blocked consumer does not see end-of-stream.
    blocked = asyncio.create_task(anext(stream.async_chunks()))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    blocked.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await blocked


async def test_interrupt_after_finish_keeps_end_of_stream_sentinel():
    stream = AudioStream()
    stream.finish()
    stream.interrupt()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == []


async def test_interrupt_does_not_invoke_cancellation_callback():
    cancel_calls = []
    stream = AudioStream(lambda: cancel_calls.append(True))
    await stream.add_chunk(b"a")
    stream.interrupt()
    stream.finish()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == []
    assert cancel_calls == []


async def test_interrupt_notifies_subscribers():
    stream = AudioStream()
    interruptions = []
    unsubscribe = stream.subscribe_interrupt(lambda: interruptions.append(True))

    stream.interrupt()
    unsubscribe()
    stream.interrupt()

    assert interruptions == [True]


async def test_abandoned_stream_still_invokes_cancellation_callback():
    cancel_calls = []
    stream = AudioStream(lambda: cancel_calls.append(True))
    await stream.add_chunk(b"a")
    stream.finish()

    generator = stream.async_chunks()
    assert await anext(generator) == b"a"
    await generator.aclose()
    assert cancel_calls == [True]


async def test_add_chunk_pauses_producer_when_buffer_is_full():
    stream = AudioStream(max_buffer_bytes=4)
    await stream.add_chunk(b"aaaa")

    # The producer must wait instead of racing ahead of the consumer.
    producer = asyncio.create_task(stream.add_chunk(b"b"))
    await asyncio.sleep(0.01)
    assert not producer.done()

    consumer = asyncio.create_task(anext(stream.async_chunks()))
    assert await asyncio.wait_for(consumer, 1) == b"aaaa"
    await asyncio.wait_for(producer, 1)


async def test_interrupt_unblocks_paused_producer_and_queues_replacement_audio():
    stream = AudioStream(max_buffer_bytes=4)
    await stream.add_chunk(b"old!")
    producer = asyncio.create_task(stream.add_chunk(b"new!"))
    await asyncio.sleep(0.01)
    assert not producer.done()

    stream.interrupt()
    await asyncio.wait_for(producer, 1)
    stream.finish()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == [b"new!"]


async def test_finish_unblocks_paused_producer_without_queuing_its_chunk():
    stream = AudioStream(max_buffer_bytes=4)
    await stream.add_chunk(b"kept")
    producer = asyncio.create_task(stream.add_chunk(b"drop"))
    await asyncio.sleep(0.01)
    assert not producer.done()

    stream.finish()
    await asyncio.wait_for(producer, 1)

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == [b"kept"]


def test_config_signature_changes_when_barge_in_toggles():
    legacy = _make_config()
    barge_in = _make_config(support_barge_in=True)

    signature = LiveSessionManager._config_signature

    assert signature(legacy) != signature(barge_in)
