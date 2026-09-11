"""Tests for AudioStream interruption and session configuration signatures."""

import asyncio
import contextlib

from gemini_live.live import LiveConfig
from gemini_live.runtime import AudioStream, LiveSessionManager


def _make_config(**overrides) -> LiveConfig:
    defaults = {"model": "m", "voice": "v", "system_instruction": "s"}
    defaults.update(overrides)
    return LiveConfig(**defaults)


async def test_interrupt_discards_queued_audio_and_keeps_stream_open():
    stream = AudioStream()
    stream.add_chunk(b"old-a")
    stream.add_chunk(b"old-b")

    stream.interrupt()

    consumer = asyncio.create_task(anext(stream.async_chunks()))
    stream.add_chunk(b"new")
    assert await asyncio.wait_for(consumer, 1) == b"new"


async def test_interrupted_stream_delivers_only_replacement_audio():
    stream = AudioStream()
    stream.add_chunk(b"a1")
    stream.add_chunk(b"a2")
    stream.interrupt()
    stream.add_chunk(b"b1")
    stream.finish()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == [b"b1"]


async def test_interrupt_is_not_finish():
    stream = AudioStream()
    stream.add_chunk(b"a")
    stream.interrupt()
    stream.add_chunk(b"b")

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
    stream.add_chunk(b"a")
    stream.interrupt()
    stream.finish()

    chunks = [chunk async for chunk in stream.async_chunks()]
    assert chunks == []
    assert cancel_calls == []


async def test_abandoned_stream_still_invokes_cancellation_callback():
    cancel_calls = []
    stream = AudioStream(lambda: cancel_calls.append(True))
    stream.add_chunk(b"a")
    stream.finish()

    generator = stream.async_chunks()
    assert await anext(generator) == b"a"
    await generator.aclose()
    assert cancel_calls == [True]


def test_config_signature_changes_when_barge_in_toggles():
    legacy = _make_config()
    barge_in = _make_config(support_barge_in=True)

    signature = LiveSessionManager._config_signature

    assert signature(legacy) != signature(barge_in)
