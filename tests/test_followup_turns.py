"""Follow-up voice turns and transcript streaming without barge-in."""

import asyncio
from contextlib import aclosing

import pytest

from gemini_live.const import (
    CONF_RESPONSE_TIMEOUT,
    CONF_SUPPORT_BARGE_IN,
)
from gemini_live.live import LiveEvent

from test_stt_barge_in import (
    ENTITY_CLASSES,
    MIC_CHUNK,
    FakeHass,
    MicStream,
    ScriptedClient,
    ScriptedSession,
    _bind_client,
    _make_entity,
    _metadata,
)

# 9600 bytes of 24 kHz PCM -> 6400 bytes at 16 kHz, one full stream buffer.
LONG_AUDIO = b"\x01\x00" * 4800


class LongReply(ScriptedSession):
    """A reply whose audio is larger than the AudioStream buffer."""

    def __init__(self) -> None:
        super().__init__(support_barge_in=False)

    async def receive(self):
        yield LiveEvent(input_transcript="question")
        yield LiveEvent(audio=LONG_AUDIO, output_transcript="Morgen ")
        for word in ("wird ", "es ", "trocken."):
            yield LiveEvent(audio=LONG_AUDIO, output_transcript=word)
        yield LiveEvent(turn_complete=True)


class StuckCloseClient(ScriptedClient):
    """A client whose websocket close never completes."""

    def connect(self, config):
        self.captured_config = config
        self.connect_count += 1
        outer = self

        class _Connect:
            async def __aenter__(self_inner):
                return outer._session

            async def __aexit__(self_inner, *_exc):
                outer.close_count += 1
                await asyncio.Event().wait()

        return _Connect()


async def _run(entity, mic, transcribe: bool = False):
    return await entity._async_run_audio_stream_sdk(
        _metadata(), mic.chunks(), "k", "m", "v", "", transcribe, False,
        False, asyncio.Future(), "conversation-1", None,
    )


async def _wait_for_turn(turns, timeout: float = 2.0):
    async with asyncio.timeout(timeout):
        while not turns._voice_turns:
            await asyncio.sleep(0.01)
    return turns.take_voice_turn("conversation-1", "question")


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
@pytest.mark.parametrize("playback", ["stopped_early", "never_fetched"])
async def test_third_follow_up_turn_does_not_hang(
    entity_class, playback, monkeypatch
) -> None:
    """A follow-up turn must not wait forever for the previous session to close."""
    monkeypatch.setattr("gemini_live.runtime.SESSION_CLOSE_TIMEOUT", 0.05, raising=False)
    entity, manager, turns = _make_entity(
        FakeHass(),
        {"api_key": "k", CONF_SUPPORT_BARGE_IN: False, CONF_RESPONSE_TIMEOUT: 120},
        entity_class,
    )
    client = StuckCloseClient(LongReply())
    _bind_client(entity, client)
    runs = []
    try:
        for _ in range(3):
            client._session = LongReply()
            mic = MicStream()
            mic.put(MIC_CHUNK)
            mic.close()
            runs.append(asyncio.create_task(_run(entity, mic)))
            turn = await _wait_for_turn(turns)
            assert turn is not None
            if playback == "stopped_early":
                async with aclosing(turn.audio.async_chunks()) as chunks:
                    async for _chunk in chunks:
                        break
    finally:
        for run in runs:
            run.cancel()


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_transcript_streams_before_audio_is_consumed(entity_class) -> None:
    """Unread audio must not stall the transcript Home Assistant waits for."""
    entity, manager, turns = _make_entity(
        FakeHass(),
        {"api_key": "k", CONF_SUPPORT_BARGE_IN: False, CONF_RESPONSE_TIMEOUT: 120},
        entity_class,
    )
    _bind_client(entity, ScriptedClient(LongReply()))
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    run = asyncio.create_task(_run(entity, mic, transcribe=True))
    try:
        turn = await _wait_for_turn(turns)
        assert turn is not None and turn.assistant_text_stream is not None
        # Home Assistant only starts TTS once enough text has arrived, so
        # the whole transcript must stream while nobody reads the audio yet.
        async with asyncio.timeout(2):
            text = "".join(
                [chunk async for chunk in turn.assistant_text_stream.async_chunks()]
            )
        assert text == "Morgen wird es trocken."
        chunks = [chunk async for chunk in turn.audio.async_chunks()]
        assert sum(map(len, chunks)) == 4 * 6400
        await asyncio.wait_for(run, 2)
    finally:
        run.cancel()
        await manager.async_close_all()
