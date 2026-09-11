"""Integration tests for barge-in behaviour in the shared STT pipeline.

These tests drive the real provider-neutral pipeline in stt.py against a
scripted LiveSession, covering the sender lifecycle, the interruption state
machine, and Home Assistant audio processing negotiation.
"""

import asyncio
from collections.abc import AsyncIterable
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest
from gemini_live.const import (
    CONF_SUPPORT_BARGE_IN,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    DOMAIN,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
)
from gemini_live.live import LiveConfig, LiveEvent
from gemini_live.runtime import LiveSessionManager, TurnStore
from gemini_live.stt import GeminiLiveSTT, GPTRealtimeSTT
from gemini_live.utils import PCM24kTo16kStreamResampler, resample_24k_to_16k
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResultState,
)
from homeassistant.helpers.issue_registry import DATA_REGISTRY as DATA_ISSUE_REGISTRY

AUDIO_A = b"\x01\x00" * 12
AUDIO_B = b"\x02\x00" * 12
MIC_CHUNK = b"\x00\x00" * 3200
EXTRA_MIC_CHUNKS = 2

ENTITY_CLASSES = [GeminiLiveSTT, GPTRealtimeSTT]


def test_stream_resampler_preserves_phase_across_arbitrary_chunks() -> None:
    """Streaming conversion must equal conversion of the contiguous PCM."""
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in range(10))
    resampler = PCM24kTo16kStreamResampler()

    converted = b"".join(
        (
            resampler.process(pcm[:5]),
            resampler.process(pcm[5:13]),
            resampler.process(pcm[13:]),
            resampler.flush(),
        )
    )

    assert converted == resample_24k_to_16k(pcm)


def _metadata() -> SpeechMetadata:
    return SpeechMetadata(
        language="en",
        format=AudioFormats.WAV,
        codec=AudioCodecs.PCM,
        bit_rate=AudioBitRates.BITRATE_16,
        sample_rate=AudioSampleRates.SAMPLERATE_16000,
        channel=AudioChannels.CHANNEL_MONO,
    )


class FakeConfig:
    """Minimal hass.config stub."""

    config_dir = "/tmp/fake-config"

    def path(self, *parts: str) -> str:
        return "/".join((self.config_dir, *parts))


class FakeIssueRegistry:
    """No-op issue registry so HA never constructs a real one."""

    def async_delete(self, _domain: str, _issue_id: str) -> None:
        return None


class FakeHass:
    """Minimal hass stub for the STT pipeline path under test."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            DATA_ISSUE_REGISTRY: FakeIssueRegistry(),
        }
        self.config = FakeConfig()
        self.background_tasks: list[asyncio.Task] = []

    def async_create_background_task(self, target, name, **_kwargs):
        task = asyncio.create_task(target, name=name)
        self.background_tasks.append(task)
        return task

    def async_create_task(self, target, name=None, **_kwargs):
        return asyncio.create_task(target, name=name)


class MicStream:
    """A controllable Home Assistant microphone stream."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self._sentinel = object()

    async def chunks(self) -> AsyncIterable[bytes]:
        while True:
            item = await self.queue.get()
            if item is self._sentinel:
                return
            yield item

    def put(self, chunk: bytes) -> None:
        self.queue.put_nowait(chunk)

    def close(self) -> None:
        self.queue.put_nowait(self._sentinel)


class ScriptedSession:
    """A LiveSession double whose receive() follows a scripted flow."""

    def __init__(self, support_barge_in: bool) -> None:
        self.support_barge_in = support_barge_in
        self.sent_audio: list[bytes] = []
        self.end_audio_count = 0
        self.reply_started = asyncio.Event()
        self.release_gate = asyncio.Event()
        self.is_open = True

    async def send_audio(self, audio: bytes) -> None:
        self.sent_audio.append(audio)

    async def end_audio(self) -> None:
        self.end_audio_count += 1

    async def send_text(self, _text: str) -> None:
        raise AssertionError("send_text must not be used in the voice path")

    async def send_tool_responses(self, _responses) -> None:
        raise AssertionError("no scripted tool calls in this test")

    async def receive(self):
        if self.support_barge_in:
            self.reply_started.set()
            yield LiveEvent(input_transcript="user request")
            yield LiveEvent(
                audio=AUDIO_A,
                output_transcript="interrupted transcript",
            )
            await self.release_gate.wait()
            yield LiveEvent(interrupted=True, turn_complete=True)
            yield LiveEvent(
                audio=AUDIO_B,
                output_transcript="replacement transcript",
            )
            yield LiveEvent(turn_complete=True)
        else:
            self.reply_started.set()
            yield LiveEvent(
                audio=AUDIO_A,
                output_transcript="unsolicited transcript",
            )
            await self.release_gate.wait()
            yield LiveEvent(turn_complete=True)


class ScriptedClient:
    """A LiveClient double that hands out the scripted session."""

    def __init__(self, session: ScriptedSession) -> None:
        self._session = session
        self.captured_config: LiveConfig | None = None

    def connect(self, config: LiveConfig):
        self.captured_config = config

        class _Connect:
            async def __aenter__(self_inner):
                return self._session

            async def __aexit__(self_inner, *_exc):
                return None

        return _Connect()


def _make_entity(hass: FakeHass, entry_data: dict[str, Any], entity_class):
    entry = SimpleNamespace(
        data=dict(entry_data),
        options={},
        entry_id="test_entry",
        title="Test entry",
    )
    entity = entity_class(entry)
    entity.hass = hass
    entity.entity_id = "stt.live_model"
    session_manager = LiveSessionManager()
    turn_store = TurnStore()
    hass.data[DOMAIN] = {
        entry.entry_id: {
            GEMINI_SESSION_MANAGER_KEY: session_manager,
            GEMINI_TURN_STORE_KEY: turn_store,
        }
    }
    return entity, session_manager, turn_store


def _bind_client(entity, scripted_client: ScriptedClient) -> None:
    async def fake_create_client(_api_key: str) -> ScriptedClient:
        return scripted_client

    entity._async_create_client = fake_create_client


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
@pytest.mark.parametrize("barge_in", [False, True])
async def test_audio_processing_negotiates_external_vad(
    entity_class, barge_in: bool
) -> None:
    entry_data = {
        "api_key": "k",
        CONF_SUPPORT_BARGE_IN: barge_in,
    }
    entity, _session_manager, _turn_store = _make_entity(
        FakeHass(), entry_data, entity_class
    )

    processing = entity.audio_processing

    assert processing.requires_external_vad is not barge_in
    assert processing.prefers_auto_gain_enabled is True
    assert processing.prefers_noise_reduction_enabled is True


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_barge_in_keeps_microphone_forwarding_after_reply(
    entity_class,
) -> None:
    hass = FakeHass()
    entry_data = {
        "api_key": "k",
        CONF_SUPPORT_BARGE_IN: True,
    }
    entity, _session_manager, turn_store = _make_entity(hass, entry_data, entity_class)
    session = ScriptedSession(support_barge_in=True)
    scripted_client = ScriptedClient(session)
    _bind_client(entity, scripted_client)

    mic = MicStream()
    result_future = asyncio.Future()

    run_task = asyncio.create_task(
        entity._async_run_audio_stream_sdk(
            _metadata(),
            mic.chunks(),
            "k",
            "m",
            "v",
            "",
            True,
            False,
            False,
            True,
            result_future,
            "conversation-1",
            None,
        )
    )

    mic.put(MIC_CHUNK)
    mic.put(MIC_CHUNK)
    await asyncio.wait_for(session.reply_started.wait(), 5)
    await asyncio.sleep(0.1)
    baseline = len(session.sent_audio)
    assert baseline > 0

    # Extra microphone chunks must keep flowing to the provider even though
    # the model already started replying.
    for _ in range(EXTRA_MIC_CHUNKS):
        mic.put(MIC_CHUNK)
    await _wait_until(lambda: len(session.sent_audio) >= baseline + EXTRA_MIC_CHUNKS)

    # Let the provider interrupt the response and emit its replacement.
    session.release_gate.set()
    result = await asyncio.wait_for(run_task, 15)
    mic.close()
    assert result.text
    assert result.result is SpeechResultState.SUCCESS

    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    assert turn.assistant_text_stream is not None

    chunks = [chunk async for chunk in turn.audio.async_chunks()]
    assert chunks == [resample_24k_to_16k(AUDIO_B)]
    assert resample_24k_to_16k(AUDIO_A) != resample_24k_to_16k(AUDIO_B)
    assert scripted_client.captured_config.support_barge_in is True
    assert scripted_client.captured_config.transcribe_output is True


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_legacy_mode_stops_microphone_forwarding_after_reply(
    entity_class,
) -> None:
    hass = FakeHass()
    entry_data = {
        "api_key": "k",
        CONF_SUPPORT_BARGE_IN: False,
    }
    entity, _session_manager, turn_store = _make_entity(hass, entry_data, entity_class)
    session = ScriptedSession(support_barge_in=False)
    scripted_client = ScriptedClient(session)
    _bind_client(entity, scripted_client)

    mic = MicStream()
    result_future = asyncio.Future()

    run_task = asyncio.create_task(
        entity._async_run_audio_stream_sdk(
            _metadata(),
            mic.chunks(),
            "k",
            "m",
            "v",
            "",
            False,
            False,
            False,
            False,
            result_future,
            "conversation-1",
            None,
        )
    )

    mic.put(MIC_CHUNK)
    mic.put(MIC_CHUNK)
    await asyncio.wait_for(session.reply_started.wait(), 5)
    await asyncio.sleep(0.1)
    baseline = len(session.sent_audio)
    assert baseline > 0

    # Legacy behaviour: chunks arriving after the reply must not be forwarded.
    for _ in range(EXTRA_MIC_CHUNKS):
        mic.put(MIC_CHUNK)
    await asyncio.sleep(0.1)
    assert len(session.sent_audio) == baseline

    session.release_gate.set()
    result = await asyncio.wait_for(run_task, 15)
    mic.close()
    assert result.result is SpeechResultState.SUCCESS

    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    chunks = [chunk async for chunk in turn.audio.async_chunks()]
    assert chunks == [resample_24k_to_16k(AUDIO_A)]
    assert "transcript" not in turn.assistant_text
    assert scripted_client.captured_config.support_barge_in is False
    assert scripted_client.captured_config.transcribe_output is False


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_barge_in_drops_provider_transcript_when_transcription_disabled(
    entity_class,
) -> None:
    """Do not expose an unsolicited provider transcript when disabled."""
    hass = FakeHass()
    transcribe_key = (
        CONF_TRANSCRIBE_GPT
        if entity_class is GPTRealtimeSTT
        else CONF_TRANSCRIBE_GEMINI
    )
    entry_data = {
        "api_key": "k",
        transcribe_key: False,
        CONF_SUPPORT_BARGE_IN: True,
    }
    entity, _session_manager, turn_store = _make_entity(hass, entry_data, entity_class)
    session = ScriptedSession(support_barge_in=True)
    scripted_client = ScriptedClient(session)
    _bind_client(entity, scripted_client)

    mic = MicStream()
    process_task = asyncio.create_task(
        entity.async_process_audio_stream(_metadata(), mic.chunks())
    )

    mic.put(MIC_CHUNK)
    await asyncio.wait_for(session.reply_started.wait(), 5)
    result = await asyncio.wait_for(process_task, 15)

    assert scripted_client.captured_config.support_barge_in is True
    assert scripted_client.captured_config.transcribe_output is False
    assert result.text == "user request"

    session.release_gate.set()
    mic.close()
    for task in list(hass.background_tasks):
        await asyncio.wait_for(task, 15)

    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    assert turn.assistant_text_stream is None
    assert "transcript" not in turn.assistant_text
