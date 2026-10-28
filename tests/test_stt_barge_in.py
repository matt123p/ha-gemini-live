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
    CONF_ENABLE_SOUND_RESPONSES,
    CONF_RESPONSE_TIMEOUT,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    DOMAIN,
    GEMINI_LIVE_TTS_PLACEHOLDER,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
)
from gemini_live.live import LiveConfig, LiveEvent, LiveToolCall
from gemini_live.sounds import PLAY_SOUND_TOOL, load_sound_pcm
from gemini_live.runtime import LiveSessionManager, TurnStore
from gemini_live.stt import (
    GeminiLiveSTT,
    GPTRealtimeSTT,
    _add_search_tool_instruction,
)
from gemini_live.utils import PCM24kTo16kStreamResampler, resample_24k_to_16k
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context
from homeassistant.helpers.issue_registry import DATA_REGISTRY as DATA_ISSUE_REGISTRY

AUDIO_A = b"\x01\x00" * 12
AUDIO_B = b"\x02\x00" * 12
MIC_CHUNK = b"\x00\x00" * 3200
EXTRA_MIC_CHUNKS = 2

ENTITY_CLASSES = [GeminiLiveSTT, GPTRealtimeSTT]


@pytest.fixture(autouse=True)
def _core_supports_interruption(monkeypatch: pytest.MonkeyPatch):
    """Pretend Core provides the complete TTS interruption path."""
    monkeypatch.setattr("gemini_live.stt.supports_tts_interruption", lambda: True)


def test_native_search_grounding_adds_search_instruction_without_assist_tool():
    instruction = _add_search_tool_instruction(
        "base",
        [],
        True,
        native_search_grounding=True,
    )

    assert instruction.startswith("base\n\nYou MUST use")


def test_exposed_search_instruction_still_requires_search_like_tool():
    assert _add_search_tool_instruction("base", [], True) == "base"
    instruction = _add_search_tool_instruction(
        "base",
        [SimpleNamespace(name="search_google")],
        True,
    )
    assert instruction.startswith("base\n\nYou MUST use")


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

    def __init__(
        self,
        support_barge_in: bool,
        include_input_transcript: bool = True,
    ) -> None:
        self.support_barge_in = support_barge_in
        self.include_input_transcript = include_input_transcript
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
            if self.include_input_transcript:
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
            # A real session stays open for further user activity until the
            # pipeline closes the microphone stream.
            await asyncio.Event().wait()
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
        self.connect_count = 0
        self.close_count = 0

    def connect(self, config: LiveConfig):
        self.captured_config = config
        self.connect_count += 1

        class _Connect:
            async def __aenter__(self_inner):
                return self._session

            async def __aexit__(self_inner, *_exc):
                self.close_count += 1
                return None

        return _Connect()


class SilentSession(ScriptedSession):
    """A LiveSession that never recognizes speech or produces a response."""

    async def receive(self):
        self.reply_started.set()
        await asyncio.Event().wait()
        yield LiveEvent()  # pragma: no cover - makes this an async generator


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
async def test_unsupported_barge_in_uses_external_vad(
    entity_class,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gemini_live.stt.supports_tts_interruption", lambda: False)
    entity, _session_manager, _turn_store = _make_entity(
        FakeHass(),
        {
            "api_key": "k",
            CONF_SUPPORT_BARGE_IN: True,
        },
        entity_class,
    )

    assert entity._effective_support_barge_in() is False
    assert entity.audio_processing.requires_external_vad is True


@pytest.mark.parametrize("barge_in", [False, True])
@pytest.mark.parametrize("transcribe_output", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_sound_tool_plays_cue_without_model_speech(
    barge_in, transcribe_output, enabled
):
    hass = FakeHass()
    entity, manager, turns = _make_entity(
        hass, {"api_key": "k", CONF_ENABLE_SOUND_RESPONSES: enabled}, GeminiLiveSTT
    )

    class SoundSession(ScriptedSession):
        def __init__(self):
            super().__init__(support_barge_in=barge_in)
            self.tool_responses = []

        async def send_tool_responses(self, responses):
            self.tool_responses.extend(responses)

        async def receive(self):
            yield LiveEvent(input_transcript="Please just beep")
            yield LiveEvent(
                tool_calls=[LiveToolCall("play_sound", "cue-1", {})]
            )
            yield LiveEvent(
                tool_calls=[LiveToolCall("play_sound", "cue-2", {})]
            )
            # Even if the model ignores the instruction, its speech must not
            # be mixed with the selected cue or added to the transcript.
            yield LiveEvent(
                audio=AUDIO_A,
                text="Spoken acknowledgement",
                output_transcript="Spoken acknowledgement",
            )
            yield LiveEvent(turn_complete=True)
            if barge_in:
                await asyncio.Event().wait()

    session = SoundSession()
    client = ScriptedClient(session)
    _bind_client(entity, client)
    mic = MicStream()
    result_future = asyncio.Future()
    task = asyncio.create_task(
        entity._async_run_audio_stream_sdk(
            _metadata(),
            mic.chunks(),
            "k",
            "native-audio-test",
            "v",
            "",
            transcribe_output,
            False,
            barge_in,
            result_future,
            "conversation-1",
            None,
        )
    )
    consumer = None
    try:
        mic.put(MIC_CHUNK)
        if not barge_in:
            mic.close()
        result = await asyncio.wait_for(result_future, 5)
        assert result.text == "Please just beep"
        turn = turns.take_voice_turn("conversation-1", result.text)
        assert turn is not None

        async def consume():
            return b"".join([chunk async for chunk in turn.audio.async_chunks()])

        consumer = asyncio.create_task(consume())
        expected_audio = (
            load_sound_pcm() if enabled else resample_24k_to_16k(AUDIO_A)
        )
        assert await asyncio.wait_for(consumer, 5) == expected_audio
        if transcribe_output:
            assert turn.assistant_text_stream.text == (
                "[Sound: done]"
                if enabled
                else "Spoken acknowledgementSpoken acknowledgement"
            )
        assert session.tool_responses[0].call_id == "cue-1"
        assert ("success" in session.tool_responses[0].response) is enabled
        assert "error" in session.tool_responses[1].response
        assert (
            any(tool.name == "play_sound" for tool in client.captured_config.tools)
            is enabled
        )
        assert ("play_sound" in client.captured_config.system_instruction) is enabled
        assert manager.should_continue_conversation("conversation-1") is not enabled
        if barge_in:
            assert await manager.async_prepare_for_new_turn("conversation-1")
            assert client.close_count == int(enabled)
        else:
            await asyncio.wait_for(task, 5)
    finally:
        task.cancel()
        if consumer is not None:
            consumer.cancel()
        await asyncio.gather(
            task, *([consumer] if consumer else []), return_exceptions=True
        )
        await manager.async_close_all()


@pytest.mark.parametrize("arguments", [{}, {"sound": "bad"}])
@pytest.mark.parametrize("end_conversation", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_typed_sound_response_and_invalid_sound_fallback(
    arguments, end_conversation, enabled
):
    from gemini_live.conversation import GeminiLiveConversationAgent
    from gemini_live.utils import pcm_to_wav

    hass = FakeHass()
    entity, manager, turns = _make_entity(
        hass,
        {
            "api_key": "k",
            "model": "native-audio-test",
            "voice": "v",
            CONF_ENABLE_SOUND_RESPONSES: enabled,
        },
        GeminiLiveSTT,
    )
    agent = GeminiLiveConversationAgent(hass, entity.entry)

    class TextSession(ScriptedSession):
        def __init__(self):
            super().__init__(support_barge_in=False)
            self.tool_responses = []

        async def send_text(self, text):
            assert text == "Just beep"

        async def send_tool_responses(self, responses):
            self.tool_responses.extend(responses)

        async def receive(self):
            calls = [LiveToolCall("play_sound", "cue-1", arguments)]
            if end_conversation:
                calls.append(LiveToolCall("end_conversation", "end-1", {}))
            yield LiveEvent(tool_calls=calls)
            yield LiveEvent(audio=AUDIO_A, output_transcript="Fallback speech")
            yield LiveEvent(turn_complete=True)

    session = TextSession()
    _bind_client(agent, ScriptedClient(session))

    async def fake_llm_api(_context):
        return None, [PLAY_SOUND_TOOL], "Test instruction"

    agent._async_get_llm_api = fake_llm_api
    user_input = SimpleNamespace(as_llm_context=lambda _domain: None)
    try:
        response = await agent._async_process_text_live(
            "Just beep", user_input, "conversation-1"
        )
        if not arguments and enabled:
            assert response == "[Sound: done]"
            assert turns.take_audio(response) == pcm_to_wav(load_sound_pcm())
            assert session.tool_responses[0].response["success"]
        else:
            assert response == "Fallback speech"
            assert turns.take_audio(response) == pcm_to_wav(
                resample_24k_to_16k(AUDIO_A)
            )
            assert "error" in session.tool_responses[0].response
        assert (
            manager.should_continue_conversation("conversation-1")
            is (not end_conversation and not (enabled and not arguments))
        )
    finally:
        await manager.async_close_all()


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
@pytest.mark.parametrize("end_state", ["active", "idle", "idle_timeout"])
async def test_follow_up_reuses_only_completed_barge_in_session(
    entity_class, end_state: str
) -> None:
    completed = end_state != "active"
    hass = FakeHass()
    entity, manager, turn_store = _make_entity(
        hass,
        {
            "api_key": "k",
            CONF_SUPPORT_BARGE_IN: True,
            CONF_RESPONSE_TIMEOUT: 1 if end_state == "idle_timeout" else 30,
        },
        entity_class,
    )
    session = ScriptedSession(support_barge_in=True)
    client = ScriptedClient(session)
    _bind_client(entity, client)
    mic = MicStream()
    result_future = asyncio.Future()
    task = asyncio.create_task(
        entity._async_run_audio_stream_sdk(
            _metadata(),
            mic.chunks(),
            "k",
            "m",
            "v",
            "",
            True,
            False,
            True,
            result_future,
            "conversation-1",
            None,
        )
    )
    try:
        mic.put(MIC_CHUNK)
        result = await asyncio.wait_for(result_future, 5)
        turn = turn_store.take_voice_turn("conversation-1", result.text)
        assert turn is not None
        if completed:
            session.release_gate.set()
            await _wait_until(lambda: manager.is_turn_complete("conversation-1"))
        assert not task.done()

        # This is the next STT turn's handoff while the old microphone is
        # still open, exactly as in the satellite logs.
        if end_state == "idle_timeout":
            await asyncio.wait_for(task, 5)
            assert not await manager.async_prepare_for_new_turn("conversation-1")
        else:
            assert await manager.async_prepare_for_new_turn("conversation-1")
            assert task.cancelled()
        assert client.close_count == (0 if completed else 1)
        async with manager.acquire("conversation-1", client, client.captured_config):
            assert client.connect_count == (1 if completed else 2)
            assert not manager.is_turn_complete("conversation-1")
        assert manager.should_continue_conversation("conversation-1")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await manager.async_close_all()


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
    await asyncio.sleep(0.1)

    # The provider turn is complete now, but the pipeline is still open:
    # microphone chunks must keep flowing so the user can speak again.
    after_reply = len(session.sent_audio)
    for _ in range(EXTRA_MIC_CHUNKS):
        mic.put(MIC_CHUNK)
    await _wait_until(lambda: len(session.sent_audio) >= after_reply + EXTRA_MIC_CHUNKS)

    # The pipeline closing the microphone stream is what ends the turn.
    mic.close()
    result = await asyncio.wait_for(run_task, 15)
    assert session.end_audio_count == 1
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
async def test_non_barge_in_keeps_forwarding_until_external_vad_closes_stream(
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

    # Home Assistant must keep consuming the microphone generator until its
    # external VAD closes it. Otherwise remote satellites never receive the
    # STT_VAD_END event that tells them to stop streaming.
    for _ in range(EXTRA_MIC_CHUNKS):
        mic.put(MIC_CHUNK)
    await _wait_until(lambda: len(session.sent_audio) >= baseline + EXTRA_MIC_CHUNKS)
    assert not result_future.done()

    mic.close()
    await _wait_until(lambda: session.end_audio_count == 1)
    await asyncio.wait_for(result_future, 5)
    session.release_gate.set()
    result = await asyncio.wait_for(run_task, 15)
    assert result.result is SpeechResultState.SUCCESS

    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    chunks = [chunk async for chunk in turn.audio.async_chunks()]
    assert chunks == [resample_24k_to_16k(AUDIO_A)]
    assert "transcript" not in turn.assistant_text
    assert scripted_client.captured_config.support_barge_in is False
    assert scripted_client.captured_config.transcribe_output is False


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_audio_tool_context_preserves_pipeline_provenance(
    entity_class,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = FakeHass()
    entity, _session_manager, _turn_store = _make_entity(
        hass,
        {
            "api_key": "k",
            CONF_LLM_HASS_API: ["assist", "memory"],
        },
        entity_class,
    )
    session = ScriptedSession(support_barge_in=False)
    _bind_client(entity, ScriptedClient(session))
    source_context = Context(user_id="voice-user", parent_id="parent-context")
    captured_contexts = []
    captured_api_ids = []

    async def fake_async_get_api(**kwargs):
        captured_contexts.append(kwargs["llm_context"].context)
        captured_api_ids.append(kwargs["api_id"])
        return SimpleNamespace(tools=[], api_prompt="", custom_serializer=None)

    monkeypatch.setattr("gemini_live.stt.llm.async_get_api", fake_async_get_api)

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
            result_future,
            "conversation-1",
            "device-1",
            source_context,
        )
    )

    mic.put(MIC_CHUNK)
    await asyncio.wait_for(session.reply_started.wait(), 5)
    mic.close()
    session.release_gate.set()
    await asyncio.wait_for(run_task, 15)

    assert captured_contexts == [source_context]
    assert captured_api_ids == [["assist", "memory"]]


async def test_audio_entry_point_forwards_resolved_pipeline_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = FakeHass()
    entity, _session_manager, _turn_store = _make_entity(
        hass,
        {"api_key": "k"},
        GeminiLiveSTT,
    )
    source_context = Context(user_id="voice-user")
    captured_contexts = []

    monkeypatch.setattr(
        "gemini_live.stt.active_pipeline_context",
        lambda *_args, **_kwargs: (
            "conversation-1",
            "device-1",
            source_context,
        ),
    )

    async def fake_run(*args):
        captured_contexts.append(args[-1])
        result = SpeechResult("user request", SpeechResultState.SUCCESS)
        args[9].set_result(result)
        return result

    monkeypatch.setattr(entity, "_async_run_audio_stream_sdk", fake_run)

    result = await entity._async_process_audio_stream_sdk(
        _metadata(),
        MicStream().chunks(),
        "k",
        "m",
        "v",
        "",
        False,
        False,
        False,
    )

    assert result.text == "user request"
    assert captured_contexts == [source_context]


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_barge_in_drops_provider_transcript_when_transcription_disabled(
    entity_class,
    monkeypatch: pytest.MonkeyPatch,
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

    # The full entry point resolves the pipeline conversation ID through Home
    # Assistant; pin it so the published turn can be looked up below.
    monkeypatch.setattr(
        "gemini_live.stt.active_pipeline_context",
        lambda *_args, **_kwargs: ("conversation-1", None, None),
    )

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


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_placeholder_transcript_carries_unique_turn_id(
    entity_class,
) -> None:
    """A missing input transcript must still yield a unique STT output.

    Home Assistant caches TTS audio by message, so the bare placeholder
    would make later turns replay the first turn's audio. The STT transcript
    must also match the published voice turn's user_text exactly so the
    conversation agent finds it.
    """
    hass = FakeHass()
    entry_data = {
        "api_key": "k",
        CONF_SUPPORT_BARGE_IN: True,
    }
    entity, _session_manager, turn_store = _make_entity(hass, entry_data, entity_class)
    session = ScriptedSession(
        support_barge_in=True,
        include_input_transcript=False,
    )
    scripted_client = ScriptedClient(session)
    _bind_client(entity, scripted_client)

    mic = MicStream()
    result_future: asyncio.Future = asyncio.Future()

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
            True,
            result_future,
            "conversation-1",
            None,
        )
    )

    mic.put(MIC_CHUNK)
    await asyncio.wait_for(session.reply_started.wait(), 5)
    mic.close()
    result = await asyncio.wait_for(run_task, 15)

    assert result.text.startswith(GEMINI_LIVE_TTS_PLACEHOLDER)
    assert result.text != GEMINI_LIVE_TTS_PLACEHOLDER

    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    assert turn.assistant_text.startswith(GEMINI_LIVE_TTS_PLACEHOLDER)
    assert turn.assistant_text != GEMINI_LIVE_TTS_PLACEHOLDER


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
@pytest.mark.parametrize("close_silent_mic", [False, True])
async def test_speech_after_silence_timeout_gets_pending_response(
    entity_class, close_silent_mic
) -> None:
    """A silent turn must release its session and allow a real reply next time."""
    hass = FakeHass()
    entity, manager, turns = _make_entity(
        hass,
        {
            "api_key": "k",
            CONF_SUPPORT_BARGE_IN: True,
            CONF_RESPONSE_TIMEOUT: 0.05,
        },
        entity_class,
    )
    silent_client = ScriptedClient(SilentSession(support_barge_in=True))
    _bind_client(entity, silent_client)
    silent_mic = MicStream()
    silent_mic.put(MIC_CHUNK)
    if close_silent_mic:
        silent_mic.close()

    async def run(mic):
        return await entity._async_run_audio_stream_sdk(
            _metadata(), mic.chunks(), "k", "m", "v", "", True, False,
            True, asyncio.Future(), "conversation-1", None,
        )

    silent_result = await asyncio.wait_for(run(silent_mic), 5)
    silent_turn = turns.take_voice_turn("conversation-1", silent_result.text)
    assert silent_turn is not None and silent_turn.complete_conversation
    assert silent_client.close_count == 1

    class ReplyAfterAudioEnd(ScriptedSession):
        async def end_audio(self):
            await super().end_audio()
            self.release_gate.set()

        async def receive(self):
            # The provider processes the final audio after the sender ends.
            await self.release_gate.wait()
            await asyncio.sleep(0.01)
            yield LiveEvent(input_transcript="What's the time?")
            yield LiveEvent(audio=AUDIO_A)
            yield LiveEvent(turn_complete=True)

    reply = ReplyAfterAudioEnd(support_barge_in=True)
    reply_client = ScriptedClient(reply)
    _bind_client(entity, reply_client)
    entity.entry.options[CONF_RESPONSE_TIMEOUT] = 1
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    try:
        result = await asyncio.wait_for(run(mic), 5)
        assert result.result is SpeechResultState.SUCCESS
        assert result.text == "What's the time?"
        turn = turns.take_voice_turn("conversation-1", result.text)
        assert turn is not None and not turn.complete_conversation
        assert b"".join([chunk async for chunk in turn.audio.async_chunks()])
        assert reply_client.connect_count == 1
        assert manager.should_continue_conversation("conversation-1")
    finally:
        await manager.async_close_all()


@pytest.mark.parametrize("entity_class", ENTITY_CLASSES)
async def test_unrecognized_speech_closes_pipeline_silently(entity_class) -> None:
    """A provider turn with no usable content must not surface an STT error."""
    hass = FakeHass()
    entity, session_manager, turn_store = _make_entity(
        hass,
        {
            "api_key": "k",
            CONF_SUPPORT_BARGE_IN: True,
            CONF_RESPONSE_TIMEOUT: 0.05,
        },
        entity_class,
    )
    session = SilentSession(support_barge_in=True)
    client = ScriptedClient(session)
    _bind_client(entity, client)
    mic = MicStream()
    result_future: asyncio.Future = asyncio.Future()

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
            True,
            result_future,
            "conversation-1",
            None,
        )
    )

    mic.put(MIC_CHUNK)
    result = await asyncio.wait_for(run_task, 5)

    assert result.result is SpeechResultState.SUCCESS
    assert session.sent_audio
    assert client.close_count == 1
    assert result.text
    assert not session_manager.should_continue_conversation("conversation-1")
    turn = turn_store.take_voice_turn("conversation-1", result.text)
    assert turn is not None
    assert turn.assistant_text == ""
    assert turn.audio == b""
    assert turn.complete_conversation is True
