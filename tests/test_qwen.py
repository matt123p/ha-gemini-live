"""Qwen protocol fixtures and provider routing, without network or API credits."""

import base64
from collections import deque
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import WSMsgType
import pytest

from gemini_live import config_flow, conversation, stt, tts
from gemini_live.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_TRANSCRIBE_QWEN,
    CONF_VOICE,
    CONF_AFFECTIVE_DIALOG,
    CONF_SEARCH_GROUNDING,
    CONF_ENABLE_SOUND_RESPONSES,
    QWEN_AVAILABLE_MODELS,
    QWEN_DEFAULT_MODEL,
    QWEN_REALTIME_URL,
    PROVIDER_QWEN,
    qwen_supports_tools,
    QWEN_DEFAULT_VOICE,
)
from gemini_live.live import LiveConfig, LiveTool, LiveToolResponse
from gemini_live.qwen import QwenRealtimeClient, QwenRealtimeError, QwenRealtimeSession


class Socket:
    def __init__(self, events=()):
        self.events = deque(events)
        self.sent = []
        self.closed = False

    async def send_json(self, event):
        self.sent.append(event)

    async def receive(self):
        if not self.events:
            return SimpleNamespace(type=WSMsgType.CLOSED)
        return SimpleNamespace(
            type=WSMsgType.TEXT, data=json.dumps(self.events.popleft())
        )

    async def close(self):
        self.closed = True


def config(model=QWEN_DEFAULT_MODEL, **kwargs):
    return LiveConfig(
        model=model,
        voice=QWEN_DEFAULT_VOICE,
        system_instruction="Be helpful.",
        **kwargs,
    )


async def configured(model=QWEN_DEFAULT_MODEL, events=(), **kwargs):
    socket = Socket([{"type": "session.created"}, {"type": "session.updated"}, *events])
    session = QwenRealtimeSession(socket)
    await session.async_configure(config(model, **kwargs))
    return session, socket


async def collect(session):
    events = []
    async for event in session.receive():
        events.append(event)
        if event.turn_complete:
            break
    return events


@pytest.mark.parametrize("model", QWEN_AVAILABLE_MODELS)
async def test_model_configuration_uses_its_wire_format_and_capabilities(model):
    tool = LiveTool(
        name="turn_on",
        description="Turn on a light",
        parameters={"type": "object", "properties": {}},
    )
    session, socket = await configured(model, tools=[tool])
    settings = socket.sent[0]["session"]
    assert settings["turn_detection"] is None
    assert settings["modalities"] == ["text", "audio"]
    assert settings["input_audio_transcription"]["model"] == "qwen3-asr-flash-realtime"
    assert "tool_choice" not in settings and "output_modalities" not in settings
    if model.startswith("qwen3.8-"):
        assert settings["audio"]["input"]["format"]["sample_rate"] == 16000
        assert settings["audio"]["output"]["format"]["sample_rate"] == 24000
        assert settings["audio"]["output"]["voice"] == "Tina"
        assert "voice" not in settings
    else:
        assert (
            settings["input_audio_format"] == settings["output_audio_format"] == "pcm"
        )
    if qwen_supports_tools(model):
        assert settings["tools"][0] == {
            "type": "function",
            "function": {
                "name": "turn_on",
                "description": "Turn on a light",
                "parameters": tool.parameters,
            },
        }
    else:
        assert "tools" not in settings
        assert "cannot call tools" in settings["instructions"]
    assert session.is_open


async def test_connection_authentication_setup_and_cleanup():
    socket = Socket([{"type": "session.updated"}])
    http = SimpleNamespace(ws_connect=AsyncMock(return_value=socket))
    async with QwenRealtimeClient(http, "test-key").connect(config()) as session:
        assert session.is_open
    assert socket.closed
    args = http.ws_connect.call_args
    assert args.args == (QWEN_REALTIME_URL,)
    assert args.kwargs["params"] == {"model": QWEN_DEFAULT_MODEL}
    assert args.kwargs["headers"] == {"Authorization": "Bearer test-key"}


async def test_setup_failure_closes_socket():
    socket = Socket([{"type": "error", "error": {"message": "Invalid API key"}}])
    http = SimpleNamespace(ws_connect=AsyncMock(return_value=socket))
    with pytest.raises(QwenRealtimeError, match="Invalid API key"):
        async with QwenRealtimeClient(http, "bad-key").connect(config()):
            pytest.fail("A rejected session must not become ready")
    assert socket.closed


async def test_audio_is_not_resampled_and_manual_turn_commits_once():
    session, socket = await configured()
    audio = b"\x00\x00\x01\x00" * 160
    await session.send_audio(audio)
    assert base64.b64decode(socket.sent[-1]["audio"]) == audio
    await session.end_audio()
    assert [event["type"] for event in socket.sent[-2:]] == [
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert len({event["event_id"] for event in socket.sent}) == len(socket.sent)


async def test_home_assistant_union_tool_schema_is_sent_as_an_object():
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
            {
                "type": "object",
                "properties": {"area": {"type": "string"}},
                "required": ["area"],
            },
        ]
    }
    _, socket = await configured(
        tools=[LiveTool(name="turn_on", description="Turn on", parameters=schema)]
    )
    parameters = socket.sent[0]["session"]["tools"][0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == {"entity_id", "area"}
    assert "anyOf" not in parameters and "required" not in parameters
    assert "anyOf" in schema  # Never mutate Home Assistant's declaration.


async def test_text_and_tool_results_trigger_responses():
    session, socket = await configured()
    await session.send_text("Turn the light on")
    assert socket.sent[-2]["item"]["content"] == [
        {"type": "input_text", "text": "Turn the light on"}
    ]
    await session.send_tool_responses(
        [LiveToolResponse(name="turn_on", call_id="call-1", response={"success": True})]
    )
    assert socket.sent[-2]["item"] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"success":true}',
    }
    assert socket.sent[-1]["type"] == "response.create"


async def test_normalizes_audio_asr_and_transcripts_without_duplicate_final_text():
    audio = b"\x00\x01"
    session, _ = await configured(
        events=[
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hello",
            },
            {"type": "response.audio.delta", "delta": base64.b64encode(audio).decode()},
            {"type": "response.audio_transcript.delta", "delta": "Hi"},
            {"type": "response.audio_transcript.done", "transcript": "Hi"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    events = await collect(session)
    assert [event.input_transcript for event in events if event.input_transcript] == [
        "Hello"
    ]
    assert [event.output_transcript for event in events if event.output_transcript] == [
        "Hi"
    ]
    assert [event.audio for event in events if event.audio] == [audio]


async def test_output_transcription_can_be_disabled():
    session, _ = await configured(
        transcribe_output=False,
        events=[
            {"type": "response.audio_transcript.delta", "delta": "Hi"},
            {"type": "response.audio_transcript.done", "transcript": "Hi"},
            {"type": "response.done"},
        ],
    )
    assert not any(event.output_transcript for event in await collect(session))


async def test_tool_call_is_dispatched_once_at_response_done_before_followup():
    call = {
        "type": "function_call",
        "name": "turn_on",
        "call_id": "call-1",
        "arguments": '{"entity_id":"light.kitchen"}',
    }
    session, socket = await configured(
        events=[
            {**call, "type": "response.function_call_arguments.done"},
            {"type": "response.output_item.done", "item": call},
            {"type": "response.done", "response": {"output": [call]}},
            {"type": "response.created"},
            {"type": "response.text.delta", "delta": "Done"},
            {"type": "response.done"},
        ]
    )
    stream = session.receive()
    event = await anext(stream)
    assert len(event.tool_calls) == 1 and not event.turn_complete
    assert event.tool_calls[0].arguments == {"entity_id": "light.kitchen"}
    await session.send_tool_responses(
        [LiveToolResponse(name="turn_on", call_id="call-1", response=True)]
    )
    rest = [event async for event in take_until_complete(stream)]
    assert not any(event.tool_calls for event in rest)
    assert rest[-1].turn_complete


async def take_until_complete(stream):
    async for event in stream:
        yield event
        if event.turn_complete:
            break


async def test_barge_in_discards_old_audio_without_completing_the_turn():
    session, socket = await configured(
        support_barge_in=True,
        events=[
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"old").decode(),
            },
            {"type": "response.done", "response": {"status": "cancelled"}},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"new").decode(),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ],
    )
    assert socket.sent[0]["session"]["turn_detection"]["type"] == "semantic_vad"
    before = len(socket.sent)
    await session.end_audio()
    assert len(socket.sent) == before
    events = await collect(session)
    assert sum(event.interrupted for event in events) == 1
    assert [event.audio for event in events if event.audio] == [b"new"]
    assert sum(event.turn_complete for event in events) == 1


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "error": {"code": "rate_limit"}},
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"message": "ASR failed"},
        },
        {"type": "response.audio.delta", "delta": "not-base64"},
        {
            "type": "response.function_call_arguments.done",
            "name": "bad",
            "call_id": "1",
            "arguments": "[1]",
        },
        {"type": "response.done", "response": {"status": "failed"}},
        {"type": "response.done", "response": {"status": "incomplete"}},
    ],
)
async def test_protocol_errors_fail_the_turn(event):
    session, _ = await configured(events=[event])
    with pytest.raises(QwenRealtimeError):
        await anext(session.receive())


@pytest.mark.parametrize(
    "module, expected",
    [
        (stt, stt.QwenRealtimeSTT),
        (tts, tts.QwenRealtimeTTS),
        (conversation, conversation.QwenRealtimeConversationAgent),
    ],
)
async def test_provider_routes_each_pipeline_platform(module, expected):
    entities = []
    hass = SimpleNamespace()
    entry = SimpleNamespace(
        entry_id="qwen",
        data={CONF_PROVIDER: PROVIDER_QWEN, CONF_MODEL: QWEN_DEFAULT_MODEL},
        options={},
    )
    await module.async_setup_entry(hass, entry, entities.extend)
    assert len(entities) == 1 and isinstance(entities[0], expected)
    assert "en" in entities[0].supported_languages


@pytest.mark.parametrize("model", QWEN_AVAILABLE_MODELS)
def test_config_schema_exposes_only_supported_qwen_settings(model):
    schema = config_flow._provider_schema(PROVIDER_QWEN, {CONF_MODEL: model})
    keys = {marker.schema for marker in schema.schema}
    assert CONF_TRANSCRIBE_QWEN in keys
    assert CONF_AFFECTIVE_DIALOG not in keys
    assert CONF_SEARCH_GROUNDING not in keys
    assert (CONF_ENABLE_SOUND_RESPONSES in keys) == qwen_supports_tools(model)
    value = schema(
        {CONF_API_KEY: "key", CONF_MODEL: model, CONF_VOICE: QWEN_DEFAULT_VOICE}
    )
    assert value[CONF_MODEL] == model


@pytest.mark.parametrize(
    "model", ["qwen-omni-turbo-realtime", "qwen3-omni-flash-realtime-2025-09-15"]
)
async def test_models_without_function_calling_are_rejected(model):
    socket = Socket()
    with pytest.raises(QwenRealtimeError, match="function-calling"):
        await QwenRealtimeSession(socket).async_configure(config(model))
    assert not socket.sent
