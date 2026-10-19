"""ElevenLabs Agents protocol fixtures and provider routing, without network or API credits."""

import asyncio
import base64
from collections import deque
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import WSMsgType
import pytest

import gemini_live
from gemini_live import config_flow, conversation, elevenlabs, stt, tts
from gemini_live.const import (
    CONF_API_KEY,
    CONF_ELEVENLABS_AGENT_ID,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_SUPPORT_BARGE_IN,
    CONF_TRANSCRIBE_ELEVENLABS,
    CONF_VOICE,
    ELEVENLABS_DEFAULT_MODEL,
    ELEVENLABS_DEFAULT_VOICE,
    PROVIDER_ELEVENLABS,
)
from gemini_live.elevenlabs import (
    ElevenLabsAgentsClient,
    ElevenLabsAgentsError,
    ElevenLabsAgentsSession,
    agent_base_config,
    strip_audio_tags,
    tool_config,
)
from gemini_live.live import LiveConfig, LiveTool, LiveToolResponse

METADATA = {
    "type": "conversation_initiation_metadata",
    "conversation_initiation_metadata_event": {
        "conversation_id": "conv_1",
        "user_input_audio_format": "pcm_16000",
        "agent_output_audio_format": "pcm_24000",
    },
}


class Socket:
    """A scripted WebSocket. With ``block`` it waits instead of closing."""

    def __init__(self, events=(), *, block=False):
        self.events = deque(events)
        self.sent = []
        self.closed = False
        self.block = block
        self._pushed = asyncio.Event()

    def push(self, *events):
        self.events.extend(events)
        self._pushed.set()

    async def send_json(self, event):
        self.sent.append(event)

    async def receive(self):
        while not self.events and self.block:
            self._pushed.clear()
            await self._pushed.wait()
        if not self.events:
            return SimpleNamespace(type=WSMsgType.CLOSED, extra=None)
        return SimpleNamespace(
            type=WSMsgType.TEXT, data=json.dumps(self.events.popleft()), extra=None
        )

    async def close(self):
        self.closed = True


class Response:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = {} if body is None else body
        self.content_type = "application/json"

    async def json(self):
        return self.body

    async def text(self):
        return json.dumps(self.body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Http:
    """Record ElevenLabs REST calls and answer from a route table."""

    def __init__(self, routes=None, socket=None):
        self.routes = routes or {}
        self.calls = []
        self.ws_connect = AsyncMock(return_value=socket)
        self._ids = 0

    def request(self, method, url, *, headers, json=None, params=None):
        path = url.removeprefix(elevenlabs.API_URL)
        self.calls.append((method, path, json, params, headers))
        handler = self.routes.get((method, path))
        if handler is None and (method, path) == ("post", "/v1/convai/tools"):
            self._ids += 1
            return Response(body={"id": f"tool_{self._ids}"})
        if handler is None:
            return Response()
        return handler(json, params)


def audio(data: bytes, chars: str = "") -> dict:
    event = {"audio_base_64": base64.b64encode(data).decode()}
    if chars:
        event["alignment"] = {"chars": list(chars)}
    return {"type": "audio", "audio_event": event}


def agent_response(text: str) -> dict:
    return {"type": "agent_response", "agent_response_event": {"agent_response": text}}


def tool_call(name="HassTurnOff", call_id="call_1", **parameters) -> dict:
    return {
        "type": "client_tool_call",
        "client_tool_call": {
            "tool_name": name,
            "tool_call_id": call_id,
            "parameters": parameters,
            "event_id": 1,
            "expects_response": True,
        },
    }


def config(**kwargs):
    return LiveConfig(
        model=ELEVENLABS_DEFAULT_MODEL,
        voice=ELEVENLABS_DEFAULT_VOICE,
        system_instruction="Be helpful.",
        **kwargs,
    )


async def configured(events=(), *, block=False, **kwargs):
    socket = Socket([METADATA, *events], block=block)
    session = ElevenLabsAgentsSession(socket)
    await session.async_configure(config(**kwargs))
    return session, socket


async def collect(session):
    events = []
    async for event in session.receive():
        events.append(event)
        if event.turn_complete:
            break
    return events


def test_agent_base_config_fixes_formats_events_and_overrides():
    agent = agent_base_config("de")
    conversation_config = agent["conversation_config"]
    assert conversation_config["asr"]["user_input_audio_format"] == "pcm_16000"
    assert conversation_config["tts"]["agent_output_audio_format"] == "pcm_24000"
    assert conversation_config["agent"] == {"first_message": "", "language": "de"}
    assert conversation_config["turn"]["turn_timeout"] == 4
    assert conversation_config["turn"]["initial_wait_time"] == 8
    assert {"audio", "user_transcript", "agent_response", "client_tool_call"} <= set(
        conversation_config["conversation"]["client_events"]
    )
    overrides = agent["platform_settings"]["overrides"]["conversation_config_override"]
    assert overrides["agent"]["prompt"] == {"prompt": True, "llm": True, "tool_ids": True}
    assert overrides["tts"] == {"voice_id": True}


async def test_configure_overrides_prompt_model_and_voice():
    socket = Socket([{"type": "ping", "ping_event": {"event_id": 7}}, METADATA])
    session = ElevenLabsAgentsSession(socket)
    await session.async_configure(config())
    assert socket.sent[0] == {
        "type": "conversation_initiation_client_data",
        "conversation_config_override": {
            "agent": {
                "prompt": {"prompt": "Be helpful.", "llm": ELEVENLABS_DEFAULT_MODEL}
            },
            "tts": {"voice_id": ELEVENLABS_DEFAULT_VOICE},
        },
    }
    assert socket.sent[1] == {"type": "pong", "event_id": 7}
    assert session.conversation_id == "conv_1"
    assert session.is_open


async def test_configure_rejects_an_agent_with_other_audio_formats():
    metadata = json.loads(json.dumps(METADATA))
    metadata["conversation_initiation_metadata_event"]["agent_output_audio_format"] = (
        "pcm_16000"
    )
    session = ElevenLabsAgentsSession(Socket([metadata]))
    with pytest.raises(ElevenLabsAgentsError, match="reconfigure"):
        await session.async_configure(config())


async def test_audio_is_forwarded_unchanged_and_text_becomes_user_message():
    session, socket = await configured()
    await session.send_audio(b"\x01\x02")
    await session.send_text("Hello")
    assert socket.sent[1:] == [
        {"user_audio_chunk": base64.b64encode(b"\x01\x02").decode()},
        {"type": "user_message", "text": "Hello"},
    ]


@pytest.mark.parametrize("text_first", [False, True])
async def test_turn_completes_when_audio_covers_the_response_text(text_first):
    text = "[happy] Hi there"
    chunks = [audio(b"one", "[happy] Hi"), audio(b"two", " there")]
    events = [agent_response(text), *chunks] if text_first else [*chunks, agent_response(text)]
    session, _ = await configured(
        [{"type": "user_transcript", "user_transcription_event": {"user_transcript": "Hi"}}, *events]
    )
    received = await collect(session)
    assert [event.audio for event in received if event.audio] == [b"one", b"two"]
    assert [event.input_transcript for event in received if event.input_transcript] == ["Hi"]
    assert [e.output_transcript for e in received if e.output_transcript] == ["Hi there"]
    assert received[-1].turn_complete


async def test_turn_completes_after_idle_audio_without_alignment(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_COMPLETION_IDLE_TIMEOUT", 0.01)
    session, _ = await configured(
        [agent_response("Done."), audio(b"pcm")], block=True
    )
    received = await asyncio.wait_for(collect(session), 1)
    assert [event.audio for event in received if event.audio] == [b"pcm"]
    assert received[-1].turn_complete


async def test_output_transcript_is_optional():
    session, _ = await configured(
        [audio(b"pcm", "Ok."), agent_response("Ok.")], transcribe_output=False
    )
    received = await collect(session)
    assert not any(event.output_transcript for event in received)
    assert received[-1].turn_complete


async def test_tool_call_round_trip_and_error_results():
    session, socket = await configured(
        [tool_call(area="Office"), audio(b"pcm", "Done."), agent_response("Done.")]
    )
    received = await collect(session)
    calls = received[0].tool_calls
    assert [(c.name, c.call_id, c.arguments) for c in calls] == [
        ("HassTurnOff", "call_1", {"area": "Office"})
    ]
    await session.send_tool_responses(
        [
            LiveToolResponse("HassTurnOff", "call_1", {"success": True}),
            LiveToolResponse("HassTurnOn", "call_2", {"error": "Unknown device"}),
        ]
    )
    assert socket.sent[-2:] == [
        {
            "type": "client_tool_result",
            "tool_call_id": "call_1",
            "result": '{"success":true}',
            "is_error": False,
        },
        {
            "type": "client_tool_result",
            "tool_call_id": "call_2",
            "result": '{"error":"Unknown device"}',
            "is_error": True,
        },
    ]
    assert received[-1].turn_complete


async def test_tool_call_resets_speech_from_the_previous_generation():
    session, _ = await configured(
        [
            agent_response("Let me check."),
            tool_call(),
            audio(b"pcm", "Done."),
            agent_response("Done."),
        ]
    )
    received = await collect(session)
    assert sum(event.turn_complete for event in received) == 1
    assert [event.audio for event in received if event.audio] == [b"pcm"]


async def test_pings_are_answered_while_receiving():
    session, socket = await configured(
        [{"type": "ping", "ping_event": {"event_id": 3}}, audio(b"a", "Ok"), agent_response("Ok")]
    )
    await collect(session)
    assert {"type": "pong", "event_id": 3} in socket.sent


async def test_interruption_restarts_completion_tracking():
    interruption = {"type": "interruption", "interruption_event": {"event_id": 2}}
    session, _ = await configured(
        [
            agent_response("Long reply"),
            audio(b"old", "Long"),
            interruption,
            audio(b"new", "Ok"),
            agent_response("Ok"),
        ]
    )
    received = await collect(session)
    assert not any(event.interrupted for event in received)
    assert [event.audio for event in received if event.audio] == [b"old", b"new"]
    assert sum(event.turn_complete for event in received) == 1


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "error": {"message": "quota exceeded"}},
        {"type": "audio", "audio_event": {"audio_base_64": "not-base64!"}},
        {"type": "client_tool_call", "client_tool_call": {"tool_name": "x", "parameters": []}},
    ],
)
async def test_protocol_errors_fail_the_turn(event):
    session, _ = await configured([event])
    with pytest.raises(ElevenLabsAgentsError):
        await anext(session.receive())


async def test_closed_connection_fails_the_turn():
    session, _ = await configured()
    with pytest.raises(ElevenLabsAgentsError, match="closed"):
        await anext(session.receive())


async def test_end_audio_feeds_silence_until_the_agent_responds(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SILENCE_INTERVAL", 0.001)
    socket = Socket([METADATA], block=True)
    session = ElevenLabsAgentsSession(socket)
    await session.async_configure(config())
    await session.end_audio()
    await asyncio.sleep(0.05)
    silence = [event for event in socket.sent if "user_audio_chunk" in event]
    assert silence and all(
        base64.b64decode(event["user_audio_chunk"]) == b"\x00" * 3200
        for event in silence
    )

    socket.push(
        {"type": "user_transcript", "user_transcription_event": {"user_transcript": "Hi"}}
    )
    receiver = session.receive()
    assert (await anext(receiver)).input_transcript == "Hi"
    sent = len(socket.sent)
    await asyncio.sleep(0.02)
    assert len(socket.sent) == sent
    await session.async_close()
    assert socket.closed


async def test_end_audio_after_a_response_sends_nothing():
    session, socket = await configured([tool_call()])
    await anext(session.receive())
    before = len(socket.sent)
    await session.end_audio()
    await asyncio.sleep(0)
    assert len(socket.sent) == before


async def test_trailing_silence_is_capped(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SILENCE_INTERVAL", 0.001)
    monkeypatch.setattr(elevenlabs, "_MAX_TRAILING_SILENCE", 0.005)
    session, socket = await configured(block=True)
    await session.end_audio()
    await asyncio.sleep(0.05)
    assert len([e for e in socket.sent if "user_audio_chunk" in e]) == 5


def test_strip_audio_tags():
    assert strip_audio_tags("[happy] Hi! [laughs softly] Ok.") == "Hi! Ok."
    assert strip_audio_tags("No tags [1, 2] here") == "No tags here"


def test_tool_parameters_follow_the_elevenlabs_schema_subset():
    tool = LiveTool(
        name="HassTurnOn",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "domain": {"type": "array", "items": {"type": "string", "enum": ["light", 3]}},
                "brightness": {"type": ["integer", "null"], "minimum": 0},
                "color": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                "options": {"type": "object", "properties": {"fast": {"type": "boolean"}}},
                "raw": {},
            },
            "required": ["name", "missing"],
            "anyOf": [{"properties": {"area": {"type": "string", "description": "Area"}}}],
        },
    )
    result = tool_config(tool)
    assert result["description"] == "HassTurnOn"
    assert result["expects_response"] is True
    assert result["pre_tool_speech"] == "off"
    parameters = result["parameters"]
    assert parameters["required"] == ["name"]
    props = parameters["properties"]
    assert props["name"] == {"type": "string", "description": "name"}
    assert props["area"] == {"type": "string", "description": "Area"}
    assert props["domain"] == {
        "type": "array",
        "description": "domain",
        "items": {"type": "string", "description": "domain", "enum": ["light", "3"]},
    }
    assert props["brightness"] == {"type": "integer", "description": "brightness"}
    assert props["color"] == {"type": "string", "description": "color"}
    assert props["options"] == {
        "type": "object",
        "description": "options",
        "properties": {"fast": {"type": "boolean", "description": "fast"}},
    }
    assert props["raw"] == {"type": "string", "description": "raw"}


def test_tools_without_parameters_omit_them():
    assert "parameters" not in tool_config(LiveTool("GetLiveContext", "State", None))


async def test_tools_are_created_attached_and_cached():
    http = Http()
    cache = {}
    saves = []
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache, lambda: saves.append(1))
    tools = [LiveTool("a", "A", None), LiveTool("b", "B", None)]

    assert await client.async_sync_tools(tools) == ["tool_1", "tool_2"]
    assert [(m, p) for m, p, *_ in http.calls] == [
        ("post", "/v1/convai/tools"),
        ("post", "/v1/convai/tools"),
        ("patch", "/v1/convai/agents/agent_1"),
    ]
    assert http.calls[-1][2] == {
        "conversation_config": {"agent": {"prompt": {"tool_ids": ["tool_1", "tool_2"]}}}
    }
    assert all(call[4] == {"xi-api-key": "key"} for call in http.calls)
    assert cache["agent_tool_ids"] == ["tool_1", "tool_2"] and saves == [1]

    http.calls.clear()
    assert await client.async_sync_tools(tools) == ["tool_1", "tool_2"]
    assert http.calls == [] and saves == [1]


async def test_changed_and_deleted_tools_are_updated_or_recreated():
    http = Http()
    cache = {}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    await client.async_sync_tools([LiveTool("a", "A", None), LiveTool("b", "B", None)])
    http.calls.clear()
    http.routes[("patch", "/v1/convai/tools/tool_2")] = lambda *_: Response(404)

    ids = await client.async_sync_tools(
        [LiveTool("a", "A changed", None), LiveTool("b", "B changed", None)]
    )
    assert ids == ["tool_1", "tool_3"]
    assert [(m, p) for m, p, *_ in http.calls] == [
        ("patch", "/v1/convai/tools/tool_1"),
        ("patch", "/v1/convai/tools/tool_2"),
        ("post", "/v1/convai/tools"),
        ("patch", "/v1/convai/agents/agent_1"),
    ]


async def test_a_new_agent_starts_with_a_fresh_tool_cache():
    http = Http()
    cache = {"agent_id": "old", "tools": {"a": {"id": "x", "fingerprint": "f"}}, "agent_tool_ids": ["x"]}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    assert await client.async_sync_tools([LiveTool("a", "A", None)]) == ["tool_1"]
    assert cache["agent_id"] == "agent_1" and cache["agent_tool_ids"] == ["tool_1"]


async def test_connect_syncs_tools_and_closes_the_socket():
    socket = Socket([METADATA])
    http = Http(socket=socket)
    client = ElevenLabsAgentsClient(http, "key", "agent_1")
    async with client.connect(config(tools=[LiveTool("a", "A", None)])) as session:
        assert session.is_open
    assert socket.closed
    http.ws_connect.assert_awaited_once()
    assert http.ws_connect.call_args.kwargs["params"] == {"agent_id": "agent_1"}
    assert http.ws_connect.call_args.kwargs["headers"] == {"xi-api-key": "key"}


async def test_api_errors_include_the_status():
    http = Http({("get", "/v2/voices"): lambda *_: Response(401, {"detail": "bad key"})})
    with pytest.raises(ElevenLabsAgentsError, match="401"):
        await elevenlabs.async_list_voices(http, "key")


async def test_voices_are_paged():
    pages = iter(
        [
            {"voices": [{"voice_id": "v1", "name": "One"}], "has_more": True, "next_page_token": "t"},
            {"voices": [{"voice_id": "v2"}], "has_more": False},
        ]
    )
    http = Http({("get", "/v2/voices"): lambda *_: Response(body=next(pages))})
    assert await elevenlabs.async_list_voices(http, "key") == [("v1", "One"), ("v2", "v2")]
    assert http.calls[1][3] == {"page_size": 100, "next_page_token": "t"}


@pytest.mark.parametrize(
    "module, expected",
    [
        (stt, stt.ElevenLabsAgentsSTT),
        (tts, tts.ElevenLabsAgentsTTS),
        (conversation, conversation.ElevenLabsAgentsConversationAgent),
    ],
)
async def test_provider_routes_each_pipeline_platform(module, expected):
    entities = []
    entry = SimpleNamespace(
        entry_id="el",
        data={CONF_PROVIDER: PROVIDER_ELEVENLABS, CONF_MODEL: ELEVENLABS_DEFAULT_MODEL},
        options={},
    )
    await module.async_setup_entry(SimpleNamespace(), entry, entities.extend)
    assert len(entities) == 1 and isinstance(entities[0], expected)
    assert {"en", "de"} <= set(entities[0].supported_languages)


async def test_entities_build_a_client_from_the_entry_cache(monkeypatch):
    cache = {}
    save = lambda: None  # noqa: E731
    entry = SimpleNamespace(entry_id="el", data={}, options={})
    hass = SimpleNamespace(
        data={
            "gemini_live": {
                "el": {
                    CONF_ELEVENLABS_AGENT_ID: "agent_1",
                    "elevenlabs_cache": cache,
                    "elevenlabs_cache_save": save,
                }
            }
        }
    )
    monkeypatch.setattr(stt, "async_get_clientsession", lambda _: "http")
    monkeypatch.setattr(conversation, "async_get_clientsession", lambda _: "http")
    entities = [
        stt.ElevenLabsAgentsSTT(entry),
        conversation.ElevenLabsAgentsConversationAgent(hass, entry),
    ]
    for entity in entities:
        entity.hass = hass
        client = await entity._async_create_client("key")
        assert client._agent_id == "agent_1" and client._cache is cache
        assert client._on_cache_change is save


def test_schema_shows_the_voice_only_after_voices_were_fetched():
    def keys(schema):
        return {marker.schema for marker in schema.schema}

    first = config_flow._provider_schema(PROVIDER_ELEVENLABS)
    assert CONF_VOICE not in keys(first)
    assert CONF_TRANSCRIBE_ELEVENLABS in keys(first)
    assert CONF_SUPPORT_BARGE_IN not in keys(first)

    second = config_flow._provider_schema(
        PROVIDER_ELEVENLABS, voices=[("v1", "One")]
    )
    voice = next(v for m, v in second.schema.items() if m.schema == CONF_VOICE)
    assert voice.config["custom_value"] is True
    assert voice.config["options"] == [{"value": "v1", "label": "One"}]
    model = next(v for m, v in second.schema.items() if m.schema == CONF_MODEL)
    assert model.config["custom_value"] is True
    value = second({CONF_API_KEY: "key", CONF_MODEL: "custom-llm", CONF_VOICE: "my-clone"})
    assert value[CONF_MODEL] == "custom-llm" and value[CONF_VOICE] == "my-clone"


class Flow:
    def __init__(self, language="de-DE"):
        self.hass = SimpleNamespace(config=SimpleNamespace(language=language))
        self.forms = []

    def async_show_form(self, **kwargs):
        self.forms.append(kwargs)
        return {"type": "form", **kwargs}


@pytest.fixture
def api(monkeypatch):
    calls = SimpleNamespace(voices=[("v1", "One")], created=[], updated=[], update_error=None)

    async def voices(_session, api_key):
        if api_key == "bad":
            raise ElevenLabsAgentsError("401")
        return calls.voices

    async def create(_session, api_key, name, language):
        calls.created.append((api_key, name, language))
        return "agent_new"

    async def update(_session, api_key, agent_id, language):
        calls.updated.append((api_key, agent_id, language))
        if calls.update_error:
            raise calls.update_error

    monkeypatch.setattr(config_flow, "async_list_voices", voices)
    monkeypatch.setattr(config_flow, "async_create_agent", create)
    monkeypatch.setattr(config_flow, "async_update_agent", update)
    monkeypatch.setattr(config_flow, "async_get_clientsession", lambda hass: "http")
    monkeypatch.setattr(config_flow.llm, "async_get_apis", lambda hass: [])
    return calls


async def test_flow_checks_key_then_offers_voices_then_creates_agent(api):
    flow = Flow()
    finished = []
    step = lambda user_input, current={}: config_flow._async_elevenlabs_step(  # noqa: E731
        flow, "provider", user_input, current, "ElevenLabs Agents", finished.append
    )

    form = await step(None)
    assert form["errors"] == {}

    form = await step({CONF_API_KEY: "bad", CONF_MODEL: ELEVENLABS_DEFAULT_MODEL})
    assert form["errors"] == {CONF_API_KEY: "invalid_auth"}

    form = await step({CONF_API_KEY: "key", CONF_MODEL: ELEVENLABS_DEFAULT_MODEL})
    assert form["errors"] == {} and not finished
    assert CONF_VOICE in {m.schema for m in form["data_schema"].schema}

    await step({CONF_API_KEY: "key", CONF_MODEL: ELEVENLABS_DEFAULT_MODEL, CONF_VOICE: "v1"})
    assert api.created == [("key", "Home Assistant (ElevenLabs Agents)", "de")]
    assert finished[0][CONF_ELEVENLABS_AGENT_ID] == "agent_new"
    assert finished[0][CONF_PROVIDER] == PROVIDER_ELEVENLABS
    assert finished[0][CONF_VOICE] == "v1"


async def test_flow_repairs_an_existing_agent_or_replaces_a_missing_one(api):
    flow = Flow(language="xx")
    finished = []
    current = {CONF_API_KEY: "key", CONF_ELEVENLABS_AGENT_ID: "agent_old"}
    user_input = {CONF_API_KEY: "key", CONF_MODEL: ELEVENLABS_DEFAULT_MODEL, CONF_VOICE: "v1"}

    await config_flow._async_elevenlabs_step(flow, "init", dict(user_input), current, "T", finished.append)
    assert api.updated == [("key", "agent_old", "en")]
    assert finished[-1][CONF_ELEVENLABS_AGENT_ID] == "agent_old"

    api.update_error = ElevenLabsAgentsError("missing", 404)
    await config_flow._async_elevenlabs_step(flow, "init", dict(user_input), current, "T", finished.append)
    assert finished[-1][CONF_ELEVENLABS_AGENT_ID] == "agent_new"

    # Any other failure must not leave a second agent behind.
    api.update_error = ElevenLabsAgentsError("busy", 503)
    api.created.clear()
    form = await config_flow._async_elevenlabs_step(
        flow, "init", dict(user_input), current, "T", finished.append
    )
    assert form["errors"] == {"base": "cannot_connect"} and api.created == []


async def test_flow_reports_agent_setup_failures(api, monkeypatch):
    async def fail(*_args):
        raise ElevenLabsAgentsError("500")

    monkeypatch.setattr(config_flow, "async_create_agent", fail)
    flow = Flow()
    form = await config_flow._async_elevenlabs_step(
        flow,
        "provider",
        {CONF_API_KEY: "key", CONF_MODEL: ELEVENLABS_DEFAULT_MODEL, CONF_VOICE: "v1"},
        {},
        "T",
        lambda data: pytest.fail("must not finish"),
    )
    assert form["errors"] == {"base": "cannot_connect"}


async def test_removing_an_entry_deletes_its_agent_and_tools(monkeypatch):
    calls = []

    class Store:
        async def async_load(self):
            return {"tools": {"a": {"id": "tool_1"}}}

        async def async_remove(self):
            calls.append("store removed")

    async def delete_tools(_session, api_key, cache):
        calls.append(("tools", [t["id"] for t in cache["tools"].values()]))

    async def delete_agent(_session, api_key, agent_id):
        calls.append(("agent", agent_id))

    monkeypatch.setattr(gemini_live, "_elevenlabs_store", lambda hass, entry_id: Store())
    monkeypatch.setattr(gemini_live, "async_delete_tools", delete_tools)
    monkeypatch.setattr(gemini_live, "async_delete_agent", delete_agent)
    monkeypatch.setattr(gemini_live, "async_get_clientsession", lambda hass: "http")
    entry = SimpleNamespace(
        entry_id="el",
        title="ElevenLabs Agents",
        data={CONF_PROVIDER: PROVIDER_ELEVENLABS, CONF_API_KEY: "key", CONF_ELEVENLABS_AGENT_ID: "agent_1"},
        options={},
    )
    await gemini_live.async_remove_entry(SimpleNamespace(), entry)
    assert calls == [("tools", ["tool_1"]), ("agent", "agent_1"), "store removed"]


async def test_concurrent_turns_create_each_tool_once():
    http = Http()
    cache = {}
    client = ElevenLabsAgentsClient(http, "key", "agent_lock", cache)
    tools = [LiveTool("a", "A", None)]
    first, second = await asyncio.gather(
        client.async_sync_tools(tools), client.async_sync_tools(tools)
    )
    assert first == second == ["tool_1"]
    assert [p for m, p, *_ in http.calls].count("/v1/convai/tools") == 1


async def test_idle_sessions_are_closed_after_a_turn(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_IDLE_CLOSE_TIMEOUT", 0.01)
    session, socket = await configured([audio(b"a", "Ok"), agent_response("Ok")])
    await collect(session)
    await asyncio.sleep(0.05)
    assert socket.closed and not session.is_open


async def test_a_new_turn_keeps_the_session_open(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_IDLE_CLOSE_TIMEOUT", 0.02)
    session, socket = await configured([audio(b"a", "Ok"), agent_response("Ok")])
    await collect(session)
    await session.send_audio(b"\x00\x00")
    await asyncio.sleep(0.05)
    assert not socket.closed


async def test_idle_fallback_waits_for_audio_after_early_text(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_COMPLETION_IDLE_TIMEOUT", 0.01)
    monkeypatch.setattr(elevenlabs, "_NO_AUDIO_TIMEOUT", 0.2)
    session, socket = await configured([agent_response("Hello there")], block=True)
    receiver = session.receive()
    await asyncio.sleep(0)

    async def deliver_audio_later():
        await asyncio.sleep(0.05)
        socket.push(audio(b"pcm"))

    asyncio.get_running_loop().create_task(deliver_audio_later())
    received = []
    async for event in receiver:
        received.append(event)
        if event.turn_complete:
            break
    # The text arrived first; the turn must not end before its audio.
    assert [event.audio for event in received if event.audio] == [b"pcm"]


async def test_text_without_audio_gives_up_after_the_no_audio_timeout(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_NO_AUDIO_TIMEOUT", 0.01)
    session, _ = await configured([agent_response("Hello")], block=True)
    received = await asyncio.wait_for(collect(session), 1)
    assert received[-1].turn_complete


async def test_late_audio_of_a_completed_turn_is_dropped(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SILENCE_INTERVAL", 0.001)
    user = {"type": "user_transcript", "user_transcription_event": {"user_transcript": "Next"}}
    session, socket = await configured(
        [audio(b"one", "Ok"), agent_response("Ok"), audio(b"late")], block=True
    )
    assert [e.audio for e in await collect(session) if e.audio] == [b"one"]

    # The next turn still feeds silence: late audio is not a response.
    receiver = session.receive()
    pending = asyncio.ensure_future(anext(receiver))
    await asyncio.sleep(0.01)
    await session.end_audio()
    await asyncio.sleep(0.02)
    assert any("user_audio_chunk" in event for event in socket.sent)

    socket.push(user, audio(b"two", "Fine"), agent_response("Fine"))
    events = [await pending]
    async for event in receiver:
        events.append(event)
        if event.turn_complete:
            break
    assert [e.audio for e in events if e.audio] == [b"two"]
    assert events[0].input_transcript == "Next"
    await session.async_close()


async def test_session_reports_closed_shortly_before_its_idle_close(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_IDLE_CLOSE_TIMEOUT", 1.0)
    monkeypatch.setattr(elevenlabs, "_IDLE_CLOSE_MARGIN", 2.0)
    session, socket = await configured([audio(b"a", "Ok"), agent_response("Ok")])
    assert session.is_open
    await collect(session)
    # Inside the margin: a new turn must reconnect instead of reusing it.
    assert not session.is_open and not socket.closed
    await session.send_audio(b"\x00\x00")
    assert session.is_open
    await session.async_close()


async def test_configure_selects_this_turns_tools():
    socket = Socket([METADATA])
    session = ElevenLabsAgentsSession(socket)
    await session.async_configure(config(), ["tool_1"])
    assert socket.sent[0]["conversation_config_override"]["agent"]["prompt"]["tool_ids"] == [
        "tool_1"
    ]


async def test_tools_stay_attached_when_the_tool_set_changes():
    http = Http()
    cache = {}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    assert await client.async_sync_tools([LiveTool("a", "A", None)]) == ["tool_1"]
    assert await client.async_sync_tools([LiveTool("b", "B", None)]) == ["tool_2"]
    http.calls.clear()
    # Switching back needs no API call: both tools remain attached.
    assert await client.async_sync_tools([LiveTool("a", "A", None)]) == ["tool_1"]
    assert http.calls == []
    assert cache["agent_tool_ids"] == ["tool_1", "tool_2"]


async def test_a_failed_tool_sync_keeps_the_tools_it_created():
    http = Http()
    http.routes[("post", "/v1/convai/tools")] = None
    created = iter(["tool_ok"])

    def create(body, _params):
        if body["tool_config"]["name"] == "bad":
            return Response(500, {"detail": "boom"})
        return Response(body={"id": next(created)})

    http.routes[("post", "/v1/convai/tools")] = create
    cache = {}
    saves = []
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache, lambda: saves.append(1))
    with pytest.raises(ElevenLabsAgentsError, match="500"):
        await client.async_sync_tools([LiveTool("good", "G", None), LiveTool("bad", "B", None)])
    assert cache["tools"]["good"]["id"] == "tool_ok" and saves


async def test_tool_update_errors_other_than_missing_are_raised():
    http = Http()
    cache = {}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    await client.async_sync_tools([LiveTool("a", "A", None)])
    http.routes[("patch", "/v1/convai/tools/tool_1")] = lambda *_: Response(503)
    with pytest.raises(ElevenLabsAgentsError, match="503"):
        await client.async_sync_tools([LiveTool("a", "A changed", None)])
    assert cache["tools"]["a"]["id"] == "tool_1"


async def test_tools_of_a_replaced_agent_are_deleted():
    http = Http()
    cache = {"agent_id": "old", "tools": {"a": {"id": "old_tool", "fingerprint": "f"}}}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    await client.async_sync_tools([LiveTool("a", "A", None)])
    assert ("delete", "/v1/convai/tools/old_tool") in [(m, p) for m, p, *_ in http.calls]


@pytest.mark.parametrize(
    "response, expected",
    [
        ({"success": True}, False),
        ({"error": "Unknown"}, True),
        ({"success": False}, True),
        ({"response_type": "error", "data": {"code": "no_valid_targets"}}, True),
        ({"response_type": "action_done"}, False),
    ],
)
def test_tool_results_report_home_assistant_errors(response, expected):
    assert elevenlabs._is_error(response) is expected


async def test_unloading_flushes_the_tool_cache(monkeypatch):
    saved = []

    class Store:
        async def async_save(self, data):
            saved.append(dict(data))

    class Manager:
        async def async_close_all(self):
            pass

    cache = {"tools": {"a": {"id": "tool_1"}}}
    entry = SimpleNamespace(entry_id="el")
    hass = SimpleNamespace(
        data={
            "gemini_live": {
                "el": {
                    "session_manager": Manager(),
                    "elevenlabs_cache": cache,
                    "elevenlabs_cache_store": Store(),
                }
            }
        },
        config_entries=SimpleNamespace(async_unload_platforms=AsyncMock(return_value=True)),
    )
    assert await gemini_live.async_unload_entry(hass, entry)
    assert saved == [cache]


async def test_tools_detached_outside_home_assistant_are_attached_again():
    rejection = SimpleNamespace(
        type=WSMsgType.CLOSE, extra="Tool IDs not attached to this agent: ['tool_1']"
    )
    rejected = Socket([METADATA])
    http = Http(socket=rejected)
    cache = {}
    client = ElevenLabsAgentsClient(http, "key", "agent_1", cache)
    tools = [LiveTool("a", "A", None)]
    async with client.connect(config(tools=tools)) as session:
        rejected.receive = AsyncMock(return_value=rejection)
        with pytest.raises(ElevenLabsAgentsError, match="not attached"):
            await anext(session.receive())
    assert "agent_tool_ids" not in cache

    http.calls.clear()
    http.ws_connect = AsyncMock(return_value=Socket([METADATA]))
    async with client.connect(config(tools=tools)):
        pass
    assert [(m, p) for m, p, *_ in http.calls] == [("patch", "/v1/convai/agents/agent_1")]


@pytest.mark.parametrize("transcript", ["...", "", " … ", None])
async def test_a_silent_user_turn_reports_no_speech_and_drops_the_reengagement(transcript):
    silent = {"type": "user_transcript", "user_transcription_event": {"user_transcript": transcript}}
    user = {"type": "user_transcript", "user_transcription_event": {"user_transcript": "Licht an"}}
    session, _ = await configured(
        [
            silent,
            agent_response("Bist du noch da?"),
            audio(b"reengage", "Bist du noch da?"),
            user,
            audio(b"answer", "Ok"),
            agent_response("Ok"),
        ]
    )
    received = []
    async for event in session.receive():
        received.append(event)
        if event.turn_complete:
            break
    assert received[0].no_speech and not received[0].audio
    assert [e.audio for e in received if e.audio] == [b"answer"]
    assert not any(e.output_transcript == "Bist du noch da?" for e in received)
    assert [e.input_transcript for e in received if e.input_transcript] == ["Licht an"]


async def test_a_silent_turn_stops_the_trailing_silence(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SILENCE_INTERVAL", 0.001)
    session, socket = await configured([], block=True)
    await session.end_audio()
    await asyncio.sleep(0.01)
    socket.push({"type": "user_transcript", "user_transcription_event": {"user_transcript": "..."}})
    event = await anext(session.receive())
    assert event.no_speech
    sent = len(socket.sent)
    await asyncio.sleep(0.02)
    assert len(socket.sent) == sent
    await session.async_close()


def test_the_instruction_discourages_offers_of_further_help():
    for entity_class in (stt.ElevenLabsAgentsSTT, conversation.ElevenLabsAgentsConversationAgent):
        assert "Do not end replies by offering further help" in entity_class.default_system_instruction


@pytest.mark.parametrize(
    "language, expected", [("de-DE", "de"), ("pt-BR", "pt"), ("xx", "en"), (None, "en")]
)
def test_agent_language_follows_home_assistant(language, expected):
    assert elevenlabs.agent_language(language) == expected


async def test_setup_refreshes_the_agent_base_settings(monkeypatch):
    calls = []

    async def update(_session, api_key, agent_id, language):
        calls.append((api_key, agent_id, language))

    monkeypatch.setattr(gemini_live, "async_update_agent", update)
    monkeypatch.setattr(gemini_live, "async_get_clientsession", lambda hass: "http")
    hass = SimpleNamespace(config=SimpleNamespace(language="de"))
    await gemini_live._async_refresh_agent(
        hass, {CONF_API_KEY: "key", CONF_ELEVENLABS_AGENT_ID: "agent_1"}
    )
    assert calls == [("key", "agent_1", "de")]

    async def fail(*_args):
        raise ElevenLabsAgentsError("offline")

    monkeypatch.setattr(gemini_live, "async_update_agent", fail)
    # A failed refresh must not break setup.
    await gemini_live._async_refresh_agent(
        hass, {CONF_API_KEY: "key", CONF_ELEVENLABS_AGENT_ID: "agent_1"}
    )


async def test_a_sound_response_ends_the_turn_without_speech(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SOUND_RESPONSE_GRACE", 0.01)
    session, socket = await configured(
        [tool_call("HassTurnOff", "call_1"), tool_call("play_sound", "call_2")], block=True
    )
    receiver = session.receive()
    first = await anext(receiver)
    await session.send_tool_responses([LiveToolResponse("HassTurnOff", "call_1", {"success": True})])
    second = await anext(receiver)
    assert [c.name for c in second.tool_calls] == ["play_sound"]
    await session.send_tool_responses(
        [LiveToolResponse("play_sound", "call_2", {"success": True, "response_mode": "sound"})]
    )
    done = await asyncio.wait_for(anext(receiver), 1)
    assert done.turn_complete
    assert [c.name for c in first.tool_calls] == ["HassTurnOff"]
    await session.async_close()


async def test_parallel_tool_calls_after_the_sound_still_run(monkeypatch):
    monkeypatch.setattr(elevenlabs, "_SOUND_RESPONSE_GRACE", 0.2)
    session, socket = await configured([tool_call("play_sound", "call_1")], block=True)
    receiver = session.receive()
    await anext(receiver)
    await session.send_tool_responses(
        [LiveToolResponse("play_sound", "call_1", {"success": True, "response_mode": "sound"})]
    )
    socket.push(tool_call("HassTurnOff", "call_2"))
    event = await asyncio.wait_for(anext(receiver), 1)
    assert [c.name for c in event.tool_calls] == ["HassTurnOff"]
    await session.send_tool_responses([LiveToolResponse("HassTurnOff", "call_2", {"success": True})])
    assert (await asyncio.wait_for(anext(receiver), 1)).turn_complete
    await session.async_close()

