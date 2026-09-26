"""Tests for the Gemini Live adapter's barge-in behaviour."""

import sys
from types import SimpleNamespace

from gemini_live.gemini import (
    GeminiLiveClient,
    GeminiLiveSession,
    _gemini_config,
    async_create_gemini_client,
)
from gemini_live.live import LiveConfig, LiveEvent, LiveTool


def _make_config(**overrides) -> LiveConfig:
    defaults = {"model": "m", "voice": "v", "system_instruction": "s"}
    defaults.update(overrides)
    return LiveConfig(**defaults)


def _server_content(**overrides):
    defaults = {
        "model_turn": None,
        "output_transcription": None,
        "input_transcription": None,
        "interrupted": False,
        "turn_complete": False,
        "interaction_status": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _sdk_response(server_content=None, **attributes):
    return SimpleNamespace(
        tool_call=None,
        server_content=server_content,
        go_away=None,
        session_resumption_update=None,
        **attributes,
    )


class _FakeSDKSession:
    def __init__(self, responses):
        self._responses = responses

    async def receive(self):
        for response in self._responses:
            yield response


class _TurnBoundedFakeSDKSession:
    """Model the SDK returning one iterator per completed server turn."""

    def __init__(self, turns):
        self._turns = iter(turns)
        self.receive_count = 0

    async def receive(self):
        self.receive_count += 1
        for response in next(self._turns):
            yield response


class _FakeConnectContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_exc):
        return None


class _FakeSDKClient:
    def __init__(self, session):
        self.context = _FakeConnectContext(session)
        self.aio = SimpleNamespace(
            live=SimpleNamespace(connect=lambda **_kwargs: self.context)
        )


async def _collect(session: GeminiLiveSession) -> list[LiveEvent]:
    return [event async for event in session.receive()]


def test_gemini_config_without_barge_in_disables_activity_interruption():
    config = _gemini_config(_make_config())

    assert config["realtime_input_config"] == {
        "activity_handling": "NO_INTERRUPTION",
        "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY"
    }


def test_gemini_config_barge_in_enables_activity_interruption():
    config = _gemini_config(_make_config(support_barge_in=True))

    realtime_input = config["realtime_input_config"]
    assert realtime_input["automatic_activity_detection"]["disabled"] is False
    assert realtime_input["activity_handling"] == "START_OF_ACTIVITY_INTERRUPTS"
    assert realtime_input["turn_coverage"] == "TURN_INCLUDES_ONLY_ACTIVITY"


def test_gemini_config_barge_in_does_not_change_other_settings():
    legacy = _gemini_config(_make_config())
    barge_in = _gemini_config(_make_config(support_barge_in=True))

    for key in legacy:
        if key != "realtime_input_config":
            assert legacy[key] == barge_in[key]


def test_gemini_config_legacy_tools_have_no_behaviour():
    config = _gemini_config(
        _make_config(
            model="gemini-3.1-flash-live-preview",
            tools=[LiveTool("my_tool", "does things", {"type": "object"})],
        )
    )

    declaration = config["tools"][0]["function_declarations"][0]
    assert "behavior" not in declaration


def test_gemini_config_38_live_tools_opt_into_blocking_behaviour():
    config = _gemini_config(
        _make_config(
            model="gemini-3.8-live",
            tools=[LiveTool("my_tool", "does things", {"type": "object"})],
        )
    )

    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["behavior"] == "BLOCKING"
    assert declaration["parameters"] == {
        "type": "OBJECT",
        "properties": {"json": {"type": "STRING"}},
        "required": [],
    }


def test_gemini_config_extended_thinking_uses_non_blocking_tools():
    config = _gemini_config(
        _make_config(
            model="gemini-3.8-live-extended-thinking",
            thinking_level="high",
            tools=[LiveTool("my_tool", "does things")],
        )
    )

    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["behavior"] == "NON_BLOCKING"
    assert config["thinking_config"] == {"thinking_level": "high"}


def test_gemini_config_search_grounding_combines_with_function_tools():
    config = _gemini_config(
        _make_config(
            model="gemini-3.8-live",
            search_grounding=True,
            tools=[LiveTool("my_tool", "does things")],
        )
    )

    assert config["tools"][0] == {"google_search": {}}
    assert config["tools"][1]["function_declarations"][0]["name"] == "my_tool"


def test_gemini_config_affective_dialog_only_for_25_native_audio():
    enabled = _gemini_config(
        _make_config(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            affective_dialog=True,
        )
    )
    assert enabled["generation_config"] == {"enable_affective_dialog": True}
    assert "enable_affective_dialog" not in enabled
    assert "proactivity" not in enabled

    disabled = _gemini_config(
        _make_config(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            affective_dialog=False,
        )
    )
    assert "generation_config" not in disabled

    for model in (
        "gemini-3.8-live",
        "gemini-3.8-live-extended-thinking",
        "gemini-3.1-flash-live-preview",
    ):
        removed = _gemini_config(
            _make_config(model=model, affective_dialog=True)
        )
        assert "generation_config" not in removed


async def test_gemini_client_pins_v1beta(monkeypatch):
    captured = {}

    class _FakeHttpOptions:
        def __init__(self, api_version=None):
            self.api_version = api_version
            captured["api_version"] = api_version

    class _FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.api_key = api_key
            self.http_options = http_options

    class _FakeGenaiModule:
        types = SimpleNamespace(HttpOptions=_FakeHttpOptions)
        Client = _FakeClient

    monkeypatch.setitem(
        sys.modules, "google", SimpleNamespace(genai=_FakeGenaiModule)
    )
    monkeypatch.setitem(sys.modules, "google.genai", _FakeGenaiModule)

    class _Hass:
        async def async_add_executor_job(self, func):
            return func()

    client = await async_create_gemini_client(
        _Hass(), "test-key", affective_dialog=True
    )

    assert isinstance(client, GeminiLiveClient)
    assert client._client.api_key == "test-key"
    assert client._client.http_options is not None
    assert captured["api_version"] == "v1beta"

    client = await async_create_gemini_client(_Hass(), "test-key")

    assert isinstance(client, GeminiLiveClient)
    assert client._client.http_options is None


async def test_gemini_interrupted_event_is_normalized():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(interrupted=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=False)]


async def test_gemini_combined_interrupted_and_turn_complete_single_event():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(interrupted=True, turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=True)]


async def test_gemini_plain_turn_complete_remains_terminal_marker():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(turn_complete=True)]


async def test_extended_thinking_waits_until_interaction_is_idle():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [
                _sdk_response(
                    _server_content(
                        turn_complete=True,
                        interaction_status="IN_PROGRESS",
                    ),
                )
            ],
            [
                _sdk_response(
                    _server_content(
                        turn_complete=True,
                        interaction_status="IDLE",
                    ),
                )
            ],
        ]
    )
    session = GeminiLiveSession(sdk_session, extended_thinking=True)

    events = await _collect(session)

    assert events == [LiveEvent(turn_complete=True)]
    assert sdk_session.receive_count == 2


async def test_gemini_server_content_without_interrupted_attribute():
    # Older SDK payloads may not carry the interrupted attribute at all.
    content = SimpleNamespace(
        model_turn=None,
        output_transcription=None,
        input_transcription=None,
        turn_complete=True,
    )
    session = GeminiLiveSession(_FakeSDKSession([_sdk_response(content)]))

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=False, turn_complete=True)]


async def test_gemini_full_interrupted_sequence_normalization():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(turn_complete=True, interrupted=False)),
                _sdk_response(_server_content(interrupted=True, turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert [event.turn_complete for event in events] == [True, True]
    assert [event.interrupted for event in events] == [False, True]


async def test_gemini_barge_in_reenters_sdk_receive_for_replacement_turn():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [
                _sdk_response(_server_content(interrupted=True)),
                _sdk_response(
                    _server_content(turn_complete=True)
                )
            ],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    session = GeminiLiveSession(sdk_session, support_barge_in=True)

    events = await _collect(session)

    assert events == [
        LiveEvent(interrupted=True),
        LiveEvent(turn_complete=True),
        LiveEvent(turn_complete=True),
    ]
    assert sdk_session.receive_count == 2


async def test_gemini_client_passes_barge_in_mode_to_session():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [_sdk_response(_server_content(interrupted=True, turn_complete=True))],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    client = GeminiLiveClient(_FakeSDKClient(sdk_session))

    async with client.connect(_make_config(support_barge_in=True)) as session:
        events = await _collect(session)

    assert events[-1] == LiveEvent(turn_complete=True)
    assert sdk_session.receive_count == 2


async def test_gemini_legacy_does_not_reenter_sdk_receive():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [
                _sdk_response(
                    _server_content(interrupted=True, turn_complete=True)
                )
            ],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    session = GeminiLiveSession(sdk_session, support_barge_in=False)

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=True)]
    assert sdk_session.receive_count == 1
