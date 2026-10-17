"""Regression coverage for quota rejections and SDK-wrapped WebSocket closes."""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai.errors import APIError
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from gemini_live.conversation import GeminiLiveConversationAgent
from gemini_live.live import LiveEvent
from gemini_live.stt import (
    GeminiLiveSTT,
    _QUOTA_USER_MESSAGE,
    _SPENDING_CAP_USER_MESSAGE,
    _is_connection_closed_ok,
    _user_visible_api_error,
)
from homeassistant.components.stt import SpeechResultState

from test_stt_barge_in import (
    AUDIO_A,
    MIC_CHUNK,
    FakeHass,
    MicStream,
    ScriptedClient,
    ScriptedSession,
    _bind_client,
    _make_entity,
    _metadata,
)


def test_normal_close_recognizes_sdk_wrapper_and_exception_chain():
    closed = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), False)
    wrapped = APIError(1000, None)
    wrapped.__context__ = closed
    outer = RuntimeError("receive failed")
    outer.__cause__ = wrapped
    assert _is_connection_closed_ok(closed)
    assert _is_connection_closed_ok(wrapped)
    assert _is_connection_closed_ok(outer)
    assert _is_connection_closed_ok(APIError(1000, None))


@pytest.mark.parametrize("code", [1007, 1011, 429, 500])
def test_other_api_errors_are_not_normal_closes(code):
    assert not _is_connection_closed_ok(APIError(code, "failure"))


def test_cyclic_exception_chain_terminates():
    error = RuntimeError("failure")
    error.__context__ = error
    assert not _is_connection_closed_ok(error)
    assert _user_visible_api_error(error) is None


def test_generic_quota_error_has_no_assumption_about_exhausted_credit():
    error = APIError(
        1011,
        "You exceeded your current quota, please check your plan and billing details.",
    )
    outer = RuntimeError("receive failed")
    outer.__cause__ = error
    assert _user_visible_api_error(outer) == _QUOTA_USER_MESSAGE
    assert (
        _user_visible_api_error(APIError(1011, "exceeded its monthly spending cap"))
        == _SPENDING_CAP_USER_MESSAGE
    )
    assert _user_visible_api_error(APIError(1011, "internal server error")) is None


@pytest.mark.parametrize("with_response", [False, True])
async def test_voice_normal_close_preserves_response_and_retires_session(
    with_response, caplog
):
    hass = FakeHass()
    entity, manager, turns = _make_entity(hass, {"api_key": "k"}, GeminiLiveSTT)

    class ClosingSession(ScriptedSession):
        async def receive(self):
            yield LiveEvent(input_transcript="hello")
            if with_response:
                yield LiveEvent(audio=AUDIO_A, output_transcript="Hello back")
            raise APIError(1000, None)

    client = ScriptedClient(ClosingSession(support_barge_in=False))
    _bind_client(entity, client)
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    result = await asyncio.wait_for(
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
            asyncio.Future(),
            "conversation-1",
            None,
        ),
        5,
    )
    assert result.result == (
        SpeechResultState.SUCCESS if with_response else SpeechResultState.ERROR
    )
    assert client.close_count == 1
    assert not manager._connections
    assert not any(record.exc_info for record in caplog.records)
    if with_response:
        assert turns.take_voice_turn("conversation-1", result.text) is not None


@pytest.mark.parametrize("during_setup", [False, True])
async def test_voice_quota_returns_message_and_creates_repair(
    during_setup, monkeypatch, caplog
):
    hass = FakeHass()
    entity, manager, turns = _make_entity(hass, {"api_key": "k"}, GeminiLiveSTT)
    issues = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: issues.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "gemini_live.stt.active_pipeline_context",
        lambda *_: ("conversation-1", None, None),
    )

    class QuotaSession(ScriptedSession):
        async def receive(self):
            yield LiveEvent(input_transcript="hello")
            raise APIError(1011, "You exceeded your current quota")

    class QuotaClient(ScriptedClient):
        def connect(self, config):
            if during_setup:
                raise APIError(1011, "You exceeded your current quota")
            return super().connect(config)

    _bind_client(entity, QuotaClient(QuotaSession(support_barge_in=False)))
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    result = await entity._async_process_audio_stream_sdk(
        _metadata(),
        mic.chunks(),
        "k",
        "m",
        "v",
        "",
        True,
        False,
        False,
    )
    assert result.result == SpeechResultState.SUCCESS
    assert result.text == _QUOTA_USER_MESSAGE
    assert issues[0][0][2] == "quota_exceeded_test_entry"
    assert issues[0][1]["translation_key"] == "quota_exceeded"
    assert not manager.should_continue_conversation("conversation-1")
    assert turns.take_voice_turn("conversation-1", result.text).complete_conversation
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("failure", ["quota", "unexpected", "normal_close"])
async def test_text_connection_failure_is_handled_before_session_exists(
    failure, monkeypatch, caplog
):
    hass = FakeHass()
    stt, manager, _ = _make_entity(
        hass, {"api_key": "k", "model": "m", "voice": "v"}, GeminiLiveSTT
    )
    agent = GeminiLiveConversationAgent(hass, stt.entry)
    issues = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: issues.append(kwargs),
    )

    async def llm_api(_context):
        return None, [], "instruction"

    errors = {
        "quota": APIError(1011, "You exceeded your current quota"),
        "unexpected": RuntimeError("connection setup failed"),
        "normal_close": APIError(1000, None),
    }

    class FailingClient:
        def connect(self, _config):
            raise errors[failure]

    async def create(_key):
        return FailingClient()

    agent._async_get_llm_api = llm_api
    agent._async_create_client = create
    result = await agent._async_process_text_live(
        "hello",
        SimpleNamespace(as_llm_context=lambda _: None),
        "conversation-1",
    )
    if failure == "quota":
        assert result == _QUOTA_USER_MESSAGE
        assert issues[0]["translation_key"] == "quota_exceeded"
        assert not manager.should_continue_conversation("conversation-1")
    else:
        assert result is None
    if failure != "unexpected":
        assert not any(record.exc_info for record in caplog.records)


async def test_text_normal_close_keeps_partial_response_and_clears_repair(
    monkeypatch, caplog
):
    hass = FakeHass()
    stt, manager, _ = _make_entity(
        hass, {"api_key": "k", "model": "m", "voice": "v"}, GeminiLiveSTT
    )
    agent = GeminiLiveConversationAgent(hass, stt.entry)
    cleared = []
    monkeypatch.setattr(
        "gemini_live.stt.async_delete_issue", lambda *args: cleared.append(args[2])
    )

    class ClosingSession(ScriptedSession):
        async def send_text(self, _text):
            pass

        async def receive(self):
            yield LiveEvent(output_transcript="Hello back")
            raise APIError(1000, None)

    client = ScriptedClient(ClosingSession(support_barge_in=False))

    async def create(_key):
        return client

    async def llm_api(_context):
        return None, [], "instruction"

    agent._async_create_client = create
    agent._async_get_llm_api = llm_api
    result = await agent._async_process_text_live(
        "hello",
        SimpleNamespace(as_llm_context=lambda _: None),
        "conversation-1",
    )
    assert result == "Hello back"
    assert "quota_exceeded_test_entry" in cleared
    assert client.close_count == 1
    assert not manager._connections
    assert not any(record.exc_info for record in caplog.records)
