"""Follow-up decisions through the voice and typed conversation pipelines."""

import asyncio
from types import SimpleNamespace

import pytest

from homeassistant.const import CONF_LLM_HASS_API

from gemini_live.config_flow import _provider_schema
from gemini_live.const import (
    CONF_FOLLOW_UP_ONLY_AFTER_QUESTION,
    PROVIDER_ELEVENLABS,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
    PROVIDER_QWEN,
)
from gemini_live.conversation import (
    GPTRealtimeConversationAgent,
    GeminiLiveConversationAgent,
)
from gemini_live.live import LiveEvent, LiveToolCall
from gemini_live.stt import GPTRealtimeSTT, GeminiLiveSTT

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

CONVERSATION_ID = "conversation-1"
PROVIDERS = [
    (GeminiLiveSTT, GeminiLiveConversationAgent),
    (GPTRealtimeSTT, GPTRealtimeConversationAgent),
]


class RecordingChatLog:
    """Collect assistant content without starting a Home Assistant instance."""

    def __init__(self):
        self.content = []

    def async_add_assistant_content_without_tools(self, content):
        self.content.append(content.content)

    async def async_add_delta_content_stream(self, _entity_id, deltas):
        async for delta in deltas:
            if "content" in delta:
                self.content.append(delta["content"])
            yield delta


class FollowUpReply(ScriptedSession):
    """Gate completion so the conversation handoff runs during audio output."""

    def __init__(self, calls=(), *, late=False):
        super().__init__(support_barge_in=False)
        self.calls = calls
        self.late = late
        self.tool_responses = []

    async def send_text(self, text):
        assert text == "question"

    async def send_tool_responses(self, responses):
        self.tool_responses.extend(responses)

    async def receive(self):
        yield LiveEvent(input_transcript="question")
        if self.calls and not self.late:
            yield LiveEvent(tool_calls=list(self.calls))
        yield LiveEvent(audio=AUDIO_A, output_transcript="Which lamp?")
        await self.release_gate.wait()
        if self.calls and self.late:
            yield LiveEvent(tool_calls=list(self.calls))
        yield LiveEvent(turn_complete=True)


def _calls(*names):
    return [LiveToolCall(name, str(index), {}) for index, name in enumerate(names)]


@pytest.mark.parametrize("entity_class,agent_class", PROVIDERS)
@pytest.mark.parametrize("transcribe", [False, True])
@pytest.mark.parametrize(
    "enabled,names,expected",
    [
        (False, (), True),
        (True, (), False),
        (True, ("keep_listening",), True),
        (True, ("keep_listening", "end_conversation"), False),
    ],
)
async def test_voice_follow_up_handoff(
    entity_class, agent_class, transcribe, enabled, names, expected
):
    await _voice_handoff(
        entity_class, agent_class, transcribe, enabled, names, expected
    )


async def _voice_handoff(
    entity_class, agent_class, transcribe, enabled, names, expected, *, late=False
):
    hass = FakeHass()
    entity, manager, _turns = _make_entity(
        hass, {"api_key": "k", CONF_LLM_HASS_API: []}, entity_class
    )
    entity.entry.options[CONF_FOLLOW_UP_ONLY_AFTER_QUESTION] = enabled
    # A previous question must not keep the next ordinary answer listening.
    manager.request_listen(CONVERSATION_ID)
    session = FollowUpReply(_calls(*names), late=late)
    _bind_client(entity, ScriptedClient(session))
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    result_future = asyncio.Future()
    run = asyncio.create_task(
        entity._async_run_audio_stream_sdk(
            _metadata(),
            mic.chunks(),
            "k",
            "m",
            "v",
            "",
            transcribe,
            False,
            False,
            result_future,
            CONVERSATION_ID,
            None,
        )
    )
    try:
        stt_result = await asyncio.wait_for(result_future, 2)
        if transcribe:
            session.release_gate.set()
        agent = agent_class(hass, entity.entry)
        agent.entity_id = "conversation.live_model"
        agent._fire_conversation_entry = lambda *_args: None
        chat_log = RecordingChatLog()
        result = await asyncio.wait_for(
            agent._async_handle_message(
                SimpleNamespace(
                    text=stt_result.text, language="en", conversation_id=CONVERSATION_ID
                ),
                chat_log,
            ),
            2,
        )
        assert result.continue_conversation is expected
        assert chat_log.content
        assert [response.name for response in session.tool_responses] == (
            [] if late else list(names)
        )
        session.release_gate.set()
        await asyncio.wait_for(run, 2)
        if late:
            assert manager.should_continue_conversation(
                CONVERSATION_ID, only_after_question=True
            )
            # HA already received this result; later tool calls cannot alter it.
            assert not result.continue_conversation
    finally:
        session.release_gate.set()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await manager.async_close_all()


async def test_tool_after_voice_handoff_cannot_change_returned_decision():
    await _voice_handoff(
        GeminiLiveSTT,
        GeminiLiveConversationAgent,
        False,
        True,
        ("keep_listening",),
        False,
        late=True,
    )


@pytest.mark.parametrize(
    "names,expected",
    [
        ((), False),
        (("keep_listening",), True),
        (("keep_listening", "end_conversation"), False),
    ],
)
async def test_typed_follow_up_handoff(names, expected):
    hass = FakeHass()
    entity, manager, _turns = _make_entity(
        hass,
        {
            "api_key": "k",
            CONF_LLM_HASS_API: [],
            CONF_FOLLOW_UP_ONLY_AFTER_QUESTION: True,
        },
        GeminiLiveSTT,
    )
    agent = GeminiLiveConversationAgent(hass, entity.entry)
    agent._fire_conversation_entry = lambda *_args: None
    session = FollowUpReply(_calls(*names))
    session.release_gate.set()
    _bind_client(agent, ScriptedClient(session))
    manager.request_listen(CONVERSATION_ID)
    try:
        result = await asyncio.wait_for(
            agent._async_handle_message(
                SimpleNamespace(
                    text="question",
                    language="en",
                    conversation_id=CONVERSATION_ID,
                    as_llm_context=lambda _domain: None,
                ),
                RecordingChatLog(),
            ),
            2,
        )
        assert result.continue_conversation is expected
        assert [response.name for response in session.tool_responses] == list(names)
    finally:
        await manager.async_close_all()


@pytest.mark.parametrize(
    "provider",
    [
        PROVIDER_GEMINI,
        PROVIDER_OPENAI,
        PROVIDER_QWEN,
        PROVIDER_ELEVENLABS,
    ],
)
def test_follow_up_setting_is_off_by_default_and_preserves_selection(provider):
    for config, expected in [
        ({}, False),
        ({CONF_FOLLOW_UP_ONLY_AFTER_QUESTION: True}, True),
    ]:
        schema = _provider_schema(provider, config)
        marker = next(
            marker
            for marker in schema.schema
            if marker.schema == CONF_FOLLOW_UP_ONLY_AFTER_QUESTION
        )
        assert marker.default() is expected


def test_personaplex_does_not_offer_follow_up_setting():
    assert CONF_FOLLOW_UP_ONLY_AFTER_QUESTION not in [
        marker.schema for marker in _provider_schema(PROVIDER_PERSONAPLEX).schema
    ]
