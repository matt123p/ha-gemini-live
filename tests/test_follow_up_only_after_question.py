"""Tests for the optional 'listen for a follow-up only after a question' mode."""

from gemini_live.const import CONF_FOLLOW_UP_ONLY_AFTER_QUESTION
from gemini_live.runtime import LiveSessionManager
from gemini_live.stt import (
    END_CONVERSATION_TOOL_NAME,
    KEEP_LISTENING_TOOL_NAME,
    _add_end_conversation_instruction,
    _add_end_conversation_tool,
    follow_up_only_after_question,
)

CONVERSATION_ID = "conv-1"


def test_default_keeps_listening_after_every_reply():
    manager = LiveSessionManager()

    assert manager.should_continue_conversation(CONVERSATION_ID)
    assert manager.should_continue_conversation(
        CONVERSATION_ID, only_after_question=False
    )


def test_only_after_question_stops_listening_unless_requested():
    manager = LiveSessionManager()

    assert not manager.should_continue_conversation(
        CONVERSATION_ID, only_after_question=True
    )

    manager.request_listen(CONVERSATION_ID)
    assert manager.should_continue_conversation(
        CONVERSATION_ID, only_after_question=True
    )


def test_listen_request_is_cleared_for_the_next_turn():
    manager = LiveSessionManager()
    manager.request_listen(CONVERSATION_ID)

    manager.reset_conversation(CONVERSATION_ID)

    assert not manager.should_continue_conversation(
        CONVERSATION_ID, only_after_question=True
    )


def test_end_conversation_wins_over_a_listen_request():
    manager = LiveSessionManager()
    manager.request_listen(CONVERSATION_ID)

    manager.complete_conversation(CONVERSATION_ID)

    assert not manager.should_continue_conversation(
        CONVERSATION_ID, only_after_question=True
    )
    assert not manager.should_continue_conversation(CONVERSATION_ID)


def test_keep_listening_tool_and_instruction_only_when_enabled():
    default_tools = [tool.name for tool in _add_end_conversation_tool([])]
    enabled_tools = [
        tool.name for tool in _add_end_conversation_tool([], keep_listening=True)
    ]

    assert default_tools == [END_CONVERSATION_TOOL_NAME]
    assert enabled_tools == [END_CONVERSATION_TOOL_NAME, KEEP_LISTENING_TOOL_NAME]
    assert KEEP_LISTENING_TOOL_NAME not in _add_end_conversation_instruction("x")
    assert KEEP_LISTENING_TOOL_NAME in _add_end_conversation_instruction(
        "x", keep_listening=True
    )


def test_option_defaults_to_off():
    assert not follow_up_only_after_question({})
    assert follow_up_only_after_question({CONF_FOLLOW_UP_ONLY_AFTER_QUESTION: True})
