"""Tests for the OpenAI Realtime adapter's barge-in behaviour."""

import base64
import json
from collections import deque
from types import SimpleNamespace

from aiohttp import WSMsgType
from gemini_live.live import LiveConfig, LiveEvent
from gemini_live.openai import OpenAIRealtimeError, OpenAIRealtimeSession


def _make_config(**overrides) -> LiveConfig:
    defaults = {"model": "m", "voice": "v", "system_instruction": "s"}
    defaults.update(overrides)
    return LiveConfig(**defaults)


class _FakeWebSocket:
    def __init__(self, incoming: list[dict]) -> None:
        self.sent: list[dict] = []
        self._incoming = deque(incoming)
        self.closed = False

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)

    async def receive(self):
        if not self._incoming:
            return SimpleNamespace(type=WSMsgType.CLOSE, data="")
        payload = self._incoming.popleft()
        return SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(payload))

    async def close(self) -> None:
        self.closed = True


async def _configured_session(
    support_barge_in: bool,
    incoming: list[dict] | None = None,
) -> tuple[OpenAIRealtimeSession, _FakeWebSocket]:
    websocket = _FakeWebSocket(
        [
            {"type": "session.updated"},
            *(incoming or []),
        ]
    )
    session = OpenAIRealtimeSession(websocket, support_barge_in=support_barge_in)
    await session.async_configure(_make_config())
    return session, websocket


async def _collect(session: OpenAIRealtimeSession) -> list[LiveEvent]:
    events = []
    try:
        async for event in session.receive():
            events.append(event)
    except OpenAIRealtimeError:
        pass
    return events


async def test_configure_legacy_uses_manual_turn_detection():
    _, websocket = await _configured_session(support_barge_in=False)

    session_update = websocket.sent[0]["session"]
    assert session_update["audio"]["input"]["turn_detection"] is None


async def test_configure_barge_in_uses_server_vad():
    _, websocket = await _configured_session(support_barge_in=True)

    turn_detection = websocket.sent[0]["session"]["audio"]["input"]["turn_detection"]
    assert turn_detection["type"] == "server_vad"
    assert turn_detection["create_response"] is True
    assert turn_detection["interrupt_response"] is True


async def test_end_audio_legacy_commits_and_creates_response():
    session, websocket = await _configured_session(support_barge_in=False)
    configure_count = len(websocket.sent)

    await session.end_audio()

    trailing = websocket.sent[configure_count:]
    assert [event["type"] for event in trailing] == [
        "input_audio_buffer.commit",
        "response.create",
    ]


async def test_end_audio_barge_in_does_not_commit_or_create():
    session, websocket = await _configured_session(support_barge_in=True)
    configure_count = len(websocket.sent)

    await session.end_audio()

    assert len(websocket.sent) == configure_count


AUDIO_A = base64.b64encode(b"a").decode("ascii")
AUDIO_B = base64.b64encode(b"b").decode("ascii")


async def test_barge_in_interruption_sequence_normalization():
    session, _ = await _configured_session(
        support_barge_in=True,
        incoming=[
            {"type": "response.output_audio.delta", "delta": AUDIO_A},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.output_audio.delta", "delta": AUDIO_B},
            {"type": "response.done", "response": {"status": "completed"}},
        ],
    )

    events = await _collect(session)

    assert events == [
        LiveEvent(audio=b"a"),
        LiveEvent(interrupted=True, user_activity_started=True),
        LiveEvent(interrupted=True),
        LiveEvent(user_activity_stopped=True),
        LiveEvent(audio=b"b"),
        LiveEvent(turn_complete=True),
    ]


async def test_barge_in_repeated_speech_start_yields_single_interruption():
    session, _ = await _configured_session(
        support_barge_in=True,
        incoming=[
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {"type": "response.output_audio.delta", "delta": AUDIO_B},
            {"type": "response.done", "response": {"status": "completed"}},
        ],
    )

    events = await _collect(session)

    # speech_started deduped: only the first one yields an interruption event.
    speech_start_events = [event for event in events if event.user_activity_started]
    assert len(speech_start_events) == 1
    assert events[-1] == LiveEvent(turn_complete=True)


async def test_legacy_completed_response_remains_terminal():
    session, _ = await _configured_session(
        support_barge_in=False,
        incoming=[
            {"type": "response.output_audio.delta", "delta": AUDIO_A},
            {"type": "response.done", "response": {"status": "completed"}},
        ],
    )

    events = await _collect(session)

    assert events == [LiveEvent(audio=b"a"), LiveEvent(turn_complete=True)]


async def test_legacy_cancelled_response_keeps_existing_turn_complete_behaviour():
    """Without barge-in, cancelled responses keep the pre-barge-in handling."""
    session, _ = await _configured_session(
        support_barge_in=False,
        incoming=[
            {"type": "response.done", "response": {"status": "cancelled"}},
        ],
    )

    events = await _collect(session)

    assert events == [LiveEvent(turn_complete=True)]


async def test_legacy_speech_events_are_not_emitted():
    session, _ = await _configured_session(
        support_barge_in=False,
        incoming=[
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.done", "response": {"status": "completed"}},
        ],
    )

    events = await _collect(session)

    assert events == [LiveEvent(turn_complete=True)]


async def test_failed_response_still_raises():
    session, _ = await _configured_session(
        support_barge_in=True,
        incoming=[
            {
                "type": "response.done",
                "response": {"status": "failed", "status_details": "boom"},
            },
        ],
    )

    try:
        async for _ in session.receive():
            pass
    except OpenAIRealtimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected OpenAIRealtimeError")


async def test_transcript_deltas_passthrough_when_enabled():
    websocket = _FakeWebSocket(
        [
            {"type": "session.updated"},
            {"type": "response.output_audio_transcript.delta", "delta": "hello"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = OpenAIRealtimeSession(websocket, support_barge_in=False)
    config = _make_config(transcribe_output=True)
    await session.async_configure(config)

    events = []
    try:
        async for event in session.receive():
            events.append(event)
    except OpenAIRealtimeError:
        pass

    assert events == [
        LiveEvent(output_transcript="hello"),
        LiveEvent(turn_complete=True),
    ]
