"""Runtime state for persistent Gemini Live conversations and pipeline turns."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineTurn:
    """A Gemini turn waiting for later pipeline stages."""

    conversation_id: str
    user_text: str
    assistant_text: str
    audio: bytes | AudioStream
    assistant_text_stream: TextStream | None = None


class AudioStream:
    """Buffer one Gemini audio response for the TTS stage."""

    def __init__(self) -> None:
        """Initialize an audio stream."""
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._finished = False

    def add_chunk(self, chunk: bytes) -> None:
        """Add one PCM audio chunk."""
        if chunk and not self._finished:
            self._queue.put_nowait(chunk)

    def finish(self) -> None:
        """Signal that no more audio chunks will arrive."""
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(None)

    async def async_chunks(self) -> AsyncGenerator[bytes]:
        """Yield buffered and future PCM chunks."""
        while (chunk := await self._queue.get()) is not None:
            yield chunk


class TextStream:
    """Buffer one Gemini response transcript for the conversation stage."""

    def __init__(self) -> None:
        """Initialize a text stream."""
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._parts: list[str] = []
        self._finished = False

    @property
    def text(self) -> str:
        """Return all transcript text received so far."""
        return "".join(self._parts)

    def add_chunk(self, chunk: str) -> None:
        """Add one transcript chunk."""
        if chunk and not self._finished:
            self._parts.append(chunk)
            self._queue.put_nowait(chunk)

    def finish(self) -> None:
        """Signal that no more transcript chunks will arrive."""
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(None)

    async def async_chunks(self) -> AsyncGenerator[str]:
        """Yield buffered and future transcript chunks."""
        while (chunk := await self._queue.get()) is not None:
            yield chunk


class TurnStore:
    """Keep pipeline handoffs isolated and correlate them by their text."""

    def __init__(self) -> None:
        """Initialize the turn store."""
        self._voice_turns: deque[PipelineTurn] = deque(maxlen=100)
        self._audio: deque[tuple[str, bytes | AudioStream]] = deque(maxlen=100)
        self._streaming_audio: deque[tuple[TextStream, AudioStream]] = deque(maxlen=100)

    def add_voice_turn(self, turn: PipelineTurn) -> None:
        """Store a voice turn for the conversation stage."""
        self._discard_stale_voice_audio()
        self._voice_turns = deque(
            (
                existing_turn
                for existing_turn in self._voice_turns
                if existing_turn.conversation_id != turn.conversation_id
            ),
            maxlen=100,
        )
        self._voice_turns.append(turn)

    def _discard_stale_voice_audio(self) -> None:
        """Discard unconsumed audio left by an earlier voice pipeline turn."""
        for _text, audio in self._audio:
            if isinstance(audio, AudioStream):
                audio.finish()
        for _text_stream, audio_stream in self._streaming_audio:
            audio_stream.finish()
        self._audio = deque(
            (
                (text, audio)
                for text, audio in self._audio
                if not isinstance(audio, AudioStream)
            ),
            maxlen=100,
        )
        self._streaming_audio.clear()

    def take_voice_turn(
        self,
        conversation_id: str,
        user_text: str,
    ) -> PipelineTurn | None:
        """Take the oldest completed voice turn matching the STT transcript."""
        for index, turn in enumerate(self._voice_turns):
            if turn.user_text != user_text:
                continue
            if turn.conversation_id == conversation_id:
                del self._voice_turns[index]
                return turn
        return None

    def add_audio(self, assistant_text: str, audio: bytes | AudioStream) -> None:
        """Store one response's audio for the TTS stage."""
        if audio:
            self._audio.append((assistant_text, audio))

    def take_audio(self, assistant_text: str) -> bytes | AudioStream | None:
        """Take the oldest audio response matching the TTS message."""
        for index, (text, audio) in enumerate(self._audio):
            if text == assistant_text:
                del self._audio[index]
                return audio
        return None

    def add_streaming_audio(self, text: TextStream, audio: AudioStream) -> None:
        """Store audio waiting for Home Assistant's streaming TTS input."""
        self._streaming_audio.append((text, audio))
        _LOGGER.debug(
            "Queued streaming audio for transcript prefix %r",
            text.text[:80],
        )

    def take_streaming_audio(self, initial_text: str) -> AudioStream | None:
        """Take streaming audio whose transcript matches the TTS input."""
        for index, (text_stream, audio_stream) in enumerate(self._streaming_audio):
            transcript = text_stream.text
            if transcript and (
                transcript.startswith(initial_text) or initial_text.startswith(transcript)
            ):
                del self._streaming_audio[index]
                _LOGGER.debug(
                    "Matched streaming audio for TTS input prefix %r",
                    initial_text[:80],
                )
                return audio_stream
        _LOGGER.debug(
            "No streaming audio matched TTS input prefix %r",
            initial_text[:80],
        )
        return None


@dataclass(slots=True)
class _LiveConnection:
    """An open Gemini Live connection."""

    context_manager: Any
    session: Any
    config_signature: str


class LiveSessionManager:
    """Own one open Gemini Live connection per Home Assistant conversation."""

    def __init__(self) -> None:
        """Initialize the manager."""
        self._connections: dict[str, _LiveConnection] = {}
        self._completed_conversations: set[str] = set()
        self._conversation_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._cleanup_registered: set[str] = set()

    @asynccontextmanager
    async def acquire(
        self,
        conversation_id: str,
        client: Any,
        model: str,
        config: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Yield the conversation's open session, reconnecting when necessary."""
        signature = self._config_signature(model, config)
        conversation_lock = self._lock_for(conversation_id)
        async with conversation_lock:
            connection = self._connections.get(conversation_id)
            if connection and (
                connection.config_signature != signature
                or not self._is_open(connection.session)
            ):
                await self._async_close(conversation_id, connection)
                connection = None
            if connection is None:
                context_manager = client.aio.live.connect(model=model, config=config)
                session = await context_manager.__aenter__()
                connection = _LiveConnection(
                    context_manager=context_manager,
                    session=session,
                    config_signature=signature,
                )
                self._connections[conversation_id] = connection
                _LOGGER.info(
                    "Opened Gemini Live session for conversation %s with config %s",
                    conversation_id,
                    signature[:12],
                )

            try:
                yield connection.session
            except Exception:
                await self._async_close(conversation_id, connection)
                raise

    async def async_close_all(self) -> None:
        """Close every open Live connection."""
        for conversation_id, connection in list(self._connections.items()):
            await self._async_close(conversation_id, connection)
        self._completed_conversations.clear()

    def complete_conversation(self, conversation_id: str) -> None:
        """Mark one conversation as completed by Gemini."""
        self._completed_conversations.add(conversation_id)

    def reset_conversation(self, conversation_id: str) -> None:
        """Reset completed conversation status for a new user input."""
        self._completed_conversations.discard(conversation_id)

    def should_continue_conversation(self, conversation_id: str) -> bool:
        """Return whether Home Assistant should keep listening."""
        return conversation_id not in self._completed_conversations

    def register_chat_session(
        self,
        hass: HomeAssistant,
        chat_session: Any,
    ) -> None:
        """Close the Live connection when Home Assistant expires the chat."""
        conversation_id = chat_session.conversation_id
        if conversation_id in self._cleanup_registered:
            return
        self._cleanup_registered.add(conversation_id)

        def close_live_session() -> None:
            self._cleanup_registered.discard(conversation_id)
            hass.async_create_task(
                self.async_close(conversation_id),
                f"close Gemini Live conversation {conversation_id}",
            )

        chat_session.async_on_cleanup(close_live_session)

    async def async_close(self, conversation_id: str) -> None:
        """Close one conversation's Live connection and discard its state."""
        async with self._lock_for(conversation_id):
            connection = self._connections.get(conversation_id)
            if connection is not None:
                await self._async_close(conversation_id, connection)
            self._completed_conversations.discard(conversation_id)

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        """Return the serialization lock for a conversation."""
        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        return lock

    @staticmethod
    def _is_open(session: Any) -> bool:
        """Return whether the SDK's websocket is still open, when observable."""
        websocket = getattr(session, "_ws", None)
        state = getattr(websocket, "state", None)
        state_name = getattr(state, "name", None)
        return state_name is None or state_name == "OPEN"

    @staticmethod
    def _config_signature(model: str, config: dict[str, Any]) -> str:
        """Return a stable signature for settings fixed when a Live session opens."""
        payload = json.dumps(
            {"model": model, "config": config},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _async_close(
        self,
        conversation_id: str,
        connection: _LiveConnection,
    ) -> None:
        """Close a connection if it is still the active one."""
        if self._connections.get(conversation_id) is not connection:
            return
        self._connections.pop(conversation_id, None)
        try:
            await connection.context_manager.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "Error closing Gemini Live session for conversation %s",
                conversation_id,
                exc_info=True,
            )


def new_conversation_id() -> str:
    """Return an integration-owned ID when Home Assistant did not supply one."""
    return uuid4().hex


def active_pipeline_conversation_id(hass: HomeAssistant, stt_entity_id: str) -> str:
    """Resolve the conversation ID for the active STT pipeline run.

    Home Assistant keeps this ID on PipelineRun but does not currently pass it
    through SpeechMetadata. The guarded introspection can be removed when the
    public STT API exposes a pipeline or conversation identifier.
    """
    try:
        from homeassistant.components.assist_pipeline.pipeline import (  # noqa: PLC0415
            KEY_ASSIST_PIPELINE,
            PipelineEventType,
        )

        pipeline_data = hass.data[KEY_ASSIST_PIPELINE]
        candidates: list[tuple[str, str]] = []
        for runs in pipeline_data.pipeline_runs._pipeline_runs.values():
            for run in runs.values():
                provider = getattr(run, "stt_provider", None)
                if getattr(provider, "entity_id", None) != stt_entity_id:
                    continue
                debug = pipeline_data.pipeline_debug.get(run.pipeline.id, {}).get(
                    run.id
                )
                if debug is None:
                    continue
                conversation_id: str | None = None
                stt_started: str | None = None
                stt_ended = False
                for event in debug.events:
                    if event.type == PipelineEventType.RUN_START and event.data:
                        conversation_id = event.data.get("conversation_id")
                    elif event.type == PipelineEventType.STT_START:
                        stt_started = event.timestamp
                        stt_ended = False
                    elif event.type == PipelineEventType.STT_END:
                        stt_ended = True
                if conversation_id and stt_started and not stt_ended:
                    candidates.append((stt_started, conversation_id))
        if candidates:
            return max(candidates)[1]
    except Exception:  # noqa: BLE001
        _LOGGER.debug(
            "Could not resolve active pipeline conversation ID",
            exc_info=True,
        )

    conversation_id = new_conversation_id()
    _LOGGER.warning(
        "Home Assistant did not expose the active STT conversation ID; "
        "using temporary conversation %s",
        conversation_id,
    )
    return conversation_id
