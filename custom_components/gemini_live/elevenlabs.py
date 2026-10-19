"""ElevenLabs Agents WebSocket adapter for the shared live pipeline.

ElevenLabs Agents is a cascaded service: ElevenLabs transcribes the user,
queries the configured LLM, and synthesizes the reply with an ElevenLabs voice,
all within one streaming WebSocket session. The integration owns a dedicated
agent whose base configuration fixes the audio formats and client events; the
system instruction, LLM, voice, and tools are applied per conversation through
conversation overrides.

Protocol: https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket
Audio is raw mono PCM16: 16 kHz input, 24 kHz output.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator, Callable, MutableMapping
import hashlib
import json
import logging
import re
import time
from typing import Any

from aiohttp import ClientError, ClientSession, ClientWebSocketResponse, WSMsgType

from .const import ELEVENLABS_SUPPORTED_LANGUAGES
from .live import LiveConfig, LiveEvent, LiveTool, LiveToolCall, LiveToolResponse

_LOGGER = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io"
_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"
_INPUT_FORMAT = "pcm_16000"
_OUTPUT_FORMAT = "pcm_24000"
_CLIENT_EVENTS = [
    "conversation_initiation_metadata",
    "ping",
    "audio",
    "interruption",
    "user_transcript",
    "agent_response",
    "client_tool_call",
]
# Expressive TTS models speak tags such as "[happy]" as emotion, not words,
# but they remain in the agent's text response.
_AUDIO_TAG = re.compile(r"\[[^\]\n]{1,40}\]\s*")
# Alignment data is optional per audio event. If the aligned characters never
# cover the final response text, finish the turn once audio has been idle
# for this long after the text arrived. Audio is delivered faster than
# real time, so a gap this long means the reply has been sent completely.
_COMPLETION_IDLE_TIMEOUT = 1.5
# A reply whose text arrived but whose audio never starts is given up on after
# this long, instead of waiting for the pipeline's inactivity timeout.
_NO_AUDIO_TIMEOUT = 10.0
_TOOL_RESPONSE_TIMEOUT = 20
_API_TIMEOUT = 15
# Close reason when a conversation selects tools the agent does not have.
_TOOLS_NOT_ATTACHED = "Tool IDs not attached"
# ElevenLabs detects the end of a turn from the audio it receives and has no
# commit message. When the microphone stream ends, keep feeding real-time
# silence until the agent responds, or its turn model may wait indefinitely.
_SILENCE_CHUNK = b"\x00" * 3200  # 100 ms of 16 kHz mono PCM16
_SILENCE_INTERVAL = 0.1
_MAX_TRAILING_SILENCE = 10.0
# ElevenLabs bills a conversation from connect to disconnect, while the
# pipeline keeps a session open between turns for follow-up questions. Close an
# idle session after this long; the next turn then opens a new conversation.
# The agent's own silence_end_call_timeout does not fire once the microphone
# stream has stopped.
_IDLE_CLOSE_TIMEOUT = 20.0
# A session this close to its idle close reports itself closed, so a starting
# turn opens a new conversation instead of racing the close.
_IDLE_CLOSE_MARGIN = 2.0
# After a tool result that answered with a sound instead of speech, the agent
# stays silent and sends no further event. ElevenLabs calls tools in parallel,
# so wait this long for more calls of the same step before ending the turn.
_SOUND_RESPONSE_GRACE = 0.5
_SILENCE_TURN_TIMEOUT = 4
_INITIAL_WAIT_TIME = 8
# When the user stays silent, ElevenLabs sends a user turn whose transcript
# has no words (typically "...") and lets the LLM re-engage ("Are you still
# there?"). Such a turn means the user said nothing.
_WORD = re.compile(r"\w")
# Concurrent turns must not create the same tool twice.
_SYNC_LOCKS: dict[str, asyncio.Lock] = {}


class ElevenLabsAgentsError(Exception):
    """An ElevenLabs Agents protocol, connection, or API error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def agent_language(language: str | None) -> str:
    """Return the agent language for a Home Assistant language tag."""
    primary = (language or "en").split("-")[0].lower()
    return primary if primary in ELEVENLABS_SUPPORTED_LANGUAGES else "en"


def agent_base_config(language: str) -> dict[str, Any]:
    """Return the agent settings the adapter relies on.

    Everything that varies per conversation is supplied as an override, so
    the agent only needs to permit those overrides.
    """
    return {
        "conversation_config": {
            "asr": {"user_input_audio_format": _INPUT_FORMAT},
            "tts": {"agent_output_audio_format": _OUTPUT_FORMAT},
            "turn": {
                "turn_eagerness": "eager",
                # How long the agent waits in silence before it takes the
                # turn itself. The adapter turns that silent turn into "no
                # speech", so this is how fast an unanswered follow-up window
                # closes. Right after the wake word, the user gets longer.
                "turn_timeout": _SILENCE_TURN_TIMEOUT,
                "initial_wait_time": _INITIAL_WAIT_TIME,
            },
            "conversation": {"client_events": _CLIENT_EVENTS},
            "agent": {"first_message": "", "language": language},
        },
        "platform_settings": {
            "overrides": {
                "conversation_config_override": {
                    "agent": {
                        "prompt": {"prompt": True, "llm": True, "tool_ids": True},
                    },
                    "tts": {"voice_id": True},
                }
            }
        },
    }


async def async_create_agent(
    http_session: ClientSession, api_key: str, name: str, language: str
) -> str:
    """Create the dedicated agent and return its ID."""
    body = await _api_request(
        http_session,
        api_key,
        "post",
        "/v1/convai/agents/create",
        {"name": name, **agent_base_config(language)},
    )
    if not isinstance(body.get("agent_id"), str):
        raise ElevenLabsAgentsError("ElevenLabs did not return an agent ID")
    return body["agent_id"]


async def async_update_agent(
    http_session: ClientSession, api_key: str, agent_id: str, language: str
) -> None:
    """Reapply the base configuration to an existing agent."""
    await _api_request(
        http_session,
        api_key,
        "patch",
        f"/v1/convai/agents/{agent_id}",
        agent_base_config(language),
    )


async def async_delete_agent(
    http_session: ClientSession, api_key: str, agent_id: str
) -> None:
    """Delete the dedicated agent."""
    await _api_request(
        http_session, api_key, "delete", f"/v1/convai/agents/{agent_id}"
    )


async def async_list_voices(
    http_session: ClientSession, api_key: str
) -> list[tuple[str, str]]:
    """Return (voice_id, name) pairs available to the account."""
    voices: list[tuple[str, str]] = []
    params: dict[str, Any] = {"page_size": 100}
    while True:
        body = await _api_request(
            http_session, api_key, "get", "/v2/voices", params=params
        )
        voices.extend(
            (voice["voice_id"], voice.get("name") or voice["voice_id"])
            for voice in body.get("voices", [])
            if voice.get("voice_id")
        )
        if not body.get("has_more") or not body.get("next_page_token"):
            return voices
        params = {"page_size": 100, "next_page_token": body["next_page_token"]}


async def async_delete_tools(
    http_session: ClientSession, api_key: str, cache: MutableMapping[str, Any]
) -> None:
    """Delete every tool recorded in the cache."""
    tools = cache.get("tools", {})
    for entry in list(tools.values()):
        try:
            await _api_request(
                http_session, api_key, "delete", f"/v1/convai/tools/{entry['id']}"
            )
        except ElevenLabsAgentsError:
            # A tool deleted from the ElevenLabs dashboard is already gone.
            pass
    tools.clear()
    cache.pop("agent_tool_ids", None)


def tool_config(tool: LiveTool) -> dict[str, Any]:
    """Return an ElevenLabs client tool definition for a live tool."""
    config: dict[str, Any] = {
        "type": "client",
        "name": tool.name,
        "description": tool.description or tool.name,
        "expects_response": True,
        "response_timeout_secs": _TOOL_RESPONSE_TIMEOUT,
        # Home Assistant tools run in milliseconds. A spoken preamble would
        # only delay the answer.
        "pre_tool_speech": "off",
    }
    if tool.parameters:
        config["parameters"] = _parameters(tool.parameters)
    return config


class ElevenLabsAgentsClient:
    """Create configured sessions for one ElevenLabs agent.

    A conversation can only use tools that are attached to its agent, so Home
    Assistant's tools are mirrored as ElevenLabs client tools. Every mirrored
    tool stays attached to the dedicated agent, and each conversation selects
    the tools of its own turn through an override, so turns with different
    tool sets never change the agent under each other. ``cache`` records each
    tool's ID and fingerprint under ``"tools"`` and the attached IDs under
    ``"agent_tool_ids"``, so the API is only called when Home Assistant's tool
    definitions change. ``on_cache_change`` is called after the cache was
    modified so the caller can persist it.
    """

    def __init__(
        self,
        session: ClientSession,
        api_key: str,
        agent_id: str,
        cache: MutableMapping[str, Any] | None = None,
        on_cache_change: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._api_key = api_key
        self._agent_id = agent_id
        self._cache: MutableMapping[str, Any] = {} if cache is None else cache
        self._on_cache_change = on_cache_change

    def connect(self, config: LiveConfig) -> _ElevenLabsConnect:
        return _ElevenLabsConnect(self, config)

    async def async_sync_tools(self, tools: list[LiveTool]) -> list[str]:
        """Mirror tools as attached client tools and return this turn's IDs."""
        async with _SYNC_LOCKS.setdefault(self._agent_id, asyncio.Lock()):
            changed = False
            try:
                if self._cache.get("agent_id") != self._agent_id:
                    # The entry got a new agent. Its old tools are only reachable
                    # if the API key still names the same account.
                    await async_delete_tools(self._session, self._api_key, self._cache)
                    self._cache.clear()
                    self._cache["agent_id"] = self._agent_id
                    changed = True
                tool_cache: MutableMapping[str, Any] = self._cache.setdefault(
                    "tools", {}
                )
                try:
                    async with asyncio.TaskGroup() as group:
                        tasks = [
                            group.create_task(self._async_sync_tool(tool_cache, tool))
                            for tool in tools
                        ]
                except BaseExceptionGroup as group_error:
                    changed = True
                    raise _first_error(group_error) from group_error
                results = [task.result() for task in tasks]
                changed |= any(updated for _tool_id, updated in results)
                attached = sorted({entry["id"] for entry in tool_cache.values()})
                if attached != sorted(self._cache.get("agent_tool_ids", ())):
                    await _api_request(
                        self._session,
                        self._api_key,
                        "patch",
                        f"/v1/convai/agents/{self._agent_id}",
                        {"conversation_config": {"agent": {"prompt": {"tool_ids": attached}}}},
                    )
                    self._cache["agent_tool_ids"] = attached
                    changed = True
                return [tool_id for tool_id, _updated in results]
            finally:
                if changed and self._on_cache_change is not None:
                    self._on_cache_change()

    async def _async_sync_tool(
        self, tool_cache: MutableMapping[str, Any], tool: LiveTool
    ) -> tuple[str, bool]:
        """Return the tool's ID and whether it was created or updated."""
        config = tool_config(tool)
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        cached = tool_cache.get(tool.name)
        if cached and cached.get("fingerprint") == fingerprint:
            return cached["id"], False
        body = {"tool_config": config}
        if cached:
            try:
                await _api_request(
                    self._session,
                    self._api_key,
                    "patch",
                    f"/v1/convai/tools/{cached['id']}",
                    body,
                )
            except ElevenLabsAgentsError as exc:
                if exc.status != 404:
                    raise
                # Deleted outside Home Assistant; create it again below.
            else:
                tool_cache[tool.name] = {"id": cached["id"], "fingerprint": fingerprint}
                return cached["id"], True
        created = await _api_request(
            self._session, self._api_key, "post", "/v1/convai/tools", body
        )
        if not isinstance(created.get("id"), str):
            raise ElevenLabsAgentsError("ElevenLabs did not return a tool ID")
        tool_cache[tool.name] = {"id": created["id"], "fingerprint": fingerprint}
        return created["id"], True

    def forget_attached_tools(self) -> None:
        """Make the next sync attach the tools to the agent again."""
        self._cache.pop("agent_tool_ids", None)
        if self._on_cache_change is not None:
            self._on_cache_change()

    async def async_open_websocket(self) -> ClientWebSocketResponse:
        """Open the conversation WebSocket for the agent."""
        try:
            return await self._session.ws_connect(
                _WS_URL,
                params={"agent_id": self._agent_id},
                headers={"xi-api-key": self._api_key},
                heartbeat=20,
            )
        except ClientError as exc:
            raise ElevenLabsAgentsError(
                f"Could not connect to ElevenLabs Agents: {exc}"
            ) from exc


class _ElevenLabsConnect:
    def __init__(self, client: ElevenLabsAgentsClient, config: LiveConfig) -> None:
        self._client = client
        self._config = config
        self._session: ElevenLabsAgentsSession | None = None

    async def __aenter__(self) -> ElevenLabsAgentsSession:
        # Not under a timeout: cancelling a sync could orphan tools that were
        # created but not yet recorded. Each API request has its own timeout.
        tool_ids = await self._client.async_sync_tools(self._config.tools)
        async with asyncio.timeout(15):
            websocket = await self._client.async_open_websocket()
        self._session = ElevenLabsAgentsSession(
            websocket, on_tools_rejected=self._client.forget_attached_tools
        )
        try:
            async with asyncio.timeout(15):
                await self._session.async_configure(self._config, tool_ids)
        except BaseException:
            await self._session.async_close()
            raise
        return self._session

    async def __aexit__(self, *_exc: Any) -> None:
        if self._session is not None:
            await self._session.async_close()


class ElevenLabsAgentsSession:
    """Translate ElevenLabs Agents events without exposing its protocol."""

    def __init__(
        self,
        websocket: ClientWebSocketResponse,
        on_tools_rejected: Callable[[], None] | None = None,
    ) -> None:
        self._ws = websocket
        self._on_tools_rejected = on_tools_rejected
        self._transcribe_output = True
        self.conversation_id: str | None = None
        self.last_outgoing = "none"
        # Whether the current user turn already got a response. Until then,
        # end_audio keeps feeding silence.
        self._responded = asyncio.Event()
        # Whether events belong to the current turn. After a turn completes,
        # stray audio or text from that turn must not become the next answer.
        self._accepting = True
        self._silence_task: asyncio.Task[None] | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        # A tool result replaced the spoken reply with a sound.
        self._sound_sent = False
        self._idle_deadline: float | None = None
        self._close_task: asyncio.Task[None] | None = None

    @property
    def is_open(self) -> bool:
        if self._ws.closed:
            return False
        return (
            self._idle_deadline is None
            or time.monotonic() < self._idle_deadline - _IDLE_CLOSE_MARGIN
        )

    async def async_configure(
        self, config: LiveConfig, tool_ids: list[str] | None = None
    ) -> None:
        self._transcribe_output = config.transcribe_output
        prompt: dict[str, Any] = {
            "prompt": config.system_instruction,
            "llm": config.model,
        }
        if tool_ids is not None:
            prompt["tool_ids"] = tool_ids
        override: dict[str, Any] = {"agent": {"prompt": prompt}}
        if config.voice:
            override["tts"] = {"voice_id": config.voice}
        await self._send(
            {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": override,
            }
        )
        while True:
            event = await self._receive_event()
            kind = event.get("type")
            if kind == "conversation_initiation_metadata":
                metadata = event.get("conversation_initiation_metadata_event", {})
                if (
                    metadata.get("user_input_audio_format") != _INPUT_FORMAT
                    or metadata.get("agent_output_audio_format") != _OUTPUT_FORMAT
                ):
                    raise ElevenLabsAgentsError(
                        "The ElevenLabs agent uses unexpected audio formats "
                        f"({metadata.get('user_input_audio_format')} in, "
                        f"{metadata.get('agent_output_audio_format')} out); "
                        "reconfigure the integration to repair the agent"
                    )
                self.conversation_id = metadata.get("conversation_id")
                return
            if kind == "ping":
                await self._pong(event)
            elif kind == "error":
                raise _error(event)

    async def send_audio(self, audio: bytes) -> None:
        self._cancel_idle_close()
        # The shared STT stage already supplies 16 kHz PCM. Do not resample it.
        await self._ws.send_json(
            {"user_audio_chunk": base64.b64encode(audio).decode("ascii")}
        )
        self.last_outgoing = f"user_audio_chunk ({len(audio)} bytes)"

    async def end_audio(self) -> None:
        if self._responded.is_set() or (
            self._silence_task is not None and not self._silence_task.done()
        ):
            return
        self._silence_task = asyncio.create_task(self._send_trailing_silence())

    async def _send_trailing_silence(self) -> None:
        silence = base64.b64encode(_SILENCE_CHUNK).decode("ascii")
        loop = asyncio.get_running_loop()
        start = loop.time()
        sent = 0
        while not self._responded.is_set() and not self._ws.closed:
            if sent * _SILENCE_INTERVAL >= _MAX_TRAILING_SILENCE:
                return
            try:
                await self._ws.send_json({"user_audio_chunk": silence})
            except (ClientError, ConnectionError, RuntimeError):
                # The session is closing; its receiver reports the closure.
                return
            sent += 1
            # Pace by elapsed time so the stream stays real time.
            delay = start + sent * _SILENCE_INTERVAL - loop.time()
            try:
                await asyncio.wait_for(self._responded.wait(), max(delay, 0))
            except TimeoutError:
                pass

    async def send_text(self, text: str) -> None:
        self._cancel_idle_close()
        self._responded.clear()
        self._accepting = True
        await self._send({"type": "user_message", "text": text})

    async def send_tool_responses(self, responses: list[LiveToolResponse]) -> None:
        if any(
            isinstance(result.response, dict)
            and result.response.get("response_mode") == "sound"
            for result in responses
        ):
            self._sound_sent = True
        for result in responses:
            if not result.call_id:
                raise ElevenLabsAgentsError("ElevenLabs tool response has no call ID")
            await self._send(
                {
                    "type": "client_tool_result",
                    "tool_call_id": result.call_id,
                    "result": json.dumps(
                        result.response, ensure_ascii=False, separators=(",", ":")
                    ),
                    "is_error": _is_error(result.response),
                }
            )

    async def receive(self) -> AsyncIterator[LiveEvent]:
        # ElevenLabs sends the final text ("agent_response") and the audio
        # chunks in either order, and has no event that marks the end of the
        # audio. Each audio event carries the characters it speaks, so the
        # reply is complete once those cover the whole response text.
        final_text: str | None = None
        aligned_chars = 0
        audio_received = False
        while True:
            if self._sound_sent:
                timeout = _SOUND_RESPONSE_GRACE
            elif final_text is None:
                timeout = None
            elif audio_received:
                timeout = _COMPLETION_IDLE_TIMEOUT
            else:
                timeout = _NO_AUDIO_TIMEOUT
            try:
                event = await self._receive_event(timeout)
            except TimeoutError:
                final_text, aligned_chars, audio_received = None, 0, False
                yield self._turn_complete()
                continue
            kind = event.get("type")
            if kind == "ping":
                await self._pong(event)
            elif kind == "error":
                raise _error(event)
            elif kind == "user_transcript":
                # A new user turn: everything before it is settled.
                final_text, aligned_chars, audio_received = None, 0, False
                self._mark_responded()
                transcript = event.get("user_transcription_event", {}).get(
                    "user_transcript"
                )
                if not isinstance(transcript, str) or not _WORD.search(transcript):
                    # The user stayed silent. Drop the agent's re-engagement
                    # and let the pipeline stop listening.
                    yield self._silent_turn()
                    continue
                self._accepting = True
                yield LiveEvent(input_transcript=transcript)
            elif not self._accepting:
                # Late audio or text of a completed turn.
                continue
            elif kind == "audio":
                audio_event = event.get("audio_event", {})
                try:
                    audio = base64.b64decode(
                        audio_event.get("audio_base_64", ""), validate=True
                    )
                except (ValueError, binascii.Error) as exc:
                    raise ElevenLabsAgentsError(
                        "Invalid ElevenLabs response audio"
                    ) from exc
                self._mark_responded()
                audio_received = True
                aligned_chars += len(
                    (audio_event.get("alignment") or {}).get("chars") or ()
                )
                if audio:
                    yield LiveEvent(audio=audio)
                if final_text is not None and aligned_chars >= len(final_text):
                    final_text, aligned_chars, audio_received = None, 0, False
                    yield self._turn_complete()
            elif kind == "agent_response":
                text = event.get("agent_response_event", {}).get("agent_response")
                if not isinstance(text, str):
                    continue
                self._mark_responded()
                if self._transcribe_output and (spoken := strip_audio_tags(text)):
                    yield LiveEvent(output_transcript=spoken)
                if aligned_chars >= len(text):
                    aligned_chars, audio_received = 0, False
                    yield self._turn_complete()
                else:
                    final_text = text
            elif kind == "client_tool_call":
                call = _tool_call(event)
                self._mark_responded()
                # Speech before a tool call belongs to an earlier generation.
                final_text, aligned_chars, audio_received = None, 0, False
                yield LiveEvent(tool_calls=[call])
            elif kind == "interruption":
                # The agent stopped speaking; a new reply follows.
                final_text, aligned_chars, audio_received = None, 0, False

    async def async_close(self) -> None:
        self._cancel_idle_close()
        self._stop_trailing_silence()
        if not self._ws.closed:
            await self._ws.close()

    def _turn_complete(self) -> LiveEvent:
        # The next user turn needs a fresh response before silence stops.
        self._responded.clear()
        self._sound_sent = False
        self._accepting = False
        self._schedule_idle_close()
        return LiveEvent(turn_complete=True)

    def _silent_turn(self) -> LiveEvent:
        self._responded.clear()
        self._accepting = False
        self._schedule_idle_close()
        return LiveEvent(no_speech=True)

    def _schedule_idle_close(self) -> None:
        self._cancel_idle_close()
        loop = asyncio.get_running_loop()
        self._idle_deadline = time.monotonic() + _IDLE_CLOSE_TIMEOUT
        self._idle_timer = loop.call_later(_IDLE_CLOSE_TIMEOUT, self._close_when_idle)

    def _close_when_idle(self) -> None:
        self._idle_timer = None
        self._close_task = asyncio.get_running_loop().create_task(self._async_idle_close())

    async def _async_idle_close(self) -> None:
        try:
            await self.async_close()
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Closing an idle ElevenLabs session failed", exc_info=True)

    def _cancel_idle_close(self) -> None:
        self._idle_deadline = None
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _mark_responded(self) -> None:
        self._responded.set()
        self._stop_trailing_silence()

    def _stop_trailing_silence(self) -> None:
        if self._silence_task is not None:
            self._silence_task.cancel()
            self._silence_task = None

    async def _pong(self, event: dict[str, Any]) -> None:
        await self._send(
            {"type": "pong", "event_id": event.get("ping_event", {}).get("event_id")}
        )

    async def _send(self, event: dict[str, Any]) -> None:
        await self._ws.send_json(event)
        self.last_outgoing = str(event.get("type"))

    async def _receive_event(self, timeout: float | None = None) -> dict[str, Any]:
        async with asyncio.timeout(timeout):
            message = await self._ws.receive()
        if message.type == WSMsgType.TEXT:
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ElevenLabsAgentsError("Invalid ElevenLabs event JSON") from exc
            if isinstance(event, dict):
                return event
            raise ElevenLabsAgentsError("Invalid ElevenLabs event object")
        if message.type == WSMsgType.ERROR:
            raise ElevenLabsAgentsError(
                "ElevenLabs WebSocket transport error"
            ) from self._ws.exception()
        reason = f": {message.extra}" if message.extra else ""
        if _TOOLS_NOT_ATTACHED in reason and self._on_tools_rejected is not None:
            # The agent's tools were changed outside Home Assistant, e.g. in
            # the ElevenLabs dashboard. The next turn attaches them again.
            self._on_tools_rejected()
        raise ElevenLabsAgentsError(
            f"ElevenLabs Agents connection closed ({message.type}){reason}"
        )


def strip_audio_tags(text: str) -> str:
    """Remove expressive audio tags such as "[happy]" from response text."""
    return _AUDIO_TAG.sub("", text).strip()


def _parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Home Assistant tool schema to ElevenLabs' schema subset.

    ElevenLabs accepts objects, arrays, and literals, requires every literal
    to describe the value the LLM provides, and rejects schema combinators.
    As in the other adapters, top-level unions become one object holding the
    union of their properties; Home Assistant validates the complete schema
    when it executes the call.
    """
    combinators = ("oneOf", "anyOf", "allOf")
    properties = dict(schema.get("properties", {}))
    required = set(schema.get("required", ()))
    for keyword in combinators:
        for branch in schema.get(keyword, ()):
            if not isinstance(branch, dict):
                continue
            properties.update(branch.get("properties", {}))
            if keyword == "allOf":
                required.update(branch.get("required", ()))
    result: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: _property(name, value) for name, value in properties.items()
        },
    }
    if description := schema.get("description"):
        result["description"] = description
    if required:
        result["required"] = sorted(required & set(properties))
    return result


def _property(name: str, schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "string", "description": name}
    for keyword in ("anyOf", "oneOf"):
        branches = [b for b in schema.get(keyword, ()) if isinstance(b, dict)]
        if branches and "type" not in schema:
            # Prefer the first concrete branch; HA validates the real union.
            merged = {**branches[0], **{k: v for k, v in schema.items() if k != keyword}}
            return _property(name, merged)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    description = schema.get("description") or name.replace("_", " ")
    if kind == "object":
        converted = _parameters(schema)
        converted["description"] = description
        return converted
    if kind == "array":
        return {
            "type": "array",
            "description": description,
            "items": _property(name, schema.get("items") or {"type": "string"}),
        }
    if kind not in ("string", "integer", "number", "boolean"):
        kind = "string"
    result: dict[str, Any] = {"type": kind, "description": description}
    enum = schema.get("enum")
    if kind == "string" and isinstance(enum, list) and enum:
        result["enum"] = [str(value) for value in enum]
    return result


def _tool_call(event: dict[str, Any]) -> LiveToolCall:
    call = event.get("client_tool_call", {})
    arguments = call.get("parameters") or {}
    if (
        not isinstance(arguments, dict)
        or not call.get("tool_call_id")
        or not call.get("tool_name")
    ):
        raise ElevenLabsAgentsError("Invalid ElevenLabs tool call")
    return LiveToolCall(
        name=call["tool_name"], call_id=call["tool_call_id"], arguments=arguments
    )


def _is_error(response: Any) -> bool:
    """Return whether a Home Assistant tool result reports a failure."""
    return isinstance(response, dict) and (
        "error" in response
        or response.get("success") is False
        or response.get("response_type") == "error"
    )


def _first_error(group: BaseExceptionGroup) -> BaseException:
    """Return the first leaf exception of a task group failure."""
    error: BaseException = group
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


def _error(event: dict[str, Any]) -> ElevenLabsAgentsError:
    error = event.get("error") or event.get("error_event") or {}
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
    else:
        message = str(error)
    return ElevenLabsAgentsError(str(message or "ElevenLabs Agents request failed"))


async def _api_request(
    http_session: ClientSession,
    api_key: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(_API_TIMEOUT), http_session.request(
            method,
            f"{API_URL}{path}",
            headers={"xi-api-key": api_key},
            json=body,
            params=params,
        ) as response:
            if response.status >= 400:
                detail = await response.text()
                raise ElevenLabsAgentsError(
                    f"ElevenLabs API {method.upper()} {path} failed "
                    f"({response.status}): {detail[:300]}",
                    response.status,
                )
            if response.content_type != "application/json":
                return {}
            data = await response.json()
    except (ClientError, TimeoutError) as exc:
        raise ElevenLabsAgentsError(f"ElevenLabs API request failed: {exc!r}") from exc
    return data if isinstance(data, dict) else {}
