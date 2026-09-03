"""OpenAI Realtime adapter for the provider-neutral live contract."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
import json
from typing import Any

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType

from .live import LiveConfig, LiveEvent, LiveToolCall, LiveToolResponse
from .utils import resample_16k_to_24k

_REALTIME_URL = "wss://api.openai.com/v1/realtime"
_TRANSCRIPTION_MODEL = "gpt-live-transcribe"


class OpenAIRealtimeError(Exception):
    """An error event returned by the OpenAI Realtime API."""


class OpenAIRealtimeClient:
    """Create normalized sessions backed by an OpenAI Realtime WebSocket."""

    def __init__(self, session: ClientSession, api_key: str) -> None:
        self._session = session
        self._api_key = api_key

    def connect(self, config: LiveConfig) -> _OpenAIConnect:
        return _OpenAIConnect(self._session, self._api_key, config)


class _OpenAIConnect:
    def __init__(
        self,
        http_session: ClientSession,
        api_key: str,
        config: LiveConfig,
    ) -> None:
        self._http_session = http_session
        self._api_key = api_key
        self._config = config
        self._session: OpenAIRealtimeSession | None = None

    async def __aenter__(self) -> OpenAIRealtimeSession:
        websocket = await self._http_session.ws_connect(
            _REALTIME_URL,
            params={"model": self._config.model},
            headers={"Authorization": f"Bearer {self._api_key}"},
            heartbeat=20,
        )
        self._session = OpenAIRealtimeSession(websocket)
        try:
            async with asyncio.timeout(15):
                await self._session.async_configure(self._config)
        except BaseException:
            await self._session.async_close()
            raise
        return self._session

    async def __aexit__(self, *_exc: Any) -> None:
        if self._session is not None:
            await self._session.async_close()


class OpenAIRealtimeSession:
    """Translate OpenAI client/server events to the neutral contract."""

    def __init__(self, websocket: ClientWebSocketResponse) -> None:
        self._ws = websocket
        self._transcribe_output = False
        self._tool_calls_seen: set[str] = set()

    @property
    def is_open(self) -> bool:
        return not self._ws.closed

    async def async_configure(self, config: LiveConfig) -> None:
        self._transcribe_output = config.transcribe_output
        tools = []
        for declaration in config.tools:
            tool: dict[str, Any] = {
                "type": "function",
                "name": declaration.name,
                "description": declaration.description,
            }
            if declaration.parameters:
                tool["parameters"] = declaration.parameters
            tools.append(tool)

        await self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": config.model,
                "output_modalities": ["audio"],
                "instructions": config.system_instruction,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                        "transcription": {"model": _TRANSCRIPTION_MODEL},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": config.voice,
                    },
                },
                "tools": tools,
                "tool_choice": "auto",
            },
        })
        while True:
            event = await self._receive_event()
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise _error_from_event(event)

    async def send_audio(self, audio: bytes) -> None:
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(resample_16k_to_24k(audio)).decode("ascii"),
        })

    async def end_audio(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    async def send_text(self, text: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        })
        await self._send({"type": "response.create"})

    async def send_tool_responses(self, responses: list[LiveToolResponse]) -> None:
        for result in responses:
            await self._send({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": json.dumps(result.response, separators=(",", ":")),
                },
            })
        await self._send({"type": "response.create"})

    async def receive(self) -> AsyncIterator[LiveEvent]:
        while True:
            event = await self._receive_event()
            event_type = event.get("type")
            if event_type == "error":
                raise _error_from_event(event)
            if event_type == "response.output_audio.delta":
                yield LiveEvent(
                    audio=base64.b64decode(event.get("delta", ""), validate=True)
                )
            elif (
                self._transcribe_output
                and event_type == "response.output_audio_transcript.delta"
            ):
                yield LiveEvent(output_transcript=event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                if transcript := event.get("transcript"):
                    yield LiveEvent(input_transcript=transcript)
            elif event_type == "response.done":
                response = event.get("response", {})
                if response.get("status") == "failed":
                    raise OpenAIRealtimeError(
                        str(
                            response.get("status_details")
                            or "Realtime response failed"
                        )
                    )
                calls = []
                has_function_calls = False
                for item in response.get("output", []):
                    if item.get("type") != "function_call":
                        continue
                    has_function_calls = True
                    call_id = item.get("call_id", "")
                    if call_id in self._tool_calls_seen:
                        continue
                    self._tool_calls_seen.add(call_id)
                    calls.append(LiveToolCall(
                        name=item.get("name", ""),
                        call_id=call_id,
                        arguments=_decode_arguments(item.get("arguments")),
                    ))
                if calls:
                    yield LiveEvent(tool_calls=calls)
                elif not has_function_calls:
                    yield LiveEvent(turn_complete=True)

    async def async_close(self) -> None:
        if not self._ws.closed:
            await self._ws.close()

    async def _send(self, event: dict[str, Any]) -> None:
        await self._ws.send_json(event)

    async def _receive_event(self) -> dict[str, Any]:
        message = await self._ws.receive()
        if message.type == WSMsgType.TEXT:
            return json.loads(message.data)
        if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING}:
            raise OpenAIRealtimeError("OpenAI Realtime connection closed")
        if message.type == WSMsgType.ERROR:
            raise OpenAIRealtimeError(str(self._ws.exception()))
        raise OpenAIRealtimeError(f"Unexpected WebSocket message: {message.type}")


def _decode_arguments(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"json": str(value)}
    return decoded if isinstance(decoded, dict) else {"json": decoded}


def _error_from_event(event: dict[str, Any]) -> OpenAIRealtimeError:
    error = event.get("error", event)
    return OpenAIRealtimeError(
        str(error.get("message") or error.get("code") or "OpenAI Realtime error")
    )
