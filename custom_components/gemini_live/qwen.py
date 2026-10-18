"""QwenCloud Omni Realtime WebSocket adapter for the shared live pipeline.

Protocol: https://docs.qwencloud.com/api-reference/real-time-multimodal/client-events
Audio is raw mono PCM16: 16 kHz input, 24 kHz output.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator
import json
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType

from .const import QWEN_REALTIME_URL, qwen_supports_tools
from .live import LiveConfig, LiveEvent, LiveToolCall, LiveToolResponse


class QwenRealtimeError(Exception):
    """A Qwen protocol, connection, or model error."""


class QwenRealtimeClient:
    """Create configured sessions using Home Assistant's HTTP session."""

    def __init__(self, session: ClientSession, api_key: str) -> None:
        self._session = session
        self._api_key = api_key

    def connect(self, config: LiveConfig) -> _QwenConnect:
        if not qwen_supports_tools(config.model):
            raise QwenRealtimeError(
                "Unsupported Qwen model: select a function-calling Omni Realtime model"
            )
        return _QwenConnect(self._session, self._api_key, config)


class _QwenConnect:
    def __init__(
        self, http_session: ClientSession, api_key: str, config: LiveConfig
    ) -> None:
        self._http_session = http_session
        self._api_key = api_key
        self._config = config
        self._session: QwenRealtimeSession | None = None

    async def __aenter__(self) -> QwenRealtimeSession:
        websocket = await self._http_session.ws_connect(
            QWEN_REALTIME_URL,
            params={"model": self._config.model},
            headers={"Authorization": f"Bearer {self._api_key}"},
            heartbeat=20,
        )
        self._session = QwenRealtimeSession(websocket)
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


class QwenRealtimeSession:
    """Translate Qwen events without exposing its protocol to the pipeline."""

    def __init__(self, websocket: ClientWebSocketResponse) -> None:
        self._ws = websocket
        self._support_barge_in = False
        self._transcribe_output = True

    @property
    def is_open(self) -> bool:
        return not self._ws.closed

    async def async_configure(self, config: LiveConfig) -> None:
        self._support_barge_in = config.support_barge_in
        self._transcribe_output = config.transcribe_output
        if not qwen_supports_tools(config.model):
            raise QwenRealtimeError(
                "Unsupported Qwen model: select a function-calling Omni Realtime model"
            )
        settings: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "instructions": config.system_instruction,
            "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
            "turn_detection": {
                "type": "semantic_vad",
                "threshold": 0.5,
                "silence_duration_ms": 800,
            }
            if config.support_barge_in
            else None,
        }
        if config.model.startswith("qwen3.8-"):
            settings["audio"] = {
                "input": {
                    "format": {
                        "type": "pcm",
                        "sample_rate": 16000,
                        "sample_format": "s16le",
                        "channels": 1,
                        "packing": "interleaved",
                        "channel_layout": "mono",
                    }
                },
                "output": {
                    "voice": config.voice,
                    "format": {"type": "pcm", "sample_rate": 24000},
                },
            }
        else:
            settings.update(
                voice=config.voice, input_audio_format="pcm", output_audio_format="pcm"
            )
        if config.tools:
            settings["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        **(
                            {"parameters": _parameters(tool.parameters)}
                            if tool.parameters
                            else {}
                        ),
                    },
                }
                for tool in config.tools
            ]
        await self._send({"type": "session.update", "session": settings})
        while True:
            event = await self._receive_event()
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise _error(event)

    async def send_audio(self, audio: bytes) -> None:
        # The shared STT stage already supplies 16 kHz PCM. Do not resample it.
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )

    async def end_audio(self) -> None:
        if not self._support_barge_in:
            await self._send({"type": "input_audio_buffer.commit"})
            await self._send({"type": "response.create"})

    async def send_text(self, text: str) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._send({"type": "response.create"})

    async def send_tool_responses(self, responses: list[LiveToolResponse]) -> None:
        for result in responses:
            if not result.call_id:
                raise QwenRealtimeError("Qwen tool response has no call ID")
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": json.dumps(result.response, separators=(",", ":")),
                    },
                }
            )
        if responses:
            await self._send({"type": "response.create"})

    async def receive(self) -> AsyncIterator[LiveEvent]:
        pending: dict[str, LiveToolCall] = {}
        interrupted = False
        transcript_streamed = False
        while True:
            event = await self._receive_event()
            kind = event.get("type")
            if (
                kind == "error"
                or kind == "conversation.item.input_audio_transcription.failed"
            ):
                raise _error(event)
            if kind == "response.created":
                interrupted = False
                transcript_streamed = False
            elif kind == "response.audio.delta" and not interrupted:
                try:
                    audio = base64.b64decode(event.get("delta", ""), validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise QwenRealtimeError("Invalid Qwen response audio") from exc
                if audio:
                    yield LiveEvent(audio=audio)
            elif kind == "response.text.delta" and not interrupted:
                yield LiveEvent(text=event.get("delta", ""))
            elif (
                kind == "response.audio_transcript.delta"
                and not interrupted
                and self._transcribe_output
            ):
                transcript_streamed = True
                yield LiveEvent(output_transcript=event.get("delta", ""))
            elif (
                kind == "response.audio_transcript.done"
                and not interrupted
                and self._transcribe_output
                and not transcript_streamed
            ):
                yield LiveEvent(output_transcript=event.get("transcript", ""))
            elif kind == "conversation.item.input_audio_transcription.completed":
                if transcript := event.get("transcript"):
                    yield LiveEvent(input_transcript=transcript)
            elif kind == "input_audio_buffer.speech_started" and self._support_barge_in:
                if not interrupted:
                    interrupted = True
                    yield LiveEvent(interrupted=True, user_activity_started=True)
            elif kind == "input_audio_buffer.speech_stopped" and self._support_barge_in:
                yield LiveEvent(user_activity_stopped=True)
            elif kind == "response.function_call_arguments.done":
                call = _tool_call(event)
                pending[call.call_id] = call
            elif (
                kind == "response.output_item.done"
                and event.get("item", {}).get("type") == "function_call"
            ):
                call = _tool_call(event["item"])
                pending[call.call_id] = call
            elif kind == "response.done":
                response = event.get("response", {})
                status = response.get("status")
                if status == "cancelled" and self._support_barge_in:
                    pending.clear()
                    if not interrupted:
                        interrupted = True
                        yield LiveEvent(interrupted=True)
                    continue
                if status not in (None, "completed"):
                    raise QwenRealtimeError(
                        str(response.get("status_details") or f"Qwen response {status}")
                    )
                if interrupted:
                    pending.clear()
                    continue
                for item in response.get("output", []):
                    if item.get("type") == "function_call":
                        call = _tool_call(item)
                        pending[call.call_id] = call
                if pending:
                    calls = list(pending.values())
                    pending.clear()
                    yield LiveEvent(tool_calls=calls)
                else:
                    yield LiveEvent(turn_complete=True)

    async def async_close(self) -> None:
        if not self._ws.closed:
            await self._ws.close()

    async def _send(self, event: dict[str, Any]) -> None:
        await self._ws.send_json({"event_id": f"event_{uuid4().hex}", **event})

    async def _receive_event(self) -> dict[str, Any]:
        message = await self._ws.receive()
        if message.type == WSMsgType.TEXT:
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError) as exc:
                raise QwenRealtimeError("Invalid Qwen event JSON") from exc
            if isinstance(event, dict):
                return event
            raise QwenRealtimeError("Invalid Qwen event object")
        if message.type == WSMsgType.ERROR:
            raise QwenRealtimeError(
                "Qwen WebSocket transport error"
            ) from self._ws.exception()
        raise QwenRealtimeError(
            f"Qwen Realtime connection closed or unexpected message ({message.type})"
        )


def _parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Represent HA's top-level schema unions as a function parameter object.

    As in the OpenAI adapter, HA remains responsible for validating the selected
    tool's complete schema when executing the call.
    """
    combinators = ("oneOf", "anyOf", "allOf")
    result = {
        key: value
        for key, value in schema.items()
        if key not in (*combinators, "enum", "const", "not")
    }
    properties = dict(result.get("properties", {}))
    required = set(result.get("required", ()))
    for keyword in combinators:
        for branch in schema.get(keyword, ()):
            if not isinstance(branch, dict):
                continue
            properties.update(branch.get("properties", {}))
            if keyword == "allOf":
                required.update(branch.get("required", ()))
    result.update(type="object", properties=properties)
    if required:
        result["required"] = sorted(required)
    else:
        result.pop("required", None)
    return result


def _tool_call(item: dict[str, Any]) -> LiveToolCall:
    try:
        arguments = json.loads(item.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise QwenRealtimeError("Invalid Qwen tool arguments") from exc
    if (
        not isinstance(arguments, dict)
        or not item.get("call_id")
        or not item.get("name")
    ):
        raise QwenRealtimeError("Invalid Qwen function call")
    return LiveToolCall(name=item["name"], call_id=item["call_id"], arguments=arguments)


def _error(event: dict[str, Any]) -> QwenRealtimeError:
    error = event.get("error", {})
    return QwenRealtimeError(
        str(error.get("message") or error.get("code") or "Qwen Realtime request failed")
    )
