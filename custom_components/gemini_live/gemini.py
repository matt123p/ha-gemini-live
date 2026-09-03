"""Google Gemini Live adapter for the provider-neutral live contract."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

from .live import LiveConfig, LiveEvent, LiveTool, LiveToolCall, LiveToolResponse

_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "max_items",
    "min_items",
    "properties",
    "required",
    "items",
}


async def async_create_gemini_client(hass: Any, api_key: str) -> GeminiLiveClient:
    """Create the Google SDK client and wrap it in the neutral adapter."""
    from google import genai  # noqa: PLC0415

    client = await hass.async_add_executor_job(lambda: genai.Client(api_key=api_key))
    return GeminiLiveClient(client)


class GeminiLiveClient:
    """Create normalized sessions backed by the Google Gen AI SDK."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def connect(self, config: LiveConfig) -> _GeminiConnect:
        return _GeminiConnect(self._client, config)


class _GeminiConnect:
    def __init__(self, client: Any, config: LiveConfig) -> None:
        self._context: AbstractAsyncContextManager[Any] = client.aio.live.connect(
            model=config.model,
            config=_gemini_config(config),
        )
        self._session: GeminiLiveSession | None = None

    async def __aenter__(self) -> GeminiLiveSession:
        self._session = GeminiLiveSession(await self._context.__aenter__())
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        await self._context.__aexit__(*exc)


class GeminiLiveSession:
    """Translate Gemini SDK calls and responses to the neutral contract."""

    def __init__(self, session: Any) -> None:
        self._session = session

    @property
    def is_open(self) -> bool:
        websocket = getattr(self._session, "_ws", None)
        if getattr(websocket, "closed", False):
            return False
        state_name = getattr(getattr(websocket, "state", None), "name", None)
        return state_name is None or state_name == "OPEN"

    async def send_audio(self, audio: bytes) -> None:
        from google.genai import types  # noqa: PLC0415

        await self._session.send_realtime_input(
            audio=types.Blob(data=audio, mime_type="audio/pcm;rate=16000")
        )

    async def end_audio(self) -> None:
        await self._session.send_realtime_input(audio_stream_end=True)

    async def send_text(self, text: str) -> None:
        await self._session.send_realtime_input(text=text)

    async def send_tool_responses(
        self, responses: list[LiveToolResponse]
    ) -> None:
        from google.genai import types  # noqa: PLC0415

        await self._session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    name=response.name,
                    id=response.call_id,
                    response=response.response,
                )
                for response in responses
            ]
        )

    async def receive(self) -> AsyncIterator[LiveEvent]:
        async for response in self._session.receive():
            if response.tool_call:
                yield LiveEvent(
                    tool_calls=[
                        LiveToolCall(
                            name=call.name or "",
                            call_id=call.id,
                            arguments=_escape_decode(call.args or {}),
                        )
                        for call in response.tool_call.function_calls or []
                    ]
                )

            content = response.server_content
            if content:
                if content.model_turn:
                    for part in content.model_turn.parts or []:
                        if part.text:
                            yield LiveEvent(text=part.text)
                        if part.inline_data and part.inline_data.data:
                            yield LiveEvent(audio=part.inline_data.data)
                if content.output_transcription and content.output_transcription.text:
                    yield LiveEvent(
                        output_transcript=content.output_transcription.text
                    )
                if content.input_transcription and content.input_transcription.text:
                    yield LiveEvent(input_transcript=content.input_transcription.text)
                if content.turn_complete:
                    yield LiveEvent(turn_complete=True)

            if response.go_away or response.session_resumption_update:
                yield LiveEvent(
                    go_away=response.go_away,
                    session_resumption_update=response.session_resumption_update,
                )


def _gemini_config(config: LiveConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": config.voice}
            }
        },
        "system_instruction": {"parts": [{"text": config.system_instruction}]},
        "input_audio_transcription": {},
        "realtime_input_config": {
            "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY"
        },
    }
    if config.transcribe_output:
        result["output_audio_transcription"] = {}
    if config.tools:
        result["tools"] = [
            {"function_declarations": [_gemini_tool(tool)]}
            for tool in config.tools
        ]
    return result


def _gemini_tool(tool: LiveTool) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
    }
    if tool.parameters:
        declaration["parameters"] = _gemini_schema(tool.parameters)
    return declaration


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if subschemas := schema.get("allOf"):
        for subschema in subschemas:
            if "type" in subschema:
                return _gemini_schema(subschema)
        return _gemini_schema(subschemas[0])

    result: dict[str, Any] = {}
    for original_key, original_value in schema.items():
        key = _camel_to_snake(original_key)
        if key not in _SUPPORTED_SCHEMA_KEYS:
            continue
        value = original_value
        if key == "type":
            value = value.upper()
        elif key == "format":
            schema_type = schema.get("type")
            supported = {
                "string": ("enum", "date-time"),
                "number": ("float", "double"),
                "integer": ("int32", "int64"),
            }
            if value not in supported.get(schema_type, ()):
                continue
        elif key == "items":
            value = _gemini_schema(value)
        elif key == "properties":
            value = {name: _gemini_schema(item) for name, item in value.items()}
        result[key] = value

    if result.get("enum") and result.get("type") != "STRING":
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]
    if result.get("type") == "OBJECT" and not result.get("properties"):
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []
    return result


def _camel_to_snake(name: str) -> str:
    return "".join(
        "_" + char.lower() if char.isupper() else char for char in name
    ).lstrip("_")


def _escape_decode(value: Any) -> Any:
    """Decode escaped Gemini string arguments recursively."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {key: _escape_decode(item) for key, item in value.items()}
    return value
