"""Google Gemini Live adapter for the provider-neutral live contract."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

from .const import (
    DEFAULT_THINKING_LEVEL,
    EXTENDED_THINKING_MODEL,
    supports_affective_dialog,
)
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

# Model generations whose default function-calling mode is asynchronous
# (NON_BLOCKING). The integration executes tools synchronously inside its
# receive loop, so its tool declarations must opt back into BLOCKING
# behaviour on these models.
_ASYNC_FUNCTION_CALLING_MODEL_PREFIXES = ("gemini-3.8",)


def _uses_async_function_calling(model: str | None) -> bool:
    """Return whether a model defaults to asynchronous function calling."""
    if not model:
        return False
    return model.startswith(_ASYNC_FUNCTION_CALLING_MODEL_PREFIXES)


async def async_create_gemini_client(
    hass: Any, api_key: str, affective_dialog: bool = False
) -> GeminiLiveClient:
    """Create the Google SDK client and wrap it in the neutral adapter."""
    from google.genai import types  # noqa: PLC0415
    from google import genai  # noqa: PLC0415

    http_options = None
    if affective_dialog:
        # Affective dialog is only accepted on the v1beta endpoint; newer SDK
        # default versions reject it with "Request contains an invalid
        # argument" (websocket close 1007).
        http_options = types.HttpOptions(api_version="v1beta")
    client = await hass.async_add_executor_job(
        lambda: genai.Client(api_key=api_key, http_options=http_options)
    )
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
        self._support_barge_in = config.support_barge_in
        self._extended_thinking = config.model == EXTENDED_THINKING_MODEL
        self._session: GeminiLiveSession | None = None

    async def __aenter__(self) -> GeminiLiveSession:
        self._session = GeminiLiveSession(
            await self._context.__aenter__(),
            support_barge_in=self._support_barge_in,
            extended_thinking=self._extended_thinking,
        )
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        await self._context.__aexit__(*exc)


class GeminiLiveSession:
    """Translate Gemini SDK calls and responses to the neutral contract."""

    def __init__(
        self,
        session: Any,
        support_barge_in: bool = False,
        extended_thinking: bool = False,
    ) -> None:
        self._session = session
        self._support_barge_in = support_barge_in
        self._extended_thinking = extended_thinking
        # Describes the most recent client message handed to the provider, so
        # a transport failure can be attributed to the message that likely
        # triggered it. The setup message counts as accepted once connect()
        # returns; the server rejects invalid setups before yielding a
        # session at all.
        self.last_outgoing = "setup (accepted)"

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
        self.last_outgoing = f"realtime audio chunk ({len(audio)} bytes)"

    async def end_audio(self) -> None:
        await self._session.send_realtime_input(audio_stream_end=True)
        self.last_outgoing = "realtime audio_stream_end"

    async def send_text(self, text: str) -> None:
        await self._session.send_realtime_input(text=text)
        self.last_outgoing = f"realtime text ({len(text)} chars)"

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
        self.last_outgoing = (
            f"tool_response for {[response.name for response in responses]}"
        )

    async def receive(self) -> AsyncIterator[LiveEvent]:
        while True:
            interrupted_turn = False
            receive_next_turn = False
            async for response in self._session.receive():
                interaction_status = _interaction_status(response)
                interaction_in_progress = interaction_status == "IN_PROGRESS"
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

                    interrupted = bool(getattr(content, "interrupted", False))
                    if interrupted:
                        interrupted_turn = True
                    if content.turn_complete and (
                        interrupted_turn or interaction_in_progress
                    ):
                        # The SDK ends each receive() iterator at turn_complete.
                        # Re-enter it so the replacement response can arrive.
                        receive_next_turn = (
                            self._support_barge_in or self._extended_thinking
                        )
                    terminal_turn = bool(content.turn_complete) and not (
                        self._extended_thinking and interaction_in_progress
                    )
                    if interrupted or terminal_turn:
                        # A single server message may carry both flags; one
                        # normalized event keeps interrupted processed before the
                        # consumer decides whether turn_complete is terminal.
                        yield LiveEvent(
                            interrupted=interrupted,
                            turn_complete=terminal_turn,
                        )

                if response.go_away or response.session_resumption_update:
                    yield LiveEvent(
                        go_away=response.go_away,
                        session_resumption_update=response.session_resumption_update,
                    )

            if not receive_next_turn:
                break


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
            # Gemini defaults to interrupting an in-progress response when new
            # activity is detected. Disable that behaviour explicitly unless
            # the user has opted into barge-in; some satellites keep sending
            # microphone audio while the model starts its response.
            "activity_handling": "NO_INTERRUPTION",
            "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY"
        },
    }
    if config.support_barge_in:
        # START_OF_ACTIVITY_INTERRUPTS is Gemini's barge-in mode: new user
        # speech interrupts the current generation. Set explicitly so the
        # behaviour is deterministic and tied to the configuration option.
        result["realtime_input_config"] = {
            "automatic_activity_detection": {
                "disabled": False,
            },
            "activity_handling": "START_OF_ACTIVITY_INTERRUPTS",
            "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY",
        }
    if config.transcribe_output:
        result["output_audio_transcription"] = {}
    if supports_affective_dialog(config.model) and config.affective_dialog:
        # Let the model read the tone and emotion in the user's voice and
        # adapt its own speaking style to match. The v1beta wire proto only
        # defines enable_affective_dialog inside generation_config; sending it
        # at the top level of the setup (which the SDK also accepts) makes the
        # server abort the session with "Request contains an invalid
        # argument" (websocket close 1007).
        result.setdefault("generation_config", {})[
            "enable_affective_dialog"
        ] = True
    if config.model == EXTENDED_THINKING_MODEL:
        result["thinking_config"] = {
            "thinking_level": config.thinking_level or DEFAULT_THINKING_LEVEL
        }
    if config.search_grounding:
        result.setdefault("tools", []).append({"google_search": {}})
    if config.tools:
        behavior = (
            "NON_BLOCKING"
            if config.model == EXTENDED_THINKING_MODEL
            else "BLOCKING" if _uses_async_function_calling(config.model) else None
        )
        result.setdefault("tools", []).extend(
            {"function_declarations": [_gemini_tool(tool, behavior)]}
            for tool in config.tools
        )
    return result


def _gemini_tool(
    tool: LiveTool, behavior: str | None = None
) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
    }
    if behavior:
        # Standard 3.8 sessions retain the integration's synchronous tool
        # flow. Extended Thinking requires non-blocking declarations.
        declaration["behavior"] = behavior
    if tool.parameters:
        declaration["parameters"] = _gemini_schema(tool.parameters)
    return declaration


def _interaction_status(response: Any) -> str | None:
    """Normalize an SDK interaction-status enum or string."""
    status = getattr(response, "interaction_status", None)
    if status is None and (content := getattr(response, "server_content", None)):
        status = getattr(content, "interaction_status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value).rsplit(".", 1)[-1].upper()


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
