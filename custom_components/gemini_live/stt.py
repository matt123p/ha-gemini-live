"""Speech-to-Text platform for Gemini Live."""

import asyncio
import codecs
from collections.abc import AsyncIterable, Callable
import datetime
import logging
import struct
import time
from uuid import uuid4
from typing import Any

from google import genai
from google.genai import types
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import chat_session, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_MODEL,
    CONF_SYSTEM_INSTRUCTION,
    CONF_TRANSCRIBE_GEMINI,
    CONF_VOICE,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_SYSTEM_INSTRUCTION,
    DOMAIN,
    GEMINI_LIVE_TTS_PLACEHOLDER,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
    SUPPORTED_LANGUAGES,
)
from .runtime import (
    AudioStream,
    PipelineTurn,
    TextStream,
    active_pipeline_conversation_id,
)
from .utils import resample_24k_to_16k, set_detailed_logging

_LOGGER = logging.getLogger(__name__)

# Target optimal chunk payload size (100ms of 16kHz 16-bit mono PCM = 3200 bytes)
OPTIMAL_STREAM_CHUNK_SIZE = 3200

# Schema keys supported by the Gemini Live function declaration format
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

_SEARCH_TOOL_HINTS = ("search", "web", "google")

_SEARCH_TOOL_INSTRUCTION = (
    "Use the available web-search tool whenever the user asks for current, latest, "
    "recent, live, or otherwise time-sensitive external information, or when the "
    "answer may have changed since your training data. Also use it when the user "
    "explicitly asks you to search, look up, check online, or verify something. "
    "Do not guess current external facts when the search tool can verify them."
)

RESPONSE_INACTIVITY_TIMEOUT = 30.0

END_CONVERSATION_TOOL_NAME = "end_conversation"

_END_CONVERSATION_INSTRUCTION = (
    f"Call {END_CONVERSATION_TOOL_NAME} when the user clearly indicates that they "
    "are finished, says goodbye, or asks to end the conversation. Do not call it "
    "merely because you have finished answering the current request. If the user's "
    "first request in a conversation is only 'stop', 'cancel', 'silence', 'turn it "
    "off', or a similar short command, treat it first as a request to stop an "
    "actively ringing alarm or timer. Before ending the conversation, use the "
    "available Home Assistant tools to check for and stop the ringing alarm or "
    "timer. Do not call end_conversation instead of attempting that action. After "
    "the ringing alarm or timer has been stopped, or if none is ringing, call "
    f"{END_CONVERSATION_TOOL_NAME} so Home Assistant stops listening."
)

_END_CONVERSATION_TOOL = {
    "function_declarations": [
        {
            "name": END_CONVERSATION_TOOL_NAME,
            "description": (
                "End the current voice conversation so Home Assistant stops "
                "listening for a follow-up turn. Call only when the user indicates "
                "that the conversation is finished."
            ),
        }
    ]
}

DISPLAY_MARKDOWN_TOOL_NAME = "display_markdown"

_DISPLAY_MARKDOWN_INSTRUCTION = (
    "The user WILL NOT see the transcription of what you say. "
    "Instead, if you want to display something to the user to read, for example instructions, "
    "lists, links, code blocks, or details that are better written down for the user than read out, "
    f"then you must call the {DISPLAY_MARKDOWN_TOOL_NAME} function. This is the only way the user "
    "will see any text from you."
)

_DISPLAY_MARKDOWN_TOOL = {
    "function_declarations": [
        {
            "name": DISPLAY_MARKDOWN_TOOL_NAME,
            "description": (
                "Display markdown text to the user. Call this when you want to show written details, "
                "instructions, or formatted text that the user should read."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "text": {
                        "type": "STRING",
                        "description": "The markdown formatted text to display to the user.",
                    }
                },
                "required": ["text"],
            },
        }
    ]
}



def _is_search_tool_name(name: str) -> bool:
    """Return whether a tool name indicates web-search capability."""
    lowered_name = name.lower()
    return any(hint in lowered_name for hint in _SEARCH_TOOL_HINTS)


def _is_connection_closed_ok(exc: Exception) -> bool:
    """Return true for websockets' normal-close exception without importing it."""
    return exc.__class__.__name__ == "ConnectionClosedOK"


# ---------------------------------------------------------------------------
# Schema / tool helpers
# ---------------------------------------------------------------------------

def _camel_to_snake(name: str) -> str:
    """Convert camelCase key to snake_case (matches official integration)."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _format_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a voluptuous-openapi schema dict to Gemini Live-compatible format."""
    if subschemas := schema.get("allOf"):
        for subschema in subschemas:
            if "type" in subschema:
                return _format_schema_for_gemini(subschema)
        return _format_schema_for_gemini(subschemas[0])

    result: dict[str, Any] = {}
    for key, val in schema.items():
        key = _camel_to_snake(key)
        if key not in _SUPPORTED_SCHEMA_KEYS:
            continue
        if key == "type":
            val = val.upper()
        elif key == "format":
            if schema.get("type") == "string" and val not in ("enum", "date-time"):
                continue
            if schema.get("type") == "number" and val not in ("float", "double"):
                continue
            if schema.get("type") == "integer" and val not in ("int32", "int64"):
                continue
            if schema.get("type") not in ("string", "number", "integer"):
                continue
        elif key == "items":
            val = _format_schema_for_gemini(val)
        elif key == "properties":
            val = {k: _format_schema_for_gemini(v) for k, v in val.items()}
        result[key] = val

    if result.get("enum") and result.get("type") != "STRING":
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]

    if result.get("type") == "OBJECT" and not result.get("properties"):
        result["properties"] = {"json": {"type": "STRING"}}
        result["required"] = []

    return result


def _format_tool_for_gemini_live(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None = None,
    encourage_web_search: bool = False,
) -> dict[str, Any]:
    """Convert an HA LLM Tool to a Gemini Live functionDeclaration dict."""
    try:
        from voluptuous_openapi import convert  # type: ignore[import]

        if tool.parameters.schema:
            raw_schema = convert(
                tool.parameters,
                custom_serializer=custom_serializer,
            )
            parameters: dict | None = _format_schema_for_gemini(raw_schema)
        else:
            parameters = None
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("Could not convert schema for tool %s: %s", tool.name, exc)
        parameters = None

    decl: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or f"Execute {tool.name}",
    }
    if encourage_web_search and _is_search_tool_name(tool.name):
        decl["description"] = (
            f"{decl['description']} Use this tool for current, latest, recent, "
            "time-sensitive, or explicitly requested online information."
        )
    if parameters:
        decl["parameters"] = parameters
    return decl


def _format_tools_for_gemini_live(
    tools: list[llm.Tool],
    custom_serializer: Callable[[Any], Any] | None = None,
    encourage_web_search: bool = False,
) -> list[dict[str, Any]]:
    """Convert HA LLM tools to Gemini Live tool declarations."""
    return [
        {
            "function_declarations": [
                _format_tool_for_gemini_live(
                    tool,
                    custom_serializer,
                    encourage_web_search,
                )
            ]
        }
        for tool in tools
    ]


def _add_end_conversation_tool(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add the integration-owned conversation completion callback."""
    return [*tools, _END_CONVERSATION_TOOL]


def _add_end_conversation_instruction(system_instruction: str) -> str:
    """Tell Gemini when to finish the Home Assistant conversation."""
    return f"{system_instruction}\n\n{_END_CONVERSATION_INSTRUCTION}"


def _add_display_markdown_tool(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add the integration-owned display markdown callback."""
    return [*tools, _DISPLAY_MARKDOWN_TOOL]


def _add_display_markdown_instruction(system_instruction: str) -> str:
    """Tell Gemini to use the display_markdown callback to show text to the user."""
    return f"{system_instruction}\n\n{_DISPLAY_MARKDOWN_INSTRUCTION}"



def _add_search_tool_instruction(
    system_instruction: str,
    tools: list[llm.Tool],
    encourage_web_search: bool,
) -> str:
    """Tell Gemini when to use an exposed search-like Assist tool."""
    if not encourage_web_search or not any(
        _is_search_tool_name(tool.name) for tool in tools
    ):
        return system_instruction
    return f"{system_instruction}\n\n{_SEARCH_TOOL_INSTRUCTION}"


def _escape_decode(value: Any) -> Any:
    """Recursively escape-decode values returned by the Gemini SDK."""
    if isinstance(value, str):
        return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {key: _escape_decode(item) for key, item in value.items()}
    return value


def _validate_tool_results(value: Any) -> Any:
    """Recursively convert non-json-serializable tool results."""
    if isinstance(value, (datetime.time, datetime.date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_validate_tool_results(item) for item in value]
    if isinstance(value, dict):
        return {key: _validate_tool_results(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# PCM diagnostics helper
# ---------------------------------------------------------------------------

def _analyse_pcm(pcm: bytes, sample_rate: int = 16000) -> str:
    """Return a one-line diagnostic string for a raw 16-bit signed mono PCM buffer."""
    num_samples = len(pcm) // 2
    if num_samples == 0:
        return "0 bytes — no audio at all"

    duration_ms = (num_samples * 1000) // sample_rate
    samples = struct.unpack(f"<{num_samples}h", pcm[: num_samples * 2])

    rms = (sum(s * s for s in samples) / num_samples) ** 0.5
    peak = max(abs(s) for s in samples)

    rms_pct = rms / 32767 * 100
    peak_pct = peak / 32767 * 100

    if rms_pct < 0.5:
        label = "SILENT"
    elif rms_pct < 3.0:
        label = "VERY_QUIET"
    elif rms_pct < 10.0:
        label = "QUIET"
    else:
        label = "SPEECH"

    return (
        f"{len(pcm):,} bytes | {duration_ms} ms | "
        f"RMS {rms:.0f} ({rms_pct:.1f}%) | "
        f"peak {peak} ({peak_pct:.1f}%) | {label}"
    )


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Gemini Live STT platform."""
    async_add_entities([GeminiLiveSTT(config_entry)])


# ---------------------------------------------------------------------------
# STT Entity
# ---------------------------------------------------------------------------

class GeminiLiveSTT(SpeechToTextEntity):
    """Gemini Live STT Entity."""

    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the STT entity."""
        self.entry = entry
        self._attr_name = "Gemini Live"
        self._attr_unique_id = f"{entry.entry_id}_stt"

    @property
    def name(self) -> str:
        return self._attr_name

    @property
    def unique_id(self) -> str:
        return self._attr_unique_id

    async def _async_run_audio_stream_sdk(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
        api_key: str,
        model: str,
        voice: str,
        custom_instruction: str,
        transcribe_gemini: bool,
        encourage_web_search: bool,
        result_future: asyncio.Future[SpeechResult],
    ) -> SpeechResult:
        """Process audio using the google-genai Live SDK."""
        turn_id = uuid4().hex[:8]
        display_markdown_text: str | None = None
        started_at = time.monotonic()
        conversation_id = active_pipeline_conversation_id(self.hass, self.entity_id)
        entry_data = self.hass.data[DOMAIN][self.entry.entry_id]
        session_manager = entry_data[GEMINI_SESSION_MANAGER_KEY]
        turn_store = entry_data[GEMINI_TURN_STORE_KEY]
        active_chat_session = chat_session.current_session.get()
        if (
            active_chat_session is None
            or active_chat_session.conversation_id != conversation_id
        ):
            active_chat_session = self.hass.data.get(
                chat_session.DATA_CHAT_SESSION,
                {},
            ).get(conversation_id)
        if active_chat_session is not None:
            session_manager.register_chat_session(self.hass, active_chat_session)

        _LOGGER.warning(
            "[turn=%s] SDK helper start api_key_present=%s model=%s voice=%s language=%s",
            turn_id,
            bool(api_key),
            model,
            voice,
            metadata.language or "en",
        )

        llm_api: llm.APIInstance | None = None
        ha_tools: list[llm.Tool] = []
        system_instruction = custom_instruction or DEFAULT_SYSTEM_INSTRUCTION

        try:
            llm_api = await llm.async_get_api(
                hass=self.hass,
                api_id=llm.LLM_API_ASSIST,
                llm_context=llm.LLMContext(
                    platform=DOMAIN,
                    context=Context(),
                    language=metadata.language or "en",
                    assistant="conversation",
                    device_id=None,
                ),
            )
            ha_tools = llm_api.tools

            api_prompt = llm_api.api_prompt
            if custom_instruction:
                system_instruction = f"{custom_instruction}\n\n{api_prompt}"
            else:
                system_instruction = DEFAULT_SYSTEM_INSTRUCTION + "\n\n" + api_prompt
            system_instruction = _add_search_tool_instruction(
                system_instruction,
                ha_tools,
                encourage_web_search,
            )
            _LOGGER.debug("Loaded HA Assist LLM API with %d tools", len(ha_tools))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "Could not load HA Assist LLM API: %s. Tools will be unavailable.",
                exc,
            )

        system_instruction = _add_end_conversation_instruction(system_instruction)
        if not transcribe_gemini:
            system_instruction = _add_display_markdown_instruction(system_instruction)

        gemini_tools = _add_end_conversation_tool(
            _format_tools_for_gemini_live(
                ha_tools,
                llm_api.custom_serializer,
                encourage_web_search,
            )
            if llm_api
            else []
        )
        if not transcribe_gemini:
            gemini_tools = _add_display_markdown_tool(gemini_tools)
        function_declarations = [
            declaration
            for tool in gemini_tools
            for declaration in tool["function_declarations"]
        ]
        _LOGGER.debug(
            "Exposing %d tools to Gemini Live: %s",
            len(function_declarations),
            [definition["name"] for definition in function_declarations],
        )

        _LOGGER.warning(
            "[turn=%s] creating genai client", turn_id
        )
        client = await self.hass.async_add_executor_job(
            lambda: genai.Client(api_key=api_key)
        )
        _LOGGER.warning(
            "[turn=%s] genai client created live_config_keys=%s tool_count=%d system_instruction_chars=%d",
            turn_id,
            ["response_modalities", "speech_config", "system_instruction", "input_audio_transcription", "output_audio_transcription", "realtime_input_config"] + (["tools"] if function_declarations else []),
            len(function_declarations),
            len(system_instruction),
        )
        live_config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": voice}
                }
            },
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "input_audio_transcription": {},
            "realtime_input_config": {
                "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY"
            },
        }
        if transcribe_gemini:
            live_config["output_audio_transcription"] = {}
        if gemini_tools:
            live_config["tools"] = gemini_tools

        _LOGGER.warning(
            "[turn=%s] live config prepared model=%s response_modalities=%s voice=%s has_tools=%s input_transcription=%s output_transcription=%s realtime_turn_coverage=%s",
            turn_id,
            model,
            live_config["response_modalities"],
            voice,
            bool(function_declarations),
            bool(live_config.get("input_audio_transcription") is not None),
            bool(live_config.get("output_audio_transcription") is not None),
            live_config["realtime_input_config"].get("turn_coverage"),
        )

        native_audio_model = "native-audio" in (model or "")
        _LOGGER.warning(
            "[turn=%s] setup model=%s native_audio_model=%s response_modalities=%s tools=%d",
            turn_id,
            model,
            native_audio_model,
            live_config["response_modalities"],
            len(function_declarations),
        )

        text_response_parts: list[str] = []
        input_transcript_parts: list[str] = []
        audio_response_chunk_count = 0
        audio_response_bytes = 0
        audio_sent = False
        last_response_activity = time.monotonic()
        gemini_replied = asyncio.Event()
        first_audio = asyncio.Event()
        input_transcript_received = asyncio.Event()
        response_audio_stream = AudioStream()
        response_text_stream = TextStream() if transcribe_gemini else None

        _LOGGER.warning(
            "[turn=%s] acquiring Gemini Live session conversation=%s",
            turn_id,
            conversation_id,
        )
        async with session_manager.acquire(
            conversation_id,
            client,
            model,
            live_config,
        ) as session:
            _LOGGER.warning(
                "[turn=%s] acquired Gemini Live session conversation=%s",
                turn_id,
                conversation_id,
            )

            async def send_audio() -> None:
                nonlocal audio_sent
                try:
                    first_chunk = True
                    audio_buffer = bytearray()
                    pcm_for_diag: list[bytes] = []
                    chunk_count = 0

                    _LOGGER.warning("[turn=%s] send_audio task spawned", turn_id)

                    async for chunk in stream:
                        if not chunk:
                            continue
                        if gemini_replied.is_set():
                            _LOGGER.warning(
                                "[turn=%s] send_audio stopped because Gemini started replying",
                                turn_id,
                            )
                            break

                        if first_chunk:
                            first_chunk = False
                            if chunk[:4] == b"RIFF":
                                data_offset = chunk.find(b"data")
                                if data_offset != -1:
                                    chunk = chunk[data_offset + 8 :]

                        audio_buffer.extend(chunk)

                        while len(audio_buffer) >= OPTIMAL_STREAM_CHUNK_SIZE:
                            dispatch_chunk = bytes(audio_buffer[:OPTIMAL_STREAM_CHUNK_SIZE])
                            del audio_buffer[:OPTIMAL_STREAM_CHUNK_SIZE]

                            chunk_count += 1
                            pcm_for_diag.append(dispatch_chunk)
                            _LOGGER.debug(
                                "[turn=%s] Shipping optimized media chunk %d (%d bytes)",
                                turn_id,
                                chunk_count,
                                len(dispatch_chunk),
                            )
                            _LOGGER.debug(
                                "[turn=%s] calling session.send_realtime_input audio_pcm_16k chunk_size=%d",
                                turn_id,
                                len(dispatch_chunk),
                            )
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=dispatch_chunk,
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                            audio_sent = True

                    if len(audio_buffer) > 0 and not gemini_replied.is_set():
                        chunk_count += 1
                        dispatch_chunk = bytes(audio_buffer)
                        pcm_for_diag.append(dispatch_chunk)
                        _LOGGER.debug(
                            "[turn=%s] flushing trailing audio chunk size=%d",
                            turn_id,
                            len(dispatch_chunk),
                        )
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=dispatch_chunk,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                        audio_sent = True

                    if pcm_for_diag:
                        _LOGGER.warning(
                            "[turn=%s] Finished voice streaming. Total blocks dispatched=%d. Metrics=%s",
                            turn_id,
                            chunk_count,
                            _analyse_pcm(b"".join(pcm_for_diag)),
                        )

                    if audio_sent and not gemini_replied.is_set():
                        _LOGGER.debug("[turn=%s] signalling audio stream end", turn_id)
                        await session.send_realtime_input(audio_stream_end=True)
                except asyncio.CancelledError:
                    _LOGGER.warning(
                        "[turn=%s] audio sender cancelled — Gemini started replying",
                        turn_id,
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.exception("[turn=%s] Failure inside send_audio: %s", turn_id, exc)

            async def receive_responses() -> None:
                nonlocal audio_response_bytes, audio_response_chunk_count
                nonlocal last_response_activity, display_markdown_text
                try:
                    _LOGGER.warning("[turn=%s] receive_responses started", turn_id)
                    async for response in session.receive():
                        _LOGGER.warning(
                            "[turn=%s] received response tool_call=%s server_content=%s go_away=%s session_resumption_update=%s",
                            turn_id,
                            bool(response.tool_call),
                            bool(response.server_content),
                            bool(response.go_away),
                            bool(response.session_resumption_update),
                        )
                        if response.go_away:
                            _LOGGER.warning(
                                "[turn=%s] Gemini go_away=%s",
                                turn_id,
                                response.go_away,
                            )
                        if response.session_resumption_update:
                            _LOGGER.warning(
                                "[turn=%s] Gemini session_resumption_update=%s",
                                turn_id,
                                response.session_resumption_update,
                            )

                        if response.tool_call:
                            last_response_activity = time.monotonic()
                            function_calls = response.tool_call.function_calls or []
                            function_responses = []

                            _LOGGER.warning(
                                "[turn=%s] tool_call count=%d",
                                turn_id,
                                len(function_calls),
                            )

                            for call in function_calls:
                                tool_name = call.name or ""
                                tool_args = _escape_decode(call.args or {})
                                call_id = call.id
                                _LOGGER.info(
                                    "Gemini Live tool call: %s(%s)",
                                    tool_name,
                                    tool_args,
                                )

                                if tool_name == END_CONVERSATION_TOOL_NAME:
                                    session_manager.complete_conversation(
                                        conversation_id
                                    )
                                    tool_result = {
                                        "success": True,
                                        "conversation_ended": True,
                                    }
                                elif tool_name == DISPLAY_MARKDOWN_TOOL_NAME:
                                    display_markdown_text = tool_args.get("text")
                                    tool_result = {
                                        "success": True,
                                        "displayed": True,
                                    }
                                elif llm_api is not None:
                                    try:
                                        tool_input = llm.ToolInput(
                                            tool_name=tool_name,
                                            tool_args=tool_args,
                                        )
                                        tool_result = await llm_api.async_call_tool(
                                            tool_input
                                        )
                                    except Exception as err:  # noqa: BLE001
                                        _LOGGER.error("Tool %s failed: %s", tool_name, err)
                                        tool_result = {"error": str(err)}
                                else:
                                    tool_result = {"error": "HA LLM API not available"}
                                tool_result = _validate_tool_results(tool_result)

                                _LOGGER.warning(
                                    "[turn=%s] tool response prepared name=%s id=%s result_type=%s",
                                    turn_id,
                                    tool_name,
                                    call_id,
                                    type(tool_result).__name__,
                                )

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=tool_name,
                                        id=call_id,
                                        response=tool_result,
                                    )
                                )

                            if function_responses:
                                _LOGGER.warning(
                                    "[turn=%s] sending %d tool response(s) to Gemini",
                                    turn_id,
                                    len(function_responses),
                                )
                                await session.send_tool_response(
                                    function_responses=function_responses
                                )
                                _LOGGER.warning(
                                    "[turn=%s] sent %d tool response(s) back to Gemini",
                                    turn_id,
                                    len(function_responses),
                                )

                        content = response.server_content
                        if not content:
                            _LOGGER.debug("[turn=%s] response had no server_content", turn_id)
                            continue
                        last_response_activity = time.monotonic()

                        if content.model_turn:
                            parts = content.model_turn.parts or []
                            _LOGGER.warning(
                                "[turn=%s] modelTurn parts=%d turnComplete=%s",
                                turn_id,
                                len(parts),
                                content.turn_complete,
                            )
                            for part in parts:
                                if part.text:
                                    _LOGGER.debug(
                                        "[turn=%s] model text chunk len=%d",
                                        turn_id,
                                        len(part.text),
                                    )
                                    text_response_parts.append(part.text)
                                    if response_text_stream is not None:
                                        response_text_stream.add_chunk(part.text)
                                if part.inline_data and part.inline_data.data:
                                    if not gemini_replied.is_set():
                                        gemini_replied.set()
                                        _LOGGER.warning(
                                            "[turn=%s] gemini_replied set on first inline audio chunk",
                                            turn_id,
                                        )
                                    raw_chunk = part.inline_data.data
                                    audio_response_chunk_count += 1
                                    audio_response_bytes += len(raw_chunk)
                                    resampled_chunk = resample_24k_to_16k(raw_chunk)
                                    response_audio_stream.add_chunk(resampled_chunk)
                                    first_audio.set()
                                    _LOGGER.debug(
                                        "[turn=%s] inline audio chunk bytes=%d total_audio_chunks=%d",
                                        turn_id,
                                        len(raw_chunk),
                                        audio_response_chunk_count,
                                    )

                        if content.output_transcription and content.output_transcription.text:
                            transcription = content.output_transcription.text
                            _LOGGER.debug(
                                "[turn=%s] output transcription chunk len=%d",
                                turn_id,
                                len(transcription),
                            )
                            text_response_parts.append(transcription)
                            if response_text_stream is not None:
                                response_text_stream.add_chunk(transcription)
                            _LOGGER.warning(
                                "[turn=%s] outputTranscription text len=%d text=%r",
                                turn_id,
                                len(transcription),
                                transcription[:200],
                            )

                        if content.input_transcription and content.input_transcription.text:
                            transcription = content.input_transcription.text
                            _LOGGER.debug(
                                "[turn=%s] input transcription chunk len=%d",
                                turn_id,
                                len(transcription),
                            )
                            input_transcript_parts.append(transcription)
                            input_transcript_received.set()
                            _LOGGER.warning(
                                "[turn=%s] inputTranscription text len=%d text=%r",
                                turn_id,
                                len(transcription),
                                transcription[:200],
                            )

                        if content.turn_complete:
                            if native_audio_model and not gemini_replied.is_set():
                                _LOGGER.warning(
                                    "[turn=%s] turnComplete before audio; keeping session open and waiting",
                                    turn_id,
                                )
                                continue
                            _LOGGER.warning(
                                "[turn=%s] turnComplete received; breaking receive loop (audio_chunks=%d text_parts=%d)",
                                turn_id,
                                audio_response_chunk_count,
                                len(text_response_parts),
                            )
                            break
                except asyncio.CancelledError:
                    _LOGGER.warning("[turn=%s] receive_responses cancelled", turn_id)
                    raise
                except Exception as exc:  # noqa: BLE001
                    if _is_connection_closed_ok(exc):
                        _LOGGER.warning(
                            "[turn=%s] Gemini Live websocket closed normally",
                            turn_id,
                        )
                    else:
                        _LOGGER.exception(
                            "[turn=%s] error in receive_responses: %s",
                            turn_id,
                            exc,
                        )

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_responses())
            _LOGGER.warning("[turn=%s] created send and receive tasks", turn_id)

            async def publish_streaming_turn() -> None:
                """Release the pipeline once Gemini starts producing audio."""
                await first_audio.wait()
                if not input_transcript_parts:
                    try:
                        await asyncio.wait_for(
                            input_transcript_received.wait(),
                            timeout=0.5,
                        )
                    except TimeoutError:
                        pass

                user_text = (
                    "".join(input_transcript_parts).strip()
                    or GEMINI_LIVE_TTS_PLACEHOLDER
                )
                # HA persistently caches TTS audio by message. A per-turn message
                # prevents it from replaying an earlier Gemini audio stream.
                if not transcribe_gemini and display_markdown_text is not None:
                    tts_message = display_markdown_text
                else:
                    tts_message = f"{GEMINI_LIVE_TTS_PLACEHOLDER} {turn_id}"
                turn_store.add_voice_turn(
                    PipelineTurn(
                        conversation_id=conversation_id,
                        user_text=user_text,
                        assistant_text=tts_message,
                        audio=response_audio_stream,
                        assistant_text_stream=response_text_stream,
                    )
                )
                if not result_future.done():
                    result_future.set_result(
                        SpeechResult(user_text, SpeechResultState.SUCCESS)
                    )
                _LOGGER.warning(
                    "[turn=%s] released streaming TTS after first audio; user_transcript=%r",
                    turn_id,
                    user_text[:80],
                )

            publish_task = asyncio.create_task(publish_streaming_turn())

            async def _cancel_sender_on_reply() -> None:
                await gemini_replied.wait()
                if not send_task.done():
                    _LOGGER.warning(
                        "[turn=%s] cancelling send task because Gemini started replying",
                        turn_id,
                    )
                    send_task.cancel()

            cancel_on_reply_task = asyncio.create_task(_cancel_sender_on_reply())
            try:
                done, _pending = await asyncio.wait(
                    [send_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for completed_task in done:
                    try:
                        completed_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.error(
                            "[turn=%s] task completed with exception: %s",
                            turn_id,
                            exc,
                        )

                if receive_task in done:
                    if not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                else:
                    if not audio_sent:
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass
                        return SpeechResult(None, SpeechResultState.ERROR)

                    while not receive_task.done():
                        remaining = RESPONSE_INACTIVITY_TIMEOUT - (
                            time.monotonic() - last_response_activity
                        )
                        if remaining <= 0:
                            _LOGGER.warning(
                                "[turn=%s] cancelling Gemini receive task after %.1fs without response activity",
                                turn_id,
                                RESPONSE_INACTIVITY_TIMEOUT,
                            )
                            receive_task.cancel()
                            try:
                                await receive_task
                            except asyncio.CancelledError:
                                pass
                            break
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(receive_task),
                                timeout=remaining,
                            )
                        except asyncio.TimeoutError:
                            continue
                if first_audio.is_set():
                    await publish_task
                else:
                    publish_task.cancel()
            finally:
                if not cancel_on_reply_task.done():
                    cancel_on_reply_task.cancel()
                tasks = [send_task, receive_task, cancel_on_reply_task]
                tasks.append(publish_task)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )
                response_audio_stream.finish()
                if response_text_stream is not None:
                    response_text_stream.finish()
                _LOGGER.warning(
                    "[turn=%s] session tasks complete send_done=%s receive_done=%s audio_sent=%s replied=%s",
                    turn_id,
                    send_task.done(),
                    receive_task.done(),
                    audio_sent,
                    gemini_replied.is_set(),
                )

        response_text = "".join(text_response_parts)
        input_transcript = "".join(input_transcript_parts).strip()
        all_audio_24k_len = audio_response_bytes

        if first_audio.is_set():
            _LOGGER.warning(
                "STT: Gemini audio ready: text=%d chars, raw_audio=%d bytes",
                len(response_text),
                all_audio_24k_len,
            )
        else:
            _LOGGER.warning("STT: No audio response received from Gemini Live")

        final_text = input_transcript or response_text
        if first_audio.is_set():
            return SpeechResult(
                input_transcript or GEMINI_LIVE_TTS_PLACEHOLDER,
                SpeechResultState.SUCCESS,
            )
        if not final_text:
            _LOGGER.error(
                "STT: Gemini returned no usable transcript or response text"
            )
            return SpeechResult(None, SpeechResultState.ERROR)

        if not transcribe_gemini and display_markdown_text is not None:
            assistant_text = display_markdown_text
        else:
            assistant_text = response_text

        if assistant_text:
            turn_store.add_voice_turn(
                PipelineTurn(
                    conversation_id=conversation_id,
                    user_text=final_text,
                    assistant_text=assistant_text,
                    audio=b"",
                )
            )

        _LOGGER.warning(
            "[turn=%s] STT returning SpeechResult transcript=%r response_chars=%d elapsed=%.3fs",
            turn_id,
            final_text[:80],
            len(response_text),
            time.monotonic() - started_at,
        )
        return SpeechResult(final_text, SpeechResultState.SUCCESS)

    async def _async_process_audio_stream_sdk(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
        api_key: str,
        model: str,
        voice: str,
        custom_instruction: str,
        transcribe_gemini: bool,
        encourage_web_search: bool,
    ) -> SpeechResult:
        """Run the Live turn in the background so TTS can consume it immediately."""
        result_future: asyncio.Future[SpeechResult] = asyncio.Future()
        task = self.hass.async_create_background_task(
            self._async_run_audio_stream_sdk(
                metadata,
                stream,
                api_key,
                model,
                voice,
                custom_instruction,
                transcribe_gemini,
                encourage_web_search,
                result_future,
            ),
            "Gemini Live audio turn",
        )

        def set_final_result(completed_task: asyncio.Task[SpeechResult]) -> None:
            if result_future.done():
                return
            try:
                result_future.set_result(completed_task.result())
            except asyncio.CancelledError:
                result_future.set_result(SpeechResult(None, SpeechResultState.ERROR))
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Gemini Live audio turn failed")
                result_future.set_result(SpeechResult(None, SpeechResultState.ERROR))

        task.add_done_callback(set_final_result)
        try:
            return await result_future
        except asyncio.CancelledError:
            task.cancel()
            raise

    @property
    def supported_languages(self) -> list[str]:
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    async def async_process_audio_stream(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> SpeechResult:
        """Process the audio stream and send it directly to Gemini Live API."""
        turn_id = uuid4().hex[:8]
        started_at = time.monotonic()
        config = {**self.entry.data, **self.entry.options}
        api_key = config.get(CONF_API_KEY)
        model = config.get(CONF_MODEL)
        voice = config.get(CONF_VOICE)
        custom_instruction = config.get(CONF_SYSTEM_INSTRUCTION, "")
        transcribe_gemini = bool(
            config.get(CONF_TRANSCRIBE_GEMINI, DEFAULT_TRANSCRIBE_GEMINI)
        )
        encourage_web_search = bool(
            config.get(CONF_ENCOURAGE_WEB_SEARCH, DEFAULT_ENCOURAGE_WEB_SEARCH)
        )
        set_detailed_logging(bool(config.get(CONF_DETAILED_LOGGING, False)))

        _LOGGER.warning(
            "[turn=%s] STT start language=%s model=%s voice=%s detailed_logging=%s",
            turn_id,
            metadata.language or "en",
            model,
            voice,
            bool(config.get(CONF_DETAILED_LOGGING, False)),
        )

        if not api_key:
            _LOGGER.error("API Key not configured for Gemini Live")
            return SpeechResult(None, SpeechResultState.ERROR)

        return await self._async_process_audio_stream_sdk(
            metadata,
            stream,
            api_key,
            model,
            voice,
            custom_instruction,
            transcribe_gemini,
            encourage_web_search,
        )
