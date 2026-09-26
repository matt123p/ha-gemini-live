"""Shared speech-to-text platform for realtime model providers."""

import asyncio
from collections.abc import AsyncIterable, Callable
import datetime
import logging
import struct
import time
from uuid import uuid4
from typing import Any

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    DEFAULT_AUDIO_PROCESSING,
    SpeechAudioProcessing,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import chat_session, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.issue_registry import (
    IssueSeverity,
    async_create_issue,
    async_delete_issue,
)

from .compat import supports_tts_interruption
from .gemini import GeminiLiveClient, async_create_gemini_client
from .live import LiveConfig, LiveTool, LiveToolResponse
from .const import (
    CONF_API_KEY,
    CONF_AFFECTIVE_DIALOG,
    CONF_DETAILED_LOGGING,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_SEARCH_GROUNDING,
    CONF_SYSTEM_INSTRUCTION,
    CONF_SHOW_TEXT,
    CONF_SUPPORT_BARGE_IN,
    CONF_THINKING_LEVEL,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_SUPPORT_BARGE_IN,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_TRANSCRIBE_GPT,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_SYSTEM_INSTRUCTION,
    DEFAULT_SHOW_TEXT,
    DEFAULT_SEARCH_GROUNDING,
    DOMAIN,
    GEMINI_LIVE_TTS_PLACEHOLDER,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
    OPENAI_SYSTEM_INSTRUCTION,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    SUPPORTED_LANGUAGES,
)
from .openai import OpenAIRealtimeClient
from .runtime import (
    AudioStream,
    PipelineTurn,
    TextStream,
    active_pipeline_context,
)
from .utils import (
    PCM24kTo16kStreamResampler,
    set_detailed_logging,
)

_LOGGER = logging.getLogger(__name__)

# Target optimal chunk payload size (100ms of 16kHz 16-bit mono PCM = 3200 bytes)
OPTIMAL_STREAM_CHUNK_SIZE = 3200

# Smaller chunks while barge-in is enabled (40ms of 16kHz 16-bit mono PCM)
# so user speech is forwarded to the provider with less interruption latency.
BARGE_IN_STREAM_CHUNK_SIZE = 1280

_SEARCH_TOOL_HINTS = ("search", "web", "google")

_SEARCH_TOOL_INSTRUCTION = (
    "You MUST use the available web-search tool whenever the user asks for current, latest, "
    "recent, live, or otherwise time-sensitive external information, or when the "
    "answer may have changed since your training data. Also use it when the user "
    "explicitly asks you to search, look up, check online, or verify something. "
    "Do not guess current external facts when the search tool can verify them."
)

RESPONSE_INACTIVITY_TIMEOUT = 30.0

_SPENDING_CAP_ERROR_MARKER = "exceeded its monthly spending cap"
_SPENDING_CAP_ISSUE_PREFIX = "spending_cap_exceeded"
_SPENDING_CAP_URL = "https://ai.studio/spend"
_SPENDING_CAP_USER_MESSAGE = (
    "Gemini Live is unavailable because your Google AI project has exceeded "
    f"its monthly spending cap. Please go to {_SPENDING_CAP_URL} to manage "
    "your project's spending limit."
)

_PREPAYMENT_CREDITS_ERROR_MARKER = "prepayment credits are depleted"
_PREPAYMENT_CREDITS_ISSUE_PREFIX = "prepayment_credits_depleted"
_PREPAYMENT_CREDITS_URL = "https://ai.studio/projects"
_PREPAYMENT_CREDITS_USER_MESSAGE = (
    "Gemini Live is unavailable because your Google AI prepayment credits "
    f"are depleted. Please go to {_PREPAYMENT_CREDITS_URL} to add credits "
    "or manage your project's billing."
)

_OPENAI_NO_CREDITS_ERROR_MARKER = "no credits remaining"
_OPENAI_NO_CREDITS_ISSUE_PREFIX = "openai_no_credits"
_OPENAI_NO_CREDITS_URL = "https://platform.openai.com/settings/organization/billing/"
_OPENAI_NO_CREDITS_USER_MESSAGE = (
    "GPT Realtime is unavailable because your OpenAI account has no credits "
    f"remaining. Please go to {_OPENAI_NO_CREDITS_URL} to add credits."
)

END_CONVERSATION_TOOL_NAME = "end_conversation"

_END_CONVERSATION_INSTRUCTION = (
    f"You MUST call {END_CONVERSATION_TOOL_NAME} as the FIRST action when the "
    "user clearly indicates that they are finished, says goodbye, or asks to end "
    "the conversation. Call it before saying a farewell or any other response. "
    "This is required so Home Assistant stops listening; saying goodbye alone does "
    "not end the conversation. Examples include 'I am finished', 'we are done', "
    "'end the conversation', 'stop listening', 'goodbye', and 'bye'. Do not call it "
    "merely because you have finished answering the current request."
)

_END_CONVERSATION_TOOL = LiveTool(
    name=END_CONVERSATION_TOOL_NAME,
    description=(
        "REQUIRED: immediately end the current voice conversation when the user "
        "asks to finish, stop listening, says goodbye, or otherwise indicates they "
        "are done. Call this before speaking a farewell so Home Assistant does not "
        "listen for a follow-up turn."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)

SHOW_TEXT_TOOL_NAME = "show_text"

_SHOW_TEXT_INSTRUCTION = (
    "The user WILL NOT see the transcription of what you say. "
    f"You MUST call {SHOW_TEXT_TOOL_NAME} whenever your response contains a lot of "
    "information or would be easier to scan, follow, copy, or refer back to in writing. "
    "This includes multi-step instructions, detailed or long lists, comparisons, schedules, "
    "names, dates, links, code, and other precise details. Err on the side of showing text "
    "for detailed or information-dense answers. When using it, call it as your FIRST action, "
    "before speaking any part of the answer. Put the complete useful written content in "
    f"the {SHOW_TEXT_TOOL_NAME} call, then give a concise spoken summary. "
    "Do not call it for a simple, brief answer. This function is the only way the user "
    "will see any text from you."
)

_SHOW_TEXT_TOOL = LiveTool(
    name=SHOW_TEXT_TOOL_NAME,
    description=(
        "Display text or markdown to the user before speaking. Use this proactively as your "
        "first action whenever a response "
        "contains substantial information or details that are easier to scan, follow, "
        "copy, or revisit in writing, including instructions, lists, comparisons, "
        "schedules, names, dates, links, and code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text or markdown formatted text to display to the user.",
            }
        },
        "required": ["text"],
    },
)


def _is_search_tool_name(name: str) -> bool:
    """Return whether a tool name indicates web-search capability."""
    lowered_name = name.lower()
    return any(hint in lowered_name for hint in _SEARCH_TOOL_HINTS)


def _is_connection_closed_ok(exc: Exception) -> bool:
    """Return true for websockets' normal-close exception without importing it."""
    return exc.__class__.__name__ == "ConnectionClosedOK"


def _user_visible_api_error(exc: BaseException) -> str | None:
    """Return safe user-facing text for API failures the user can resolve."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_text = str(current).lower()
        if _SPENDING_CAP_ERROR_MARKER in error_text:
            return _SPENDING_CAP_USER_MESSAGE
        if _PREPAYMENT_CREDITS_ERROR_MARKER in error_text:
            return _PREPAYMENT_CREDITS_USER_MESSAGE
        if _OPENAI_NO_CREDITS_ERROR_MARKER in error_text:
            return _OPENAI_NO_CREDITS_USER_MESSAGE
        current = current.__cause__ or current.__context__
    return None


def _spending_cap_issue_id(entry_id: str) -> str:
    """Return the Repairs issue ID for one config entry."""
    return f"{_SPENDING_CAP_ISSUE_PREFIX}_{entry_id}"


def _prepayment_credits_issue_id(entry_id: str) -> str:
    """Return the prepaid credits Repairs issue ID for one config entry."""
    return f"{_PREPAYMENT_CREDITS_ISSUE_PREFIX}_{entry_id}"


def _openai_no_credits_issue_id(entry_id: str) -> str:
    """Return the OpenAI credits Repairs issue ID for one config entry."""
    return f"{_OPENAI_NO_CREDITS_ISSUE_PREFIX}_{entry_id}"


# ---------------------------------------------------------------------------
# Schema / tool helpers
# ---------------------------------------------------------------------------


def _format_tool_for_live(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None = None,
    encourage_web_search: bool = False,
) -> LiveTool:
    """Convert a Home Assistant tool to the provider-neutral format."""
    try:
        if tool.parameters.schema:
            try:
                # Home Assistant 2026.8 and newer use probatio for OpenAPI
                # schema generation.
                from probatio import to_openapi  # type: ignore[import]

                raw_schema = to_openapi(
                    tool.parameters,
                    custom_serializer=custom_serializer,
                )
            except ImportError:
                # Keep compatibility with older Home Assistant releases.
                from voluptuous_openapi import convert  # type: ignore[import]

                raw_schema = convert(
                    tool.parameters,
                    custom_serializer=custom_serializer,
                )
            parameters: dict[str, Any] | None = raw_schema
        else:
            parameters = None
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("Could not convert schema for tool %s: %s", tool.name, exc)
        parameters = None

    description = tool.description or f"Execute {tool.name}"
    if encourage_web_search and _is_search_tool_name(tool.name):
        description = (
            f"{description} Use this tool for current, latest, recent, "
            "time-sensitive, or explicitly requested online information."
        )
    return LiveTool(tool.name, description, parameters)


def _format_tools_for_live(
    tools: list[llm.Tool],
    custom_serializer: Callable[[Any], Any] | None = None,
    encourage_web_search: bool = False,
) -> list[LiveTool]:
    """Convert Home Assistant tools to provider-neutral declarations."""
    return [
        _format_tool_for_live(tool, custom_serializer, encourage_web_search)
        for tool in tools
    ]


def _add_end_conversation_tool(
    tools: list[LiveTool],
) -> list[LiveTool]:
    """Add the integration-owned conversation completion callback."""
    return [*tools, _END_CONVERSATION_TOOL]


def _add_end_conversation_instruction(system_instruction: str) -> str:
    """Tell Gemini when to finish the Home Assistant conversation."""
    return f"{system_instruction}\n\n{_END_CONVERSATION_INSTRUCTION}"


def _add_show_text_tool(
    tools: list[LiveTool],
) -> list[LiveTool]:
    """Add the integration-owned show text callback."""
    return [*tools, _SHOW_TEXT_TOOL]


def _add_show_text_instruction(system_instruction: str) -> str:
    """Tell the live model to use the show_text callback to show text to the user."""
    return f"{system_instruction}\n\n{_SHOW_TEXT_INSTRUCTION}"


def _add_search_tool_instruction(
    system_instruction: str,
    tools: list[llm.Tool],
    encourage_web_search: bool,
    native_search_grounding: bool = False,
) -> str:
    """Tell the model when to use native or exposed web search."""
    has_exposed_search_tool = any(_is_search_tool_name(tool.name) for tool in tools)
    if not encourage_web_search or not (
        native_search_grounding or has_exposed_search_tool
    ):
        return system_instruction
    return f"{system_instruction}\n\n{_SEARCH_TOOL_INSTRUCTION}"


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
    """Set up the selected realtime provider's STT platform."""
    config = {**config_entry.data, **config_entry.options}
    provider = config.get(CONF_PROVIDER, PROVIDER_GEMINI)
    entity_class = GPTRealtimeSTT if provider == PROVIDER_OPENAI else GeminiLiveSTT
    async_add_entities([entity_class(config_entry)])


# ---------------------------------------------------------------------------
# STT Entity
# ---------------------------------------------------------------------------


class LiveModelSTT(SpeechToTextEntity):
    """Shared speech-to-text pipeline for realtime model providers."""

    _attr_should_poll = False
    integration_domain = DOMAIN
    integration_name = "Live Model"
    session_manager_key = GEMINI_SESSION_MANAGER_KEY
    turn_store_key = GEMINI_TURN_STORE_KEY
    tts_placeholder = GEMINI_LIVE_TTS_PLACEHOLDER
    transcribe_config_key = "transcribe_output"
    default_transcribe = True
    default_system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    supported_language_codes = SUPPORTED_LANGUAGES
    supports_search_grounding = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the STT entity."""
        self.entry = entry
        self._attr_name = self.integration_name
        self._attr_unique_id = f"{entry.entry_id}_stt"

    async def _async_create_client(self, api_key: str) -> Any:
        """Create the selected provider adapter."""
        raise NotImplementedError

    @staticmethod
    def _api_error_message(exc: BaseException) -> str | None:
        """Return provider-specific user-visible API errors."""
        return None

    def _set_detailed_logging(self, enabled: bool) -> None:
        """Set provider package logging verbosity."""
        set_detailed_logging(enabled)

    @property
    def name(self) -> str:
        return self._attr_name

    @property
    def unique_id(self) -> str:
        return self._attr_unique_id

    @property
    def audio_processing(self) -> SpeechAudioProcessing:
        """Let the live provider own turn detection while barge-in is enabled.

        Home Assistant's end-of-command VAD would stop feeding the microphone
        stream after the initial user command, which prevents barge-in. With
        barge-in enabled the provider's own VAD decides when the user stops
        and starts speaking, so the stream must stay open.
        """
        config = {**self.entry.data, **self.entry.options}

        if not config.get(
            CONF_SUPPORT_BARGE_IN,
            DEFAULT_SUPPORT_BARGE_IN,
        ):
            return DEFAULT_AUDIO_PROCESSING

        return SpeechAudioProcessing(
            requires_external_vad=False,
            prefers_auto_gain_enabled=True,
            prefers_noise_reduction_enabled=True,
        )

    async def _async_run_audio_stream_sdk(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
        api_key: str,
        model: str,
        voice: str,
        custom_instruction: str,
        transcribe_output: bool,
        encourage_web_search: bool,
        show_text: bool,
        support_barge_in: bool,
        result_future: asyncio.Future[SpeechResult],
        conversation_id: str,
        device_id: str | None,
        pipeline_context: Context | None = None,
    ) -> SpeechResult:
        """Process audio using the configured live-model client."""
        turn_id = uuid4().hex[:8]
        show_text_content: str | None = None
        started_at = time.monotonic()
        entry_data = self.hass.data[self.integration_domain][self.entry.entry_id]
        session_manager = entry_data[self.session_manager_key]
        if await session_manager.async_prepare_for_new_turn(conversation_id):
            _LOGGER.info(
                "[turn=%s] cancelled unfinished provider turn for conversation %s",
                turn_id,
                conversation_id,
            )
        session_manager.reset_conversation(conversation_id)
        turn_store = entry_data[self.turn_store_key]
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
            "[turn=%s] SDK helper start api_key_present=%s model=%s voice=%s language=%s device_id=%s",
            turn_id,
            bool(api_key),
            model,
            voice,
            metadata.language or "en",
            device_id,
        )

        llm_api: llm.APIInstance | None = None
        ha_tools: list[llm.Tool] = []
        system_instruction = custom_instruction or self.default_system_instruction
        config = {**self.entry.data, **self.entry.options}
        api_ids = config.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])

        if api_ids:
            try:
                llm_api = await llm.async_get_api(
                    hass=self.hass,
                    api_id=api_ids,
                    llm_context=llm.LLMContext(
                        platform=self.integration_domain,
                        context=(
                            pipeline_context
                            if pipeline_context is not None
                            else Context()
                        ),
                        language=metadata.language or "en",
                        assistant="conversation",
                        device_id=device_id,
                    ),
                )
                ha_tools = llm_api.tools

                api_prompt = llm_api.api_prompt
                if custom_instruction:
                    system_instruction = f"{custom_instruction}\n\n{api_prompt}"
                else:
                    system_instruction = (
                        self.default_system_instruction + "\n\n" + api_prompt
                    )
                system_instruction = _add_search_tool_instruction(
                    system_instruction,
                    ha_tools,
                    encourage_web_search and not self.supports_search_grounding,
                )
                _LOGGER.debug(
                    "Loaded selected Home Assistant LLM APIs with %d tools",
                    len(ha_tools),
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "Could not load selected Home Assistant LLM APIs: %s. "
                    "Tools will be unavailable.",
                    exc,
                )
        else:
            _LOGGER.debug("No Home Assistant LLM APIs selected")

        if self.supports_search_grounding:
            system_instruction = _add_search_tool_instruction(
                system_instruction,
                [],
                encourage_web_search,
                native_search_grounding=True,
            )
        system_instruction = _add_end_conversation_instruction(system_instruction)
        if not transcribe_output and show_text:
            system_instruction = _add_show_text_instruction(system_instruction)

        live_tools = _add_end_conversation_tool(
            _format_tools_for_live(
                ha_tools,
                llm_api.custom_serializer,
                encourage_web_search and not self.supports_search_grounding,
            )
            if llm_api
            else []
        )
        if not transcribe_output and show_text:
            live_tools = _add_show_text_tool(live_tools)
        _LOGGER.debug(
            "Exposing %d tools to the live model: %s",
            len(live_tools),
            [definition.name for definition in live_tools],
        )

        _LOGGER.warning("[turn=%s] creating provider client", turn_id)
        client = await self._async_create_client(api_key)
        _LOGGER.warning(
            "[turn=%s] provider client created tool_count=%d system_instruction_chars=%d",
            turn_id,
            len(live_tools),
            len(system_instruction),
        )
        live_config = LiveConfig(
            model=model,
            voice=voice,
            system_instruction=system_instruction,
            tools=live_tools,
            transcribe_output=transcribe_output,
            support_barge_in=support_barge_in,
            affective_dialog=bool(
                {**self.entry.data, **self.entry.options}.get(
                    CONF_AFFECTIVE_DIALOG, DEFAULT_AFFECTIVE_DIALOG
                )
            ),
            search_grounding=(self.supports_search_grounding and encourage_web_search),
            thinking_level=(
                {**self.entry.data, **self.entry.options}.get(
                    CONF_THINKING_LEVEL,
                    DEFAULT_THINKING_LEVEL,
                )
            ),
        )

        _LOGGER.warning(
            "[turn=%s] live config prepared model=%s voice=%s has_tools=%s "
            "output_transcription=%s barge_in=%s affective_dialog=%s "
            "search_grounding=%s thinking_level=%s",
            turn_id,
            model,
            voice,
            bool(live_tools),
            transcribe_output,
            support_barge_in,
            live_config.affective_dialog,
            live_config.search_grounding,
            live_config.thinking_level,
        )
        if support_barge_in:
            _LOGGER.debug("[turn=%s] barge-in enabled for provider session", turn_id)

        native_audio_model = "native-audio" in (model or "")
        _LOGGER.warning(
            "[turn=%s] setup model=%s native_audio_model=%s tools=%d",
            turn_id,
            model,
            native_audio_model,
            len(live_tools),
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
        stream_chunk_size = (
            BARGE_IN_STREAM_CHUNK_SIZE
            if support_barge_in
            else OPTIMAL_STREAM_CHUNK_SIZE
        )
        response_audio_stream = AudioStream(
            lambda: session_manager.cancel_conversation(self.hass, conversation_id)
        )
        response_audio_resampler = PCM24kTo16kStreamResampler()
        response_text_stream = TextStream() if transcribe_output else None

        # The placeholder must carry the turn id: Home Assistant caches TTS
        # audio by message, so a bare placeholder would make later turns
        # replay the first turn's audio. The same string is used for the
        # published turn's user_text and the STT transcript so the voice-turn
        # lookup still matches.
        fallback_user_text = f"{self.tts_placeholder} {turn_id}"

        _LOGGER.warning(
            "[turn=%s] acquiring live-model session conversation=%s",
            turn_id,
            conversation_id,
        )
        async with session_manager.acquire(
            conversation_id,
            client,
            live_config,
        ) as session:
            async_delete_issue(
                self.hass,
                self.integration_domain,
                _spending_cap_issue_id(self.entry.entry_id),
            )
            async_delete_issue(
                self.hass,
                self.integration_domain,
                _prepayment_credits_issue_id(self.entry.entry_id),
            )
            async_delete_issue(
                self.hass,
                self.integration_domain,
                _openai_no_credits_issue_id(self.entry.entry_id),
            )
            _LOGGER.warning(
                "[turn=%s] acquired live-model session conversation=%s",
                turn_id,
                conversation_id,
            )

            async def send_audio() -> None:
                nonlocal audio_sent
                try:
                    first_chunk = True
                    audio_buffer = bytearray()
                    diagnostics_enabled = _LOGGER.isEnabledFor(logging.DEBUG)
                    pcm_for_diag: list[bytes] = []
                    chunk_count = 0

                    _LOGGER.warning("[turn=%s] send_audio task spawned", turn_id)

                    async for chunk in stream:
                        if not chunk:
                            continue
                        if not support_barge_in and gemini_replied.is_set():
                            _LOGGER.warning(
                                "[turn=%s] send_audio stopped because the model started replying",
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

                        while len(audio_buffer) >= stream_chunk_size:
                            dispatch_chunk = bytes(audio_buffer[:stream_chunk_size])
                            del audio_buffer[:stream_chunk_size]

                            chunk_count += 1
                            if diagnostics_enabled:
                                pcm_for_diag.append(dispatch_chunk)
                            _LOGGER.debug(
                                "[turn=%s] Shipping optimized media chunk %d (%d bytes)",
                                turn_id,
                                chunk_count,
                                len(dispatch_chunk),
                            )
                            _LOGGER.debug(
                                "[turn=%s] sending audio_pcm_16k chunk_size=%d",
                                turn_id,
                                len(dispatch_chunk),
                            )
                            await session.send_audio(dispatch_chunk)
                            audio_sent = True

                    if len(audio_buffer) > 0 and (
                        support_barge_in or not gemini_replied.is_set()
                    ):
                        chunk_count += 1
                        dispatch_chunk = bytes(audio_buffer)
                        if diagnostics_enabled:
                            pcm_for_diag.append(dispatch_chunk)
                        _LOGGER.debug(
                            "[turn=%s] flushing trailing audio chunk size=%d",
                            turn_id,
                            len(dispatch_chunk),
                        )
                        await session.send_audio(dispatch_chunk)
                        audio_sent = True

                    if diagnostics_enabled and pcm_for_diag:
                        _LOGGER.warning(
                            "[turn=%s] Finished voice streaming. Total blocks dispatched=%d. Metrics=%s",
                            turn_id,
                            chunk_count,
                            _analyse_pcm(b"".join(pcm_for_diag)),
                        )

                    if audio_sent and (support_barge_in or not gemini_replied.is_set()):
                        _LOGGER.debug("[turn=%s] signalling audio stream end", turn_id)
                        await session.end_audio()
                except asyncio.CancelledError:
                    _LOGGER.warning(
                        "[turn=%s] audio sender cancelled — the model started replying",
                        turn_id,
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.exception(
                        "[turn=%s] Failure inside send_audio: %s (last outgoing client message: %s)",
                        turn_id,
                        exc,
                        getattr(session, "last_outgoing", "unknown"),
                    )
                    raise

            async def receive_responses() -> None:
                nonlocal audio_response_bytes, audio_response_chunk_count
                nonlocal last_response_activity, show_text_content
                replacement_response_pending = False
                try:
                    _LOGGER.warning("[turn=%s] receive_responses started", turn_id)
                    async for response in session.receive():
                        # Audio packets are frequent and must stay on the hot
                        # path. Logging every packet can stall Home Assistant's
                        # event loop and turn steady PCM into bursty delivery.
                        if not response.audio:
                            _LOGGER.debug(
                                "[turn=%s] received event tool_calls=%s text=%s go_away=%s session_resumption_update=%s",
                                turn_id,
                                bool(response.tool_calls),
                                bool(response.text or response.output_transcript),
                                bool(response.go_away),
                                bool(response.session_resumption_update),
                            )
                        if response.user_activity_started:
                            _LOGGER.debug(
                                "[turn=%s] user activity started while model response active",
                                turn_id,
                            )
                        if response.user_activity_stopped:
                            _LOGGER.debug("[turn=%s] user activity stopped", turn_id)
                        if response.interrupted:
                            # An interrupted generation is followed by a
                            # replacement response in the same provider turn.
                            # Discard queued output audio but keep the Home
                            # Assistant stream open.
                            response_audio_stream.interrupt()
                            response_audio_resampler.reset()
                            replacement_response_pending = True
                            text_response_parts.clear()
                            _LOGGER.debug(
                                "[turn=%s] provider response interrupted; discarded queued output audio and waiting for replacement response",
                                turn_id,
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

                        if response.tool_calls:
                            last_response_activity = time.monotonic()
                            function_calls = response.tool_calls
                            function_responses = []

                            _LOGGER.warning(
                                "[turn=%s] tool_call count=%d",
                                turn_id,
                                len(function_calls),
                            )

                            for call in function_calls:
                                tool_name = call.name or ""
                                tool_args = call.arguments
                                call_id = call.call_id
                                _LOGGER.debug(
                                    "[turn=%s] LLM tool call name=%s id=%s arguments=%r",
                                    turn_id,
                                    tool_name,
                                    call_id,
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
                                elif tool_name == SHOW_TEXT_TOOL_NAME:
                                    show_text_content = tool_args.get("text")
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
                                        _LOGGER.error(
                                            "Tool %s failed: %s", tool_name, err
                                        )
                                        tool_result = {"error": str(err)}
                                else:
                                    tool_result = {"error": "HA LLM API not available"}
                                tool_result = _validate_tool_results(tool_result)

                                _LOGGER.debug(
                                    "[turn=%s] LLM tool response name=%s id=%s response=%r",
                                    turn_id,
                                    tool_name,
                                    call_id,
                                    tool_result,
                                )

                                _LOGGER.warning(
                                    "[turn=%s] tool response prepared name=%s id=%s result_type=%s",
                                    turn_id,
                                    tool_name,
                                    call_id,
                                    type(tool_result).__name__,
                                )

                                function_responses.append(
                                    LiveToolResponse(tool_name, call_id, tool_result)
                                )

                            if function_responses:
                                _LOGGER.warning(
                                    "[turn=%s] sending %d tool response(s) to the live model",
                                    turn_id,
                                    len(function_responses),
                                )
                                await session.send_tool_responses(function_responses)
                                _LOGGER.warning(
                                    "[turn=%s] sent %d tool response(s) to the live model",
                                    turn_id,
                                    len(function_responses),
                                )

                        last_response_activity = time.monotonic()

                        if response.text:
                            text_response_parts.append(response.text)
                            if response_text_stream is not None:
                                response_text_stream.add_chunk(response.text)

                        if response.audio:
                            if not gemini_replied.is_set():
                                gemini_replied.set()
                                if support_barge_in:
                                    _LOGGER.debug(
                                        "[turn=%s] microphone forwarding remains active during model output",
                                        turn_id,
                                    )
                            if replacement_response_pending:
                                replacement_response_pending = False
                                _LOGGER.debug(
                                    "[turn=%s] replacement response started",
                                    turn_id,
                                )
                            audio_response_chunk_count += 1
                            audio_response_bytes += len(response.audio)
                            await response_audio_stream.add_chunk(
                                response_audio_resampler.process(response.audio)
                            )
                            first_audio.set()

                        if transcribe_output and response.output_transcript:
                            transcription = response.output_transcript
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

                        if response.input_transcript:
                            transcription = response.input_transcript
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

                        if response.turn_complete:
                            if support_barge_in and replacement_response_pending:
                                # This completes the interrupted assistant
                                # generation, not the whole HA response stream.
                                # The provider still owes a replacement response.
                                _LOGGER.debug(
                                    "[turn=%s] interrupted generation completed; ignoring terminal turn completion",
                                    turn_id,
                                )
                                continue
                            if native_audio_model and not gemini_replied.is_set():
                                _LOGGER.warning(
                                    "[turn=%s] turnComplete before audio; keeping session open and waiting",
                                    turn_id,
                                )
                                continue
                            await response_audio_stream.add_chunk(
                                response_audio_resampler.flush()
                            )
                            if support_barge_in and first_audio.is_set():
                                # The provider turn is complete, but Home
                                # Assistant's pipeline run is still open and
                                # the response may still be playing on the
                                # satellite. End this response's audio so the
                                # pipeline can finish its TTS stage, but keep
                                # forwarding the microphone and receiving
                                # events so the user can speak again while the
                                # pipeline is open.
                                _LOGGER.debug(
                                    "[turn=%s] turnComplete; keeping the microphone open until the pipeline closes (audio_chunks=%d)",
                                    turn_id,
                                    audio_response_chunk_count,
                                )
                                response_audio_stream.finish()
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
                            "[turn=%s] live-model connection closed normally",
                            turn_id,
                        )
                    else:
                        _LOGGER.exception(
                            "[turn=%s] error in receive_responses: %s "
                            "(last outgoing client message: %s)",
                            turn_id,
                            exc,
                            getattr(session, "last_outgoing", "unknown"),
                        )
                    raise

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_responses())
            _LOGGER.warning("[turn=%s] created send and receive tasks", turn_id)

            async def publish_streaming_turn() -> None:
                """Release the pipeline once the live model starts producing audio."""
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
                    "".join(input_transcript_parts).strip() or fallback_user_text
                )
                # HA persistently caches TTS audio by message. A per-turn message
                # prevents it from replaying an earlier live-model audio stream.
                if (
                    not transcribe_output
                    and show_text
                    and show_text_content is not None
                ):
                    tts_message = show_text_content
                else:
                    tts_message = f"{self.tts_placeholder} {turn_id}"
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
                        "[turn=%s] cancelling send task because the model started replying",
                        turn_id,
                    )
                    send_task.cancel()

            # With barge-in the microphone sender must live for the whole
            # provider turn so the user can interrupt while the model speaks.
            cancel_on_reply_task: asyncio.Task[None] | None = None
            if not support_barge_in:
                cancel_on_reply_task = asyncio.create_task(_cancel_sender_on_reply())
            try:
                done: set[asyncio.Task[Any]] = set()
                while not done:
                    remaining = RESPONSE_INACTIVITY_TIMEOUT - (
                        time.monotonic() - last_response_activity
                    )
                    if remaining <= 0:
                        _LOGGER.warning(
                            "[turn=%s] cancelling receive task after %.1fs without response activity",
                            turn_id,
                            RESPONSE_INACTIVITY_TIMEOUT,
                        )
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass
                        break
                    done, _pending = await asyncio.wait(
                        [send_task, receive_task],
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                for completed_task in (send_task, receive_task):
                    if not completed_task.done():
                        continue
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
                        raise

                if receive_task in done:
                    if not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                elif not audio_sent:
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass
                    return SpeechResult(None, SpeechResultState.ERROR)
                elif support_barge_in:
                    # The microphone stream ended: Home Assistant closed it at
                    # the end of the pipeline run, so the provider turn is
                    # over. Keep listening for the whole run rather than
                    # stopping at turnComplete.
                    _LOGGER.debug(
                        "[turn=%s] microphone stream closed by pipeline; ending provider turn",
                        turn_id,
                    )
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass
                else:
                    while not receive_task.done():
                        remaining = RESPONSE_INACTIVITY_TIMEOUT - (
                            time.monotonic() - last_response_activity
                        )
                        if remaining <= 0:
                            _LOGGER.warning(
                                "[turn=%s] cancelling receive task after %.1fs without response activity",
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
                if cancel_on_reply_task is not None and not cancel_on_reply_task.done():
                    cancel_on_reply_task.cancel()
                tasks: list[asyncio.Task[Any]] = [send_task, receive_task]
                if cancel_on_reply_task is not None:
                    tasks.append(cancel_on_reply_task)
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
                "STT: live-model audio ready: text=%d chars, raw_audio=%d bytes",
                len(response_text),
                all_audio_24k_len,
            )
        else:
            _LOGGER.warning("STT: No audio response received from the live model")

        final_text = input_transcript or response_text
        if first_audio.is_set():
            # Must match the user_text published with the streaming turn so
            # the conversation agent's voice-turn lookup succeeds.
            return SpeechResult(
                input_transcript or fallback_user_text,
                SpeechResultState.SUCCESS,
            )
        if not final_text:
            _LOGGER.error(
                "STT: Live model returned no usable transcript or response text"
            )
            return SpeechResult(None, SpeechResultState.ERROR)

        if not transcribe_output and show_text and show_text_content is not None:
            assistant_text = show_text_content
        else:
            assistant_text = response_text

        conversation_complete = not session_manager.should_continue_conversation(
            conversation_id
        )
        if assistant_text or conversation_complete:
            turn_store.add_voice_turn(
                PipelineTurn(
                    conversation_id=conversation_id,
                    user_text=final_text,
                    assistant_text=(assistant_text or fallback_user_text),
                    audio=b"",
                    complete_conversation=conversation_complete,
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
        transcribe_output: bool,
        encourage_web_search: bool,
        show_text: bool,
        support_barge_in: bool,
    ) -> SpeechResult:
        """Run the Live turn in the background so TTS can consume it immediately."""
        result_future: asyncio.Future[SpeechResult] = asyncio.Future()
        conversation_id, device_id, pipeline_context = active_pipeline_context(
            self.hass,
            self.entity_id,
        )
        task = self.hass.async_create_background_task(
            self._async_run_audio_stream_sdk(
                metadata,
                stream,
                api_key,
                model,
                voice,
                custom_instruction,
                transcribe_output,
                encourage_web_search,
                show_text,
                support_barge_in,
                result_future,
                conversation_id,
                device_id,
                pipeline_context,
            ),
            f"{self.integration_name} audio turn",
        )

        def set_final_result(completed_task: asyncio.Task[SpeechResult]) -> None:
            if result_future.done():
                return
            try:
                result_future.set_result(completed_task.result())
            except asyncio.CancelledError:
                result_future.set_result(SpeechResult(None, SpeechResultState.ERROR))
            except Exception as exc:  # noqa: BLE001
                if user_message := self._api_error_message(exc):
                    if user_message == _SPENDING_CAP_USER_MESSAGE:
                        issue_id = _spending_cap_issue_id(self.entry.entry_id)
                        issue_url = _SPENDING_CAP_URL
                        translation_key = "spending_cap_exceeded"
                        url_placeholder = "spending_cap_url"
                        reason = "the monthly spending cap was exceeded"
                    elif user_message == _PREPAYMENT_CREDITS_USER_MESSAGE:
                        issue_id = _prepayment_credits_issue_id(self.entry.entry_id)
                        issue_url = _PREPAYMENT_CREDITS_URL
                        translation_key = "spending_cap_exceeded"
                        url_placeholder = "spending_cap_url"
                        reason = "prepayment credits are depleted"
                    else:
                        issue_id = _openai_no_credits_issue_id(self.entry.entry_id)
                        issue_url = _OPENAI_NO_CREDITS_URL
                        translation_key = "openai_no_credits"
                        url_placeholder = "billing_url"
                        reason = "OpenAI credits are depleted"
                    translation_placeholders = {
                        "entry_title": self.entry.title,
                        url_placeholder: issue_url,
                    }
                    async_create_issue(
                        self.hass,
                        self.integration_domain,
                        issue_id,
                        is_fixable=False,
                        is_persistent=False,
                        learn_more_url=issue_url,
                        severity=IssueSeverity.ERROR,
                        translation_key=translation_key,
                        translation_placeholders=translation_placeholders,
                    )
                    entry_data = self.hass.data[self.integration_domain][
                        self.entry.entry_id
                    ]
                    entry_data[self.session_manager_key].complete_conversation(
                        conversation_id
                    )
                    turn_store = entry_data[self.turn_store_key]
                    turn_store.add_voice_turn(
                        PipelineTurn(
                            conversation_id=conversation_id,
                            user_text=user_message,
                            assistant_text=user_message,
                            audio=b"",
                            complete_conversation=True,
                        )
                    )
                    _LOGGER.warning(
                        "%s is unavailable because %s",
                        self.integration_name,
                        reason,
                    )
                    result_future.set_result(
                        SpeechResult(user_message, SpeechResultState.SUCCESS)
                    )
                else:
                    _LOGGER.exception("%s audio turn failed", self.integration_name)
                    result_future.set_result(
                        SpeechResult(None, SpeechResultState.ERROR)
                    )

        task.add_done_callback(set_final_result)
        try:
            return await result_future
        except asyncio.CancelledError:
            task.cancel()
            raise

    @property
    def supported_languages(self) -> list[str]:
        return self.supported_language_codes

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
        """Send the audio stream directly to the configured live model."""
        turn_id = uuid4().hex[:8]
        started_at = time.monotonic()
        config = {**self.entry.data, **self.entry.options}
        api_key = config.get(CONF_API_KEY)
        model = config.get(CONF_MODEL)
        voice = config.get(CONF_VOICE)
        custom_instruction = config.get(CONF_SYSTEM_INSTRUCTION, "")
        user_requested_transcription = bool(
            config.get(self.transcribe_config_key, self.default_transcribe)
        )
        support_barge_in = (
            bool(
                config.get(
                    CONF_SUPPORT_BARGE_IN,
                    DEFAULT_SUPPORT_BARGE_IN,
                )
            )
            and supports_tts_interruption()
        )
        if self.supports_search_grounding:
            encourage_web_search = bool(
                config.get(
                    CONF_SEARCH_GROUNDING,
                    DEFAULT_SEARCH_GROUNDING,
                )
            )
        else:
            encourage_web_search = bool(
                config.get(
                    CONF_ENCOURAGE_WEB_SEARCH,
                    DEFAULT_ENCOURAGE_WEB_SEARCH,
                )
            )
        show_text = bool(config.get(CONF_SHOW_TEXT, DEFAULT_SHOW_TEXT))
        self._set_detailed_logging(bool(config.get(CONF_DETAILED_LOGGING, False)))

        _LOGGER.warning(
            "[turn=%s] STT start language=%s model=%s voice=%s detailed_logging=%s barge_in=%s",
            turn_id,
            metadata.language or "en",
            model,
            voice,
            bool(config.get(CONF_DETAILED_LOGGING, False)),
            support_barge_in,
        )

        if not api_key:
            _LOGGER.error("API key not configured for %s", self.integration_name)
            return SpeechResult(None, SpeechResultState.ERROR)

        return await self._async_process_audio_stream_sdk(
            metadata,
            stream,
            api_key,
            model,
            voice,
            custom_instruction,
            user_requested_transcription,
            encourage_web_search,
            show_text,
            support_barge_in,
        )


class GeminiLiveSTT(LiveModelSTT):
    """Stream a Home Assistant voice turn through Gemini Live."""

    integration_name = "Gemini Live"
    transcribe_config_key = CONF_TRANSCRIBE_GEMINI
    default_transcribe = DEFAULT_TRANSCRIBE_GEMINI
    supports_search_grounding = True

    async def _async_create_client(self, api_key: str) -> GeminiLiveClient:
        """Create the Gemini provider adapter."""
        affective_dialog = bool(
            {**self.entry.data, **self.entry.options}.get(
                CONF_AFFECTIVE_DIALOG, DEFAULT_AFFECTIVE_DIALOG
            )
        )
        return await async_create_gemini_client(
            self.hass, api_key, affective_dialog=affective_dialog
        )

    @staticmethod
    def _api_error_message(exc: BaseException) -> str | None:
        """Return Google billing errors that the user can resolve."""
        return _user_visible_api_error(exc)


class GPTRealtimeSTT(LiveModelSTT):
    """Stream a Home Assistant voice turn through OpenAI Realtime."""

    integration_name = "GPT Realtime"
    transcribe_config_key = CONF_TRANSCRIBE_GPT
    default_transcribe = DEFAULT_TRANSCRIBE_GPT
    default_system_instruction = OPENAI_SYSTEM_INSTRUCTION

    async def _async_create_client(self, api_key: str) -> OpenAIRealtimeClient:
        """Create an OpenAI Realtime WebSocket client."""
        return OpenAIRealtimeClient(async_get_clientsession(self.hass), api_key)

    @staticmethod
    def _api_error_message(exc: BaseException) -> str | None:
        """Return OpenAI billing errors that the user can resolve."""
        return _user_visible_api_error(exc)
