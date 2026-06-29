"""Conversation platform for Gemini Live."""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import chat_session, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.intent import IntentResponse

from .const import (
    CONF_API_KEY,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_MODEL,
    CONF_SYSTEM_INSTRUCTION,
    CONF_TRANSCRIBE_GEMINI,
    CONF_SHOW_TEXT,
    CONF_VOICE,
    DEFAULT_SYSTEM_INSTRUCTION,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_SHOW_TEXT,
    DOMAIN,
    GEMINI_LIVE_TTS_PLACEHOLDER,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
    SUPPORTED_LANGUAGES,
)
from .stt import (
    END_CONVERSATION_TOOL_NAME,
    _add_end_conversation_instruction,
    _add_end_conversation_tool,
    _add_search_tool_instruction,
    _escape_decode,
    _format_tools_for_gemini_live,
    _is_connection_closed_ok,
    _validate_tool_results,
    SHOW_TEXT_TOOL_NAME,
    _add_show_text_instruction,
    _add_show_text_tool,
)
from .runtime import AudioStream, new_conversation_id
from .utils import pcm_to_wav, resample_24k_to_16k

_LOGGER = logging.getLogger(__name__)


def _ensure_unique_tts_placeholder(assistant_text: str) -> str:
    """Make the streaming placeholder unique so HA cannot reuse cached audio."""
    if assistant_text == GEMINI_LIVE_TTS_PLACEHOLDER:
        return f"{GEMINI_LIVE_TTS_PLACEHOLDER} {uuid4().hex[:8]}"
    return assistant_text


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Gemini Live Conversation platform."""
    async_add_entities([GeminiLiveConversationAgent(hass, config_entry)])


class GeminiLiveConversationAgent(conversation.ConversationEntity):
    """Gemini Live conversation entity."""

    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._name = "Gemini Live"
        self._unique_id = f"{entry.entry_id}_conversation"

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return self._unique_id

    @property
    def supported_features(self) -> conversation.ConversationEntityFeature:
        """Return supported features."""
        return conversation.ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages."""
        return SUPPORTED_LANGUAGES

    def _fire_conversation_entry(
        self,
        user_transcript: str,
        assistant_text: str,
    ) -> None:
        """Fire the text event used by automations and dashboards."""
        self.hass.bus.async_fire(
            "gemini_live_conversation_entry",
            {
                "user_transcript": user_transcript,
                "assistant_text": assistant_text,
            },
        )
        _LOGGER.debug(
            "Fired gemini_live_conversation_entry: user=%r assistant=%r",
            (user_transcript or "")[:80],
            (assistant_text or "")[:80],
        )

    async def _async_get_llm_api(
        self,
        llm_context: llm.LLMContext,
    ) -> tuple[llm.APIInstance | None, list[dict[str, Any]], str]:
        """Load HA Assist tools and the final Gemini system instruction."""
        config = {**self.entry.data, **self.entry.options}
        custom_instruction = config.get(CONF_SYSTEM_INSTRUCTION, "")
        encourage_web_search = bool(
            config.get(CONF_ENCOURAGE_WEB_SEARCH, DEFAULT_ENCOURAGE_WEB_SEARCH)
        )
        transcribe_gemini = bool(
            config.get(CONF_TRANSCRIBE_GEMINI, DEFAULT_TRANSCRIBE_GEMINI)
        )
        show_text = bool(
            config.get(CONF_SHOW_TEXT, DEFAULT_SHOW_TEXT)
        )
        system_instruction = custom_instruction or DEFAULT_SYSTEM_INSTRUCTION

        try:
            llm_api = await llm.async_get_api(
                hass=self.hass,
                api_id=llm.LLM_API_ASSIST,
                llm_context=llm_context,
            )
            api_prompt = llm_api.api_prompt
            if custom_instruction:
                system_instruction = f"{custom_instruction}\n\n{api_prompt}"
            else:
                system_instruction = DEFAULT_SYSTEM_INSTRUCTION + "\n\n" + api_prompt
            system_instruction = _add_search_tool_instruction(
                system_instruction,
                llm_api.tools,
                encourage_web_search,
            )
            system_instruction = _add_end_conversation_instruction(system_instruction)
            if not transcribe_gemini and show_text:
                system_instruction = _add_show_text_instruction(system_instruction)

            gemini_tools = _add_end_conversation_tool(
                _format_tools_for_gemini_live(
                    llm_api.tools,
                    llm_api.custom_serializer,
                    encourage_web_search,
                )
            )
            if not transcribe_gemini and show_text:
                gemini_tools = _add_show_text_tool(gemini_tools)

            _LOGGER.debug(
                "Conversation text path loaded %d HA Assist tools",
                len(gemini_tools),
            )
            return llm_api, gemini_tools, system_instruction
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "Could not load HA Assist LLM API for text path: %s. Tools will be unavailable.",
                exc,
            )
            gemini_tools = _add_end_conversation_tool([])
            system_instruction = _add_end_conversation_instruction(system_instruction)
            if not transcribe_gemini and show_text:
                system_instruction = _add_show_text_instruction(system_instruction)
                gemini_tools = _add_show_text_tool(gemini_tools)
            return (
                None,
                gemini_tools,
                system_instruction,
            )

    async def _async_process_text_live(
        self,
        user_text: str,
        user_input: conversation.ConversationInput,
        conversation_id: str,
    ) -> str | None:
        """Send typed text to Gemini Live and cache the returned audio for TTS."""
        turn_id = uuid4().hex[:8]
        show_text_content: str | None = None
        started_at = time.monotonic()
        config = {**self.entry.data, **self.entry.options}
        api_key = config.get(CONF_API_KEY)
        model = config.get(CONF_MODEL)
        voice = config.get(CONF_VOICE)
        language = user_input.language or "en"
        transcribe_gemini = bool(
            config.get(CONF_TRANSCRIBE_GEMINI, DEFAULT_TRANSCRIBE_GEMINI)
        )
        show_text = bool(
            config.get(CONF_SHOW_TEXT, DEFAULT_SHOW_TEXT)
        )

        if not api_key:
            _LOGGER.error("API Key not configured for Gemini Live")
            return None

        llm_api, gemini_tools, system_instruction = await self._async_get_llm_api(
            user_input.as_llm_context(DOMAIN)
        )
        entry_data = self.hass.data[DOMAIN][self.entry.entry_id]
        session_manager = entry_data[GEMINI_SESSION_MANAGER_KEY]
        turn_store = entry_data[GEMINI_TURN_STORE_KEY]

        live_config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": voice}
                }
            },
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "realtime_input_config": {
                "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY"
            },
        }
        if gemini_tools:
            live_config["tools"] = gemini_tools

        text_response_parts: list[str] = []
        audio_response_chunks: list[bytes] = []
        resampled_pcm_chunks: list[bytes] = []
        wav_data = b""
        native_audio_model = "native-audio" in (model or "")

        _LOGGER.warning(
            "[turn=%s] conversation text path start model=%s voice=%s tools=%d text=%r",
            turn_id,
            model,
            voice,
            len(gemini_tools),
            user_text[:120],
        )

        client = await self.hass.async_add_executor_job(
            lambda: genai.Client(api_key=api_key)
        )

        try:
            async with session_manager.acquire(
                conversation_id,
                client,
                model,
                live_config,
            ) as session:
                await session.send_realtime_input(text=user_text)

                async with asyncio.timeout(30):
                    async for response in session.receive():
                        if response.tool_call:
                            function_responses = []
                            for call in response.tool_call.function_calls or []:
                                tool_name = call.name or ""
                                tool_args = _escape_decode(call.args or {})
                                call_id = call.id
                                _LOGGER.info(
                                    "Gemini Live text path tool call: %s(%s)",
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
                                elif tool_name == SHOW_TEXT_TOOL_NAME:
                                    show_text_content = tool_args.get("text")
                                    tool_result = {
                                        "success": True,
                                        "displayed": True,
                                    }
                                elif llm_api is not None:
                                    try:
                                        tool_result = await llm_api.async_call_tool(
                                            llm.ToolInput(
                                                tool_name=tool_name,
                                                tool_args=tool_args,
                                            )
                                        )
                                    except Exception as err:  # noqa: BLE001
                                        _LOGGER.error("Tool %s failed: %s", tool_name, err)
                                        tool_result = {"error": str(err)}
                                else:
                                    tool_result = {"error": "HA LLM API not available"}
                                tool_result = _validate_tool_results(tool_result)

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=tool_name,
                                        id=call_id,
                                        response=tool_result,
                                    )
                                )

                            if function_responses:
                                await session.send_tool_response(
                                    function_responses=function_responses
                                )

                        content = response.server_content
                        if not content:
                            continue

                        if content.model_turn:
                            for part in content.model_turn.parts or []:
                                if part.text:
                                    text_response_parts.append(part.text)
                                if part.inline_data and part.inline_data.data:
                                    raw_chunk = part.inline_data.data
                                    audio_response_chunks.append(raw_chunk)
                                    resampled_pcm_chunks.append(
                                        resample_24k_to_16k(raw_chunk)
                                    )
                                    wav_data = pcm_to_wav(
                                        b"".join(resampled_pcm_chunks),
                                        16000,
                                    )

                        if content.output_transcription and content.output_transcription.text:
                            text_response_parts.append(content.output_transcription.text)

                        if content.turn_complete:
                            if native_audio_model and not audio_response_chunks:
                                _LOGGER.warning(
                                    "[turn=%s] text path turnComplete before audio; waiting",
                                    turn_id,
                                )
                                continue
                            break
        except TimeoutError:
            _LOGGER.error("[turn=%s] Gemini Live text path timed out", turn_id)
            return None
        except Exception as exc:  # noqa: BLE001
            if _is_connection_closed_ok(exc):
                _LOGGER.warning(
                    "[turn=%s] Gemini Live text path websocket closed normally",
                    turn_id,
                )
            else:
                _LOGGER.exception(
                    "[turn=%s] error in Gemini Live text path: %s",
                    turn_id,
                    exc,
                )
                return None

        if not transcribe_gemini and show_text and show_text_content is not None:
            assistant_text = show_text_content
        else:
            assistant_text = "".join(text_response_parts)

        if not assistant_text:
            _LOGGER.error("[turn=%s] Gemini text path returned no usable text", turn_id)
            return None

        turn_store.add_audio(assistant_text, wav_data)

        _LOGGER.warning(
            "[turn=%s] conversation text path complete text_chars=%d audio_chunks=%d wav_bytes=%d elapsed=%.3fs",
            turn_id,
            len(assistant_text),
            len(audio_response_chunks),
            len(wav_data),
            time.monotonic() - started_at,
        )
        return assistant_text

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process voice pass-through responses and typed text input."""
        entry_data = self.hass.data[DOMAIN][self.entry.entry_id]
        turn_store = entry_data[GEMINI_TURN_STORE_KEY]
        language = user_input.language or "en"
        input_text = user_input.text or ""
        current_chat_session = chat_session.current_session.get()
        conversation_id = (
            current_chat_session.conversation_id
            if current_chat_session is not None
            else user_input.conversation_id
        ) or new_conversation_id()
        session_manager = entry_data[GEMINI_SESSION_MANAGER_KEY]
        if current_chat_session is not None:
            session_manager.register_chat_session(self.hass, current_chat_session)

        _LOGGER.debug("Conversation Agent received input text: %s", input_text)

        voice_turn = turn_store.take_voice_turn(conversation_id, input_text)
        if voice_turn:
            user_transcript = input_text
            fallback_assistant_text = _ensure_unique_tts_placeholder(
                voice_turn.assistant_text
            )
            if voice_turn.assistant_text_stream is not None:
                if not isinstance(voice_turn.audio, AudioStream):
                    raise RuntimeError("Streaming transcript has no streaming audio")
                turn_store.add_streaming_audio(
                    voice_turn.assistant_text_stream,
                    voice_turn.audio,
                )

                async def transcript_deltas():
                    """Yield Gemini's response transcript into Home Assistant."""
                    yield {"role": "assistant"}
                    received_text = False
                    async for chunk in voice_turn.assistant_text_stream.async_chunks():
                        received_text = True
                        yield {"content": chunk}
                    if not received_text:
                        yield {"content": fallback_assistant_text}

                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    transcript_deltas(),
                ):
                    pass
                assistant_text = (
                    voice_turn.assistant_text_stream.text or fallback_assistant_text
                )
            else:
                assistant_text = fallback_assistant_text
                turn_store.add_audio(assistant_text, voice_turn.audio)
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content=assistant_text,
                    )
                )
        else:
            session_manager.reset_conversation(conversation_id)
            user_transcript = input_text
            assistant_text = await self._async_process_text_live(
                input_text,
                user_input,
                conversation_id,
            ) or "Sorry, I could not get a response from Gemini Live."
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=assistant_text,
                )
            )

        self._fire_conversation_entry(user_transcript, assistant_text)

        intent_response = IntentResponse(language=language)
        intent_response.async_set_speech(assistant_text)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=(
                session_manager.should_continue_conversation(conversation_id)
            ),
        )
