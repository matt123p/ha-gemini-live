"""Text-to-Speech platform for Gemini Live."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from typing import Any

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_BYTES,
    ATTR_PREFERRED_SAMPLE_CHANNELS,
    ATTR_PREFERRED_SAMPLE_RATE,
    TextToSpeechEntity,
    TtsAudioType,
)
from homeassistant.components.tts.entity import TTSAudioRequest, TTSAudioResponse
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_PROVIDER,
    CONF_SUPPORT_BARGE_IN,
    DOMAIN,
    GEMINI_TURN_STORE_KEY,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    SUPPORTED_LANGUAGES,
)
from .runtime import AudioStream
from .utils import pcm_to_wav, streaming_wav_header

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Gemini Live TTS platform."""
    config = {**config_entry.data, **config_entry.options}
    provider = config.get(CONF_PROVIDER, PROVIDER_GEMINI)
    entity_class = GPTRealtimeTTS if provider == PROVIDER_OPENAI else GeminiLiveTTS
    async_add_entities([entity_class(config_entry)])


class GeminiLiveTTS(TextToSpeechEntity):
    """Gemini Live TTS Entity.

    This entity does NOT synthesise audio itself. Instead, it retrieves or
    streams the native audio from the Gemini Live turn handled by the STT
    stage of the same pipeline run.

    The conversation text is always delivered via the
    `gemini_live_conversation_entry` event fired by conversation.py. If no
    cached audio is available, a short silence lets the pipeline complete
    cleanly.
    """

    _attr_should_poll = False
    integration_domain = DOMAIN
    integration_name = "Gemini Live"
    turn_store_key = GEMINI_TURN_STORE_KEY
    supported_language_codes = SUPPORTED_LANGUAGES

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the TTS entity."""
        self.entry = entry
        config = {**entry.data, **entry.options}
        self._support_barge_in = bool(config.get(CONF_SUPPORT_BARGE_IN, False))
        self._attr_name = self.integration_name
        self._attr_unique_id = f"{entry.entry_id}_tts"

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return self.supported_language_codes

    @property
    def supported_options(self) -> list[str]:
        """Return supported options."""
        return [
            ATTR_PREFERRED_FORMAT,
            ATTR_PREFERRED_SAMPLE_RATE,
            ATTR_PREFERRED_SAMPLE_CHANNELS,
            ATTR_PREFERRED_SAMPLE_BYTES,
        ]

    @property
    def supports_audio_interrupt(self) -> bool:
        """Use Core's uncached, single-consumer stream when barge-in is enabled."""
        return self._support_barge_in

    @property
    def default_options(self) -> dict[str, Any]:
        """Use the native WAV format for interruptible playback."""
        return {ATTR_PREFERRED_FORMAT: "wav"} if self._support_barge_in else {}

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType:
        """Retrieve the cached audio response from hass.data."""
        entry_data = self.hass.data[self.integration_domain][self.entry.entry_id]
        audio = entry_data[self.turn_store_key].take_audio(message)
        _LOGGER.warning(
            "TTS: async_get_tts_audio called | message=%r | matched_audio=%s",
            message[:80] if message else "(none)",
            len(audio) if isinstance(audio, bytes) else type(audio).__name__,
        )
        if isinstance(audio, bytes):
            return "wav", audio

        _LOGGER.debug(
            "TTS: No cached audio found; returning silence so the pipeline can finish."
        )

        return "wav", self._get_dummy_wav()

    async def async_stream_tts_audio(
        self,
        request: TTSAudioRequest,
    ) -> TTSAudioResponse:
        """Stream the live model's native audio response as it arrives."""
        if self._support_barge_in:
            native_options = {
                ATTR_PREFERRED_FORMAT: "wav",
                ATTR_PREFERRED_SAMPLE_RATE: 16000,
                ATTR_PREFERRED_SAMPLE_CHANNELS: 1,
                ATTR_PREFERRED_SAMPLE_BYTES: 2,
            }
            for option, native_value in native_options.items():
                value = request.options.get(option, native_value)
                if str(value) != str(native_value):
                    raise HomeAssistantError(
                        "Barge-in requires 16 kHz, mono, 16-bit PCM WAV playback"
                    )
        entry_data = self.hass.data[self.integration_domain][self.entry.entry_id]
        turn_store = entry_data[self.turn_store_key]
        try:
            initial_text = await anext(request.message_gen)
        except StopAsyncIteration:
            initial_text = ""

        audio = turn_store.take_streaming_audio(initial_text)
        message = initial_text

        if audio is None:
            message += "".join([chunk async for chunk in request.message_gen])
            audio = turn_store.take_audio(message)

        async def data_gen() -> AsyncGenerator[bytes]:
            if isinstance(audio, AudioStream):

                async def drain_message_stream() -> None:
                    async for _chunk in request.message_gen:
                        pass

                drain_task = asyncio.create_task(drain_message_stream())
                unsubscribe_interrupt = None
                if (
                    self._support_barge_in
                    and (
                        on_audio_interrupt := getattr(
                            request, "on_audio_interrupt", None
                        )
                    )
                    is not None
                ):
                    unsubscribe_interrupt = audio.subscribe_interrupt(
                        on_audio_interrupt
                    )
                try:
                    yield streaming_wav_header()
                    async with aclosing(audio.async_chunks()) as chunks:
                        async for chunk in chunks:
                            yield chunk
                finally:
                    if unsubscribe_interrupt is not None:
                        unsubscribe_interrupt()
                    drain_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await drain_task
                return
            if audio:
                yield audio
                return
            yield self._get_dummy_wav()

        _LOGGER.warning(
            "TTS: async_stream_tts_audio called | message=%r | streaming=%s",
            message[:80] if message else "(none)",
            isinstance(audio, AudioStream),
        )
        return TTSAudioResponse("wav", data_gen())

    def _get_dummy_wav(self) -> bytes:
        """Return 1 second of silence as 16kHz mono 16-bit PCM WAV."""
        # 16000 samples/sec * 2 bytes/sample * 1 sec = 32000 bytes of zero
        pcm_data = b"\x00" * 32000
        return pcm_to_wav(pcm_data, 16000)


class GPTRealtimeTTS(GeminiLiveTTS):
    """Play native audio produced by the GPT Realtime turn."""

    integration_name = "GPT Realtime"
