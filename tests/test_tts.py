"""Tests for the Gemini Live TTS platform."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.gemini_live.const import (
    CONF_SUPPORT_BARGE_IN,
    DOMAIN,
    GEMINI_TURN_STORE_KEY,
)
from custom_components.gemini_live.runtime import AudioStream
from custom_components.gemini_live.tts import GeminiLiveTTS
from custom_components.gemini_live.utils import streaming_wav_header


@dataclass
class _InterruptResponse:
    """Represent the response type provided by Core with interrupt support."""

    extension: str
    data_gen: AsyncGenerator[bytes]


@dataclass
class _LegacyResponse:
    """Represent the response type provided by older Core versions."""

    extension: str
    data_gen: AsyncGenerator[bytes]


class _TurnStore:
    def __init__(self, audio: AudioStream) -> None:
        self.audio = audio

    def take_streaming_audio(self, _message: str) -> AudioStream:
        return self.audio


async def _message_gen() -> AsyncGenerator[str]:
    yield "-- gemini live --"


@pytest.fixture(autouse=True)
def _core_supports_interruption(monkeypatch: pytest.MonkeyPatch):
    """Pretend Core provides the complete TTS interruption path."""
    monkeypatch.setattr(
        "custom_components.gemini_live.tts.supports_tts_interruption",
        lambda: True,
    )


def _make_tts(*, barge_in: bool, audio: AudioStream) -> GeminiLiveTTS:
    entry = SimpleNamespace(
        entry_id="test-entry",
        data={},
        options={CONF_SUPPORT_BARGE_IN: barge_in},
    )
    entity = GeminiLiveTTS(entry)
    entity.hass = SimpleNamespace(
        data={DOMAIN: {entry.entry_id: {GEMINI_TURN_STORE_KEY: _TurnStore(audio)}}}
    )
    return entity


@pytest.mark.asyncio
@pytest.mark.parametrize("barge_in", [False, True])
async def test_core_interrupt_support_is_only_used_for_barge_in(
    monkeypatch: pytest.MonkeyPatch,
    barge_in: bool,
) -> None:
    """Use Core's optional interruption API only when barge-in is enabled."""
    monkeypatch.setattr(
        "custom_components.gemini_live.tts.TTSAudioResponse",
        _InterruptResponse,
    )
    audio = AudioStream()
    on_audio_interrupt = Mock()
    entity = _make_tts(barge_in=barge_in, audio=audio)
    request = SimpleNamespace(
        message_gen=_message_gen(),
        on_audio_interrupt=on_audio_interrupt,
        options={},
    )

    response = await entity.async_stream_tts_audio(request)
    assert entity.supports_audio_interrupt is barge_in

    assert await anext(response.data_gen) == streaming_wav_header()
    audio.interrupt()
    assert on_audio_interrupt.call_count == int(barge_in)
    audio.finish()
    await response.data_gen.aclose()


@pytest.mark.asyncio
async def test_barge_in_remains_compatible_with_legacy_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep streaming with barge-in disabled when Core has no interruption API."""
    monkeypatch.setattr(
        "custom_components.gemini_live.tts.TTSAudioResponse", _LegacyResponse
    )
    audio = AudioStream()
    entity = _make_tts(barge_in=False, audio=audio)
    request = SimpleNamespace(message_gen=_message_gen())

    response = await entity.async_stream_tts_audio(request)
    await anext(response.data_gen)
    audio.interrupt()
    audio.finish()
    await response.data_gen.aclose()


@pytest.mark.asyncio
async def test_interruptible_tts_rejects_non_native_audio() -> None:
    """Do not label native 16 kHz audio as a satellite's different requested rate."""
    entity = _make_tts(barge_in=True, audio=AudioStream())
    request = SimpleNamespace(options={"preferred_sample_rate": 48000})
    with pytest.raises(HomeAssistantError, match="16 kHz"):
        await entity.async_stream_tts_audio(request)


@pytest.mark.asyncio
async def test_closing_tts_cancels_live_audio_and_message_drain() -> None:
    """Closing playback must not wait for the remaining conversation text."""
    cancelled = Mock()
    audio = AudioStream(on_cancel=cancelled)
    await audio.add_chunk(b"audio")
    entity = _make_tts(barge_in=True, audio=audio)

    async def message() -> AsyncGenerator[str]:
        yield "-- gemini live --"
        await asyncio.Event().wait()

    request = SimpleNamespace(
        message_gen=message(), options={}, on_audio_interrupt=Mock()
    )
    response = await entity.async_stream_tts_audio(request)
    await anext(response.data_gen)
    assert await anext(response.data_gen) == b"audio"
    async with asyncio.timeout(1):
        await response.data_gen.aclose()
    cancelled.assert_called_once()
    audio.interrupt()
    request.on_audio_interrupt.assert_not_called()
