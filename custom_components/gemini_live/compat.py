"""Compatibility checks for optional Home Assistant Core APIs."""

from dataclasses import fields

from homeassistant.components.tts.entity import TTSAudioRequest, TTSAudioResponse


def supports_tts_interruption() -> bool:
    """Return whether Core provides the complete TTS interruption path."""
    request_fields = {field.name for field in fields(TTSAudioRequest)}
    response_fields = {field.name for field in fields(TTSAudioResponse)}
    return "on_audio_interrupt" in request_fields and "passthrough" in response_fields
