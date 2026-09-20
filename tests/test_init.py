"""Tests for Gemini Live integration setup."""

from types import SimpleNamespace

import pytest
from homeassistant.exceptions import ConfigEntryError

from custom_components.gemini_live import async_setup_entry
from custom_components.gemini_live.const import CONF_SUPPORT_BARGE_IN, DOMAIN


@pytest.mark.asyncio
async def test_setup_rejects_barge_in_without_core_interrupt_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject stale or manually edited barge-in configuration at setup."""
    monkeypatch.setattr(
        "custom_components.gemini_live.supports_tts_interruption", lambda: False
    )
    hass = SimpleNamespace(data={})
    entry = SimpleNamespace(data={CONF_SUPPORT_BARGE_IN: True}, options={})

    with pytest.raises(ConfigEntryError, match="TTS interruption support"):
        await async_setup_entry(hass, entry)

    assert DOMAIN not in hass.data
