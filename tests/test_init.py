"""Tests for Gemini Live integration setup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from custom_components.gemini_live import async_setup_entry
from custom_components.gemini_live.const import CONF_SUPPORT_BARGE_IN, DOMAIN


async def test_setup_succeeds_with_barge_in_without_core_interrupt_support() -> None:
    """Stale barge-in configuration no longer blocks setup; it is just inert."""
    hass = SimpleNamespace(
        data={},
        config_entries=SimpleNamespace(
            async_forward_entry_setups=AsyncMock(return_value=None)
        ),
    )
    entry = SimpleNamespace(
        data={CONF_SUPPORT_BARGE_IN: True},
        options={},
        entry_id="test-entry",
        async_on_unload=lambda _: None,
        add_update_listener=lambda _listener: lambda: None,
    )

    assert await async_setup_entry(hass, entry) is True
    assert DOMAIN in hass.data
    assert hass.data[DOMAIN]["test-entry"][CONF_SUPPORT_BARGE_IN] is True
