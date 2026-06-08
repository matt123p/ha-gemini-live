"""The Gemini Live integration."""

import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_DETAILED_LOGGING,
    DOMAIN,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
)
from . import stt, tts, conversation
from .runtime import LiveSessionManager, TurnStore
from .utils import set_detailed_logging

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Gemini Live from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    config = {**entry.data, **entry.options}
    set_detailed_logging(bool(config.get(CONF_DETAILED_LOGGING, False)))

    # Store configuration data (merging data and options)
    hass.data[DOMAIN][entry.entry_id] = {
        **entry.data,
        **entry.options,
        GEMINI_SESSION_MANAGER_KEY: LiveSessionManager(),
        GEMINI_TURN_STORE_KEY: TurnStore(),
    }

    # Register options update listener to reload integration when changed
    entry.async_on_unload(entry.add_update_listener(update_listener))

    # Forward setup to the platforms
    await hass.config_entries.async_forward_entry_setups(entry, ["stt", "tts", "conversation"])
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, ["stt", "tts", "conversation"]
    )
    if unload_ok:
        entry_data = hass.data[DOMAIN][entry.entry_id]
        await entry_data[GEMINI_SESSION_MANAGER_KEY].async_close_all()
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update by reloading the entry."""
    _LOGGER.debug("Gemini Live entry updated, reloading")
    await hass.config_entries.async_reload(entry.entry_id)
