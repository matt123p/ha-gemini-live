"""Live voice-model integration for Gemini and OpenAI."""

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store

from . import conversation, stt, tts  # noqa: F401
from .const import (
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ELEVENLABS_AGENT_ID,
    CONF_PROVIDER,
    DOMAIN,
    ELEVENLABS_CACHE_KEY,
    ELEVENLABS_CACHE_SAVE_KEY,
    ELEVENLABS_CACHE_STORE_KEY,
    GEMINI_SESSION_MANAGER_KEY,
    GEMINI_TURN_STORE_KEY,
    PROVIDER_ELEVENLABS,
)
from .elevenlabs import (
    ElevenLabsAgentsError,
    agent_language,
    async_delete_agent,
    async_delete_tools,
    async_update_agent,
)
from .runtime import LiveSessionManager, TurnStore
from .utils import set_detailed_logging

_LOGGER = logging.getLogger(__name__)
_ELEVENLABS_STORE_VERSION = 1


def _elevenlabs_store(hass: HomeAssistant, entry_id: str) -> Store[dict]:
    """Return the store recording the ElevenLabs tools created for an entry."""
    return Store(hass, _ELEVENLABS_STORE_VERSION, f"{DOMAIN}.elevenlabs.{entry_id}")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a live voice-model provider from a config entry."""
    config = {**entry.data, **entry.options}
    hass.data.setdefault(DOMAIN, {})
    set_detailed_logging(bool(config.get(CONF_DETAILED_LOGGING, False)))

    # Store configuration data (merging data and options)
    hass.data[DOMAIN][entry.entry_id] = {
        **entry.data,
        **entry.options,
        GEMINI_SESSION_MANAGER_KEY: LiveSessionManager(),
        GEMINI_TURN_STORE_KEY: TurnStore(),
    }

    if config.get(CONF_PROVIDER) == PROVIDER_ELEVENLABS:
        store = _elevenlabs_store(hass, entry.entry_id)
        cache = await store.async_load() or {}
        hass.data[DOMAIN][entry.entry_id][ELEVENLABS_CACHE_KEY] = cache
        hass.data[DOMAIN][entry.entry_id][ELEVENLABS_CACHE_STORE_KEY] = store
        hass.data[DOMAIN][entry.entry_id][ELEVENLABS_CACHE_SAVE_KEY] = (
            lambda: store.async_delay_save(lambda: cache, 1)
        )
        # Bring the agent up to date with this version's base settings, so an
        # update needs no reconfigure. A failure only delays that until the
        # next start; conversations still work with the older settings.
        entry.async_create_background_task(
            hass,
            _async_refresh_agent(hass, config),
            f"{DOMAIN} refresh ElevenLabs agent",
        )

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
        if store := entry_data.get(ELEVENLABS_CACHE_STORE_KEY):
            # Write now, so a reload reads the latest tool IDs and no delayed
            # write recreates the file after the entry is removed.
            await store.async_save(entry_data[ELEVENLABS_CACHE_KEY])
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update by reloading the entry."""
    _LOGGER.debug("Live model entry updated, reloading")
    await hass.config_entries.async_reload(entry.entry_id)


async def _async_refresh_agent(hass: HomeAssistant, config: dict) -> None:
    """Reapply the integration's base settings to the entry's agent."""
    agent_id = config.get(CONF_ELEVENLABS_AGENT_ID)
    if not agent_id:
        return
    try:
        await async_update_agent(
            async_get_clientsession(hass),
            config.get(CONF_API_KEY, ""),
            agent_id,
            agent_language(hass.config.language),
        )
    except ElevenLabsAgentsError as exc:
        _LOGGER.warning("Could not update the ElevenLabs agent: %s", exc)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the ElevenLabs agent and tools created for a removed entry."""
    config = {**entry.data, **entry.options}
    if config.get(CONF_PROVIDER) != PROVIDER_ELEVENLABS:
        return
    store = _elevenlabs_store(hass, entry.entry_id)
    cache = await store.async_load() or {}
    http_session = async_get_clientsession(hass)
    api_key = config.get(CONF_API_KEY, "")
    try:
        await async_delete_tools(http_session, api_key, cache)
        if agent_id := config.get(CONF_ELEVENLABS_AGENT_ID):
            await async_delete_agent(http_session, api_key, agent_id)
    except ElevenLabsAgentsError as exc:
        _LOGGER.warning(
            "Could not delete the ElevenLabs agent and tools for %s; remove "
            "them in the ElevenLabs dashboard: %s",
            entry.title,
            exc,
        )
    await store.async_remove()
