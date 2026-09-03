"""Config flow for live voice-model providers."""

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    AVAILABLE_MODELS,
    AVAILABLE_VOICES_INFO,
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_SHOW_TEXT,
    CONF_SYSTEM_INSTRUCTION,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    CONF_VOICE,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_MODEL,
    DEFAULT_SHOW_TEXT,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_TRANSCRIBE_GPT,
    DEFAULT_VOICE,
    DOMAIN,
    OPENAI_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_VOICES_INFO,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_VOICE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
)

PROVIDER_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=PROVIDER_GEMINI, label="Google Gemini"),
            selector.SelectOptionDict(value=PROVIDER_OPENAI, label="OpenAI"),
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

GEMINI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(
                value=name,
                label=f"{name} - {gender}, {description}",
            )
            for name, gender, description in AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

OPENAI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in OPENAI_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _provider(config: dict[str, Any]) -> str:
    """Return the configured provider, defaulting legacy entries to Gemini."""
    return config.get(CONF_PROVIDER, PROVIDER_GEMINI)


def _provider_schema(provider: str, config: dict[str, Any] | None = None) -> vol.Schema:
    """Build a provider-specific setup/options schema."""
    current = config or {}
    is_openai = provider == PROVIDER_OPENAI
    models = OPENAI_AVAILABLE_MODELS if is_openai else AVAILABLE_MODELS
    default_model = OPENAI_DEFAULT_MODEL if is_openai else DEFAULT_MODEL
    default_voice = OPENAI_DEFAULT_VOICE if is_openai else DEFAULT_VOICE
    voice_selector = OPENAI_VOICE_SELECTOR if is_openai else GEMINI_VOICE_SELECTOR
    transcribe_key = CONF_TRANSCRIBE_GPT if is_openai else CONF_TRANSCRIBE_GEMINI
    default_transcribe = (
        DEFAULT_TRANSCRIBE_GPT if is_openai else DEFAULT_TRANSCRIBE_GEMINI
    )

    api_key_field = (
        vol.Required(CONF_API_KEY, default=current[CONF_API_KEY])
        if CONF_API_KEY in current
        else vol.Required(CONF_API_KEY)
    )
    fields: dict[vol.Marker, Any] = {
        api_key_field: str,
        vol.Required(
            CONF_MODEL,
            default=current.get(CONF_MODEL, default_model),
        ): vol.In(models),
        vol.Required(
            CONF_VOICE,
            default=current.get(CONF_VOICE, default_voice),
        ): voice_selector,
        vol.Optional(
            CONF_SYSTEM_INSTRUCTION,
            description={
                "suggested_value": current.get(CONF_SYSTEM_INSTRUCTION, "")
            },
        ): str,
        vol.Optional(
            CONF_DETAILED_LOGGING,
            default=current.get(CONF_DETAILED_LOGGING, False),
        ): selector.BooleanSelector(),
        vol.Optional(
            transcribe_key,
            default=current.get(transcribe_key, default_transcribe),
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_ENCOURAGE_WEB_SEARCH,
            default=current.get(
                CONF_ENCOURAGE_WEB_SEARCH,
                DEFAULT_ENCOURAGE_WEB_SEARCH,
            ),
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_SHOW_TEXT,
            default=current.get(CONF_SHOW_TEXT, DEFAULT_SHOW_TEXT),
        ): selector.BooleanSelector(),
    }
    return vol.Schema(fields)


class GeminiLiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure a Gemini Live or GPT Realtime provider."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Select the live model provider."""
        if user_input is not None:
            return await self.async_step_provider(
                provider=user_input[CONF_PROVIDER]
            )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER,
                        default=PROVIDER_GEMINI,
                    ): PROVIDER_SELECTOR
                }
            ),
        )

    async def async_step_provider(
        self,
        user_input=None,
        *,
        provider: str | None = None,
    ):
        """Collect settings for the selected provider."""
        selected_provider = provider or self.context[CONF_PROVIDER]
        self.context[CONF_PROVIDER] = selected_provider
        if user_input is not None:
            user_input[CONF_PROVIDER] = selected_provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            title = (
                "GPT Realtime"
                if selected_provider == PROVIDER_OPENAI
                else "Gemini Live"
            )
            return self.async_create_entry(title=title, data=user_input)
        return self.async_show_form(
            step_id="provider",
            data_schema=_provider_schema(selected_provider),
            description_placeholders={
                "provider": (
                    "OpenAI"
                    if selected_provider == PROVIDER_OPENAI
                    else "Google Gemini"
                )
            },
        )

    async def async_step_reconfigure(self, user_input=None):
        """Reconfigure an existing provider entry."""
        entry = self._get_reconfigure_entry()
        config = {**entry.data, **entry.options}
        provider = _provider(config)
        if user_input is not None:
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_update_reload_and_abort(
                entry,
                data_updates=user_input,
                options={},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_provider_schema(provider, config),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow."""
        return GeminiLiveOptionsFlowHandler()


class GeminiLiveOptionsFlowHandler(config_entries.OptionsFlow):
    """Manage live provider options."""

    async def async_step_init(self, user_input=None):
        """Update the provider's connection and response settings."""
        config = {**self.config_entry.data, **self.config_entry.options}
        provider = _provider(config)
        if user_input is not None:
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_create_entry(title="", data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=_provider_schema(provider, config),
        )
