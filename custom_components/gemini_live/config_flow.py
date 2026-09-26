"""Config flow for live voice-model providers."""

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm, selector

from .compat import supports_tts_interruption
from .const import (
    AVAILABLE_MODELS,
    AVAILABLE_VOICES_INFO,
    CONF_AFFECTIVE_DIALOG,
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_SEARCH_GROUNDING,
    CONF_SHOW_TEXT,
    CONF_SUPPORT_BARGE_IN,
    CONF_SYSTEM_INSTRUCTION,
    CONF_THINKING_LEVEL,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_GROUNDING,
    DEFAULT_SHOW_TEXT,
    DEFAULT_SUPPORT_BARGE_IN,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_TRANSCRIBE_GPT,
    DEFAULT_VOICE,
    DOMAIN,
    OPENAI_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_VOICES_INFO,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_VOICE,
    PERSONAPLEX_AVAILABLE_MODELS,
    PERSONAPLEX_AVAILABLE_VOICES_INFO,
    PERSONAPLEX_DEFAULT_MODEL,
    PERSONAPLEX_DEFAULT_VOICE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
    THINKING_LEVELS,
    supports_affective_dialog,
    supports_thinking_level,
)

PROVIDER_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=PROVIDER_GEMINI, label="Google Gemini"),
            selector.SelectOptionDict(value=PROVIDER_OPENAI, label="OpenAI"),
            selector.SelectOptionDict(
                value=PROVIDER_PERSONAPLEX, label="fal.ai PersonaPlex"
            ),
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _model_selector(models: list[str]) -> selector.SelectSelector:
    """Return a compact dropdown for a provider's model choices."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=models,
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

PERSONAPLEX_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in PERSONAPLEX_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _provider(config: dict[str, Any]) -> str:
    """Return the configured provider, defaulting legacy entries to Gemini."""
    return config.get(CONF_PROVIDER, PROVIDER_GEMINI)


def _provider_schema(
    provider: str,
    config: dict[str, Any] | None = None,
    hass: HomeAssistant | None = None,
) -> vol.Schema:
    """Build a provider-specific setup/options schema."""
    current = config or {}
    is_openai = provider == PROVIDER_OPENAI
    is_personaplex = provider == PROVIDER_PERSONAPLEX
    models = (
        PERSONAPLEX_AVAILABLE_MODELS
        if is_personaplex
        else OPENAI_AVAILABLE_MODELS
        if is_openai
        else AVAILABLE_MODELS
    )
    default_model = (
        PERSONAPLEX_DEFAULT_MODEL
        if is_personaplex
        else OPENAI_DEFAULT_MODEL
        if is_openai
        else DEFAULT_MODEL
    )
    default_voice = (
        PERSONAPLEX_DEFAULT_VOICE
        if is_personaplex
        else OPENAI_DEFAULT_VOICE
        if is_openai
        else DEFAULT_VOICE
    )
    voice_selector = (
        PERSONAPLEX_VOICE_SELECTOR
        if is_personaplex
        else OPENAI_VOICE_SELECTOR
        if is_openai
        else GEMINI_VOICE_SELECTOR
    )
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
        ): _model_selector(models),
        vol.Required(
            CONF_VOICE,
            default=current.get(CONF_VOICE, default_voice),
        ): voice_selector,
        vol.Optional(
            CONF_SYSTEM_INSTRUCTION,
            description={"suggested_value": current.get(CONF_SYSTEM_INSTRUCTION, "")},
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
            CONF_SHOW_TEXT,
            default=current.get(CONF_SHOW_TEXT, DEFAULT_SHOW_TEXT),
        ): selector.BooleanSelector(),
    }
    if hass is not None and not is_personaplex:
        apis = [
            selector.SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(hass)
        ]
        known_api_ids = {api["value"] for api in apis}
        selected_apis = current.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])
        if isinstance(selected_apis, str):
            selected_apis = [selected_apis]
        selected_apis = [api_id for api_id in selected_apis if api_id in known_api_ids]
        fields[
            vol.Optional(
                CONF_LLM_HASS_API,
                default=selected_apis,
                description={"suggested_value": selected_apis},
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(options=apis, multiple=True)
        )
    if not is_openai and not is_personaplex:
        # All Gemini Live models offered by this integration support Google's
        # server-side Search grounding tool. It is a separate, opt-in
        # preference from the former Assist search-tool prompt hint.
        fields[
            vol.Optional(
                CONF_SEARCH_GROUNDING,
                default=current.get(
                    CONF_SEARCH_GROUNDING,
                    DEFAULT_SEARCH_GROUNDING,
                ),
            )
        ] = selector.BooleanSelector()
    elif is_openai:
        fields[
            vol.Optional(
                CONF_ENCOURAGE_WEB_SEARCH,
                default=current.get(
                    CONF_ENCOURAGE_WEB_SEARCH,
                    DEFAULT_ENCOURAGE_WEB_SEARCH,
                ),
            )
        ] = selector.BooleanSelector()
    if not is_personaplex and supports_tts_interruption():
        # Barge-in needs Core's complete TTS interruption path. When the
        # installed Core cannot propagate interruptions, the setting cannot do
        # anything, so it is hidden entirely and never blocks setup.
        fields[
            vol.Optional(
                CONF_SUPPORT_BARGE_IN,
                default=current.get(
                    CONF_SUPPORT_BARGE_IN,
                    DEFAULT_SUPPORT_BARGE_IN,
                ),
            )
        ] = selector.BooleanSelector()
    # The affective-dialog switch is Gemini-specific and only rendered for
    # models that support it; Home Assistant config forms cannot render a
    # disabled field, so it is hidden instead.
    if (
        not is_openai
        and not is_personaplex
        and supports_affective_dialog(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_AFFECTIVE_DIALOG,
                default=current.get(
                    CONF_AFFECTIVE_DIALOG,
                    DEFAULT_AFFECTIVE_DIALOG,
                ),
            )
        ] = selector.BooleanSelector()
    if (
        not is_openai
        and not is_personaplex
        and supports_thinking_level(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_THINKING_LEVEL,
                default=current.get(
                    CONF_THINKING_LEVEL,
                    DEFAULT_THINKING_LEVEL,
                ),
            )
        ] = _model_selector(THINKING_LEVELS)
    if is_personaplex:
        # PersonaPlex currently has no function-calling, display-text tool, or
        # explicit VAD event surface in its public realtime API.
        marker = next(marker for marker in fields if marker.schema == CONF_SHOW_TEXT)
        del fields[marker]
    return vol.Schema(fields)


def _strip_unsupported_settings(user_input: dict[str, Any]) -> dict[str, Any]:
    """Drop settings the selected model or Core does not support."""
    if not supports_affective_dialog(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_AFFECTIVE_DIALOG, None)
    if not supports_thinking_level(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_THINKING_LEVEL, None)
    if not supports_tts_interruption():
        user_input.pop(CONF_SUPPORT_BARGE_IN, None)
    return user_input


def _needs_model_specific_refresh(user_input: dict[str, Any]) -> bool:
    """Return whether a newly selected model needs its settings form shown."""
    model = user_input.get(CONF_MODEL)
    return (
        supports_thinking_level(model) and CONF_THINKING_LEVEL not in user_input
    ) or (supports_affective_dialog(model) and CONF_AFFECTIVE_DIALOG not in user_input)


class GeminiLiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure a Gemini Live or GPT Realtime provider."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Select the live model provider."""
        if user_input is not None:
            return await self.async_step_provider(provider=user_input[CONF_PROVIDER])
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
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="provider",
                    data_schema=_provider_schema(
                        selected_provider, user_input, self.hass
                    ),
                    description_placeholders={
                        "provider": "Google Gemini",
                    },
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = selected_provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            title = {
                PROVIDER_OPENAI: "GPT Realtime",
                PROVIDER_PERSONAPLEX: "PersonaPlex",
            }.get(selected_provider, "Gemini Live")
            return self.async_create_entry(title=title, data=user_input)
        return self.async_show_form(
            step_id="provider",
            data_schema=_provider_schema(selected_provider, hass=self.hass),
            description_placeholders={
                "provider": {
                    PROVIDER_OPENAI: "OpenAI",
                    PROVIDER_PERSONAPLEX: "fal.ai PersonaPlex",
                }.get(selected_provider, "Google Gemini")
            },
        )

    async def async_step_reconfigure(self, user_input=None):
        """Reconfigure an existing provider entry."""
        entry = self._get_reconfigure_entry()
        config = {**entry.data, **entry.options}
        provider = _provider(config)
        if user_input is not None:
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="reconfigure",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_update_reload_and_abort(
                entry,
                data_updates=user_input,
                options={},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_provider_schema(provider, config, self.hass),
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
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_create_entry(title="", data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=_provider_schema(provider, config, self.hass),
        )
