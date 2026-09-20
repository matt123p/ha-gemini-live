"""Constants for the Gemini Live integration."""

DOMAIN = "gemini_live"

CONF_API_KEY = "api_key"
CONF_PROVIDER = "provider"
CONF_MODEL = "model"
CONF_VOICE = "voice"
CONF_SYSTEM_INSTRUCTION = "system_instruction"
CONF_DETAILED_LOGGING = "detailed_logging"
CONF_TRANSCRIBE_GEMINI = "transcribe_gemini"
CONF_TRANSCRIBE_GPT = "transcribe_gpt"
CONF_ENCOURAGE_WEB_SEARCH = "encourage_web_search"
CONF_SEARCH_GROUNDING = "search_grounding"
CONF_THINKING_LEVEL = "thinking_level"
CONF_SHOW_TEXT = "show_text"
CONF_SUPPORT_BARGE_IN = "support_barge_in"
CONF_AFFECTIVE_DIALOG = "affective_dialog"

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"
DEFAULT_TRANSCRIBE_GEMINI = False
DEFAULT_TRANSCRIBE_GPT = False
DEFAULT_ENCOURAGE_WEB_SEARCH = False
DEFAULT_SEARCH_GROUNDING = False
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_SHOW_TEXT = True
DEFAULT_SUPPORT_BARGE_IN = False
DEFAULT_AFFECTIVE_DIALOG = False
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"
PROVIDER_PERSONAPLEX = "personaplex"
GEMINI_LIVE_TTS_PLACEHOLDER = "-- gemini live --"
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful, concise voice assistant for the user's smart home, powered by Home Assistant. "
    "Use the available tools to control devices, check states, run scripts, and query sensors. "
    "Always call the appropriate tool when the user asks to control something or queries a device state. "
    "Keep responses short, friendly, and natural for voice synthesis. "
    "Avoid formatting like bullet points, lists, bolding, or markdown in your speech."
)
OPENAI_SYSTEM_INSTRUCTION = (
    f"{DEFAULT_SYSTEM_INSTRUCTION}\n\n"
    "Aim for a balanced, natural voice response: answer the user's question fully "
    "enough to be useful, then stop. Lead with the direct result or confirmation. "
    "Include necessary context, clarification, or a brief follow-up question, but "
    "avoid unnecessary background, repetition, and conversational filler. Give "
    "more detail when the user asks for it or when the task genuinely needs it."
)

AVAILABLE_MODELS = [
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
]

EXTENDED_THINKING_MODEL = "gemini-3.8-live-extended-thinking"
THINKING_LEVELS = ["low", "medium", "high"]


def supports_thinking_level(model: str | None) -> bool:
    """Return whether a model exposes configurable background reasoning."""
    return model == EXTENDED_THINKING_MODEL

# Model generations that expose the affective-dialog setting. Affective
# dialog lets the model read the tone and emotion in the user's voice and
# adapt its own speaking style to match.
AFFECTIVE_DIALOG_MODELS = {"gemini-3.8-live"}


def supports_affective_dialog(model: str | None) -> bool:
    """Return whether a model supports the affective-dialog setting."""
    if not model:
        return False
    return model in AFFECTIVE_DIALOG_MODELS

OPENAI_DEFAULT_MODEL = "gpt-realtime-2.1"
OPENAI_DEFAULT_VOICE = "marin"
OPENAI_AVAILABLE_MODELS = [
    "gpt-realtime-2.1",
    "gpt-realtime-2.1-mini",
    "gpt-realtime-2",
    "gpt-realtime-1.5",
]
OPENAI_AVAILABLE_VOICES_INFO: list[tuple[str, str]] = [
    ("marin", "Natural and expressive; recommended"),
    ("cedar", "Natural and expressive; recommended"),
    ("alloy", "Neutral and balanced"),
    ("ash", "Clear and conversational"),
    ("ballad", "Warm and expressive"),
    ("coral", "Friendly and bright"),
    ("echo", "Smooth and measured"),
    ("sage", "Calm and composed"),
    ("shimmer", "Bright and energetic"),
    ("verse", "Versatile and natural"),
]

PERSONAPLEX_DEFAULT_MODEL = "fal-ai/personaplex/realtime"
PERSONAPLEX_AVAILABLE_MODELS = [PERSONAPLEX_DEFAULT_MODEL]
PERSONAPLEX_DEFAULT_VOICE = "NATF2"
PERSONAPLEX_AVAILABLE_VOICES_INFO: list[tuple[str, str]] = [
    *( (f"NATF{i}", "Natural female") for i in range(4) ),
    *( (f"NATM{i}", "Natural male") for i in range(4) ),
    *( (f"VARF{i}", "Variety female") for i in range(5) ),
    *( (f"VARM{i}", "Variety male") for i in range(5) ),
]

# Languages supported by Gemini native audio models.
# Source: https://ai.google.dev/gemini-api/docs/speech-generation#supported-languages
# en-US is retained as a Home Assistant pipeline compatibility alias for English.
SUPPORTED_LANGUAGES = [
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "ca",
    "ceb",
    "cmn",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "en-US",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "gl",
    "gu",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kn",
    "ko",
    "kok",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mai",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "nb",
    "ne",
    "nl",
    "nn",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
]

# Full list of 30 prebuilt voices available for Gemini Live / TTS.
# Source: https://ai.google.dev/gemini-api/docs/speech-generation
# Format: (voice_name, gender, description)
AVAILABLE_VOICES_INFO: list[tuple[str, str, str]] = [
    # --- Female voices ---
    ("Zephyr",          "female", "Bright and clear"),
    ("Kore",            "female", "Strong and firm"),
    ("Leda",            "female", "Youthful and energetic"),
    ("Aoede",           "female", "Relaxed and natural"),
    ("Callirhoe",       "female", "Friendly and easy-going"),
    ("Autonoe",         "female", "Bright and cheerful"),
    ("Despina",         "female", "Smooth and gentle"),
    ("Erinome",         "female", "Clear and articulate"),
    ("Laomedeia",       "female", "Positive and upbeat"),
    ("Achernar",        "female", "Soft and warm"),
    ("Gacrux",          "female", "Mature and steady"),
    ("Vindemiatrix",    "female", "Gentle and delicate"),
    ("Sulafat",         "female", "Warm and approachable"),
    # --- Male voices ---
    ("Puck",            "male",   "Upbeat and lively"),
    ("Charon",          "male",   "Calm and professional"),
    ("Fenrir",          "male",   "Passionate and energetic"),
    ("Orus",            "male",   "Calm and firm"),
    ("Enceladus",       "male",   "Soft and breathy"),
    ("Iapetus",         "male",   "Clear and clean"),
    ("Umbriel",         "male",   "Relaxed and easy-going"),
    ("Algieba",         "male",   "Smooth and flowing"),
    ("Algenib",         "male",   "Gravelly and textured"),
    ("Rasalgethi",      "male",   "Professional narrator"),
    ("Alnilam",         "male",   "Confident and firm"),
    ("Schedar",         "male",   "Even and steady"),
    ("Pulcherrima",     "male",   "Forward and enterprising"),
    ("Achird",          "male",   "Friendly and kind"),
    ("Zubenelgenubi",   "male",   "Casual and relaxed"),
    ("Sadachbia",       "male",   "Lively and vivid"),
    ("Sadaltager",      "male",   "Knowledgeable and learned"),
]

# Flat list of voice names for selectors (deduplicated, preserving order)
AVAILABLE_VOICES: list[str] = list(dict.fromkeys(name for name, _, _ in AVAILABLE_VOICES_INFO))

# Runtime objects stored under hass.data[DOMAIN][config_entry_id].
GEMINI_SESSION_MANAGER_KEY = "session_manager"
GEMINI_TURN_STORE_KEY = "turn_store"
