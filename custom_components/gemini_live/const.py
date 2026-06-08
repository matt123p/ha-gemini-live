"""Constants for the Gemini Live integration."""

DOMAIN = "gemini_live"

CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_VOICE = "voice"
CONF_SYSTEM_INSTRUCTION = "system_instruction"
CONF_DETAILED_LOGGING = "detailed_logging"
CONF_TRANSCRIBE_GEMINI = "transcribe_gemini"
CONF_ENCOURAGE_WEB_SEARCH = "encourage_web_search"

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"
DEFAULT_TRANSCRIBE_GEMINI = False
DEFAULT_ENCOURAGE_WEB_SEARCH = False
GEMINI_LIVE_TTS_PLACEHOLDER = "-- gemini live --"
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful, concise voice assistant for the user's smart home, powered by Home Assistant. "
    "Use the available tools to control devices, check states, run scripts, and query sensors. "
    "Always call the appropriate tool when the user asks to control something or queries a device state. "
    "Keep responses short, friendly, and natural for voice synthesis. "
    "Avoid formatting like bullet points, lists, bolding, or markdown in your speech."
)

AVAILABLE_MODELS = [
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
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
