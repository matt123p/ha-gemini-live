# Changelog

All notable changes to Gemini Live for Home Assistant are documented here.

## 1.0.4

- Added a user-visible response when the configured Google AI project exceeds
  its monthly spending cap, instead of failing the Assist pipeline silently.
- Added a persistent Home Assistant Repairs issue with a link to Google AI
  Studio. The issue clears automatically after Gemini Live reconnects.
- Removed obsolete instructions for patching Home Assistant Core now that the
  low-latency change is included upstream.

## 1.0.3

- Added an optional `show_text` tool so Gemini can display formatted text when
  response transcription is disabled.
- Kept conversation completion state isolated to the specific conversation
  that invoked `end_conversation`.
- Expanded setup and troubleshooting documentation.

## 1.0.2

- Added an `end_conversation` callback that lets Gemini tell Home Assistant when
  to stop listening for follow-up requests. Completion state is tracked
  independently for each conversation.
- Made short opening commands such as "stop" prioritize stopping an actively
  ringing alarm or timer before ending the conversation.
- Documented the Home Assistant Core custom-component override that reduces
  response latency on ESPHome Assist satellites.

## 1.0.1

- Fixed HACS and Hassfest validation metadata.

## 1.0.0

- Added Gemini Live speech-to-text, conversation, and cached native-audio
  text-to-speech entities.
- Added HACS metadata, brand assets, translations, validation workflow, and
  installation documentation.
