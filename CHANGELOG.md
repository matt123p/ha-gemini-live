# Changelog

All notable changes to Gemini Live for Home Assistant are documented here.

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
