# Remote Satellite Barge-In

This document describes the end-to-end barge-in path used by
`ha-gemini-live`, the required Home Assistant Core changes, and the behavior
expected from Wyoming and ESPHome voice satellites.

Barge-in lets a user speak while the assistant is playing a response. The old
response must stop promptly, microphone audio must continue reaching the live
model, and replacement audio must play without starting a new conversation
session.

## Requirements

Barge-in requires all of the following:

- the **Support barge-in** option enabled in `ha-gemini-live`;
- Home Assistant Core with the TTS interruption path;
- a full-duplex satellite that continues microphone capture during playback;
- effective acoustic echo cancellation (AEC), or headphones; and
- satellite playback that reacts correctly to an interrupted audio segment.

The integration checks for the required Core API before allowing barge-in to be
enabled. Specifically, Core must provide:

- `TTSAudioRequest.on_audio_interrupt`; and
- `TextToSpeechEntity.supports_audio_interrupt`.

If either API is missing, the config or options flow rejects barge-in. With
barge-in disabled, the integration continues to work with an unmodified Home
Assistant Core.

## End-to-end interruption path

The implemented path is:

```text
User speaks during assistant playback
    -> satellite continues sending microphone PCM
    -> Gemini/OpenAI detects speech and interrupts generation
    -> ha-gemini-live discards queued model audio
    -> Gemini Live TTS invokes Core's on_audio_interrupt callback
    -> Core forwards the callback directly to the consuming satellite
    -> Wyoming: AudioStop followed by AudioStart
       ESPHome: TTS_STREAM_END followed by TTS_STREAM_START
    -> satellite purges old playback
    -> replacement model audio continues in the same TTS response
```

The provider is authoritative about whether an interruption occurred. Local
speech detection on the satellite is still useful for stopping its speaker with
the lowest possible latency, but it does not replace the provider-to-Core
interruption path.

## Responsibilities

### Gemini Live integration and provider

The integration and live-model provider:

- keep forwarding microphone PCM while the model is speaking;
- use provider-side voice activity detection to recognize an interruption;
- cancel or abandon the interrupted model response;
- discard assistant audio still queued inside the integration;
- notify Home Assistant through the TTS interruption callback;
- stream replacement audio through the existing TTS response; and
- keep the live-model session available so conversational context is retained.

The response-transcription option is independent of barge-in. Enabling
barge-in must not implicitly enable or display the assistant transcript.

### Home Assistant Core

The Core patch:

- lets an entity opt in through `supports_audio_interrupt`;
- bypasses memory and disk caches entirely for that entity's responses;
- starts generation when a single interrupt-capable satellite consumes the stream;
- passes the satellite's `on_audio_interrupt` callback directly to the engine; and
- converts that notification into the native satellite transport boundaries
  described below.

Interruptible streams require native output: this integration supplies 16 kHz,
mono, 16-bit PCM WAV. Core rejects unsupported output options or an extension
mismatch instead of routing the stream through ffmpeg. Buffered consumers such
as VoIP, HTTP/media-player playback, and repeated consumers are not supported
with barge-in enabled. Ordinary TTS caching and conversion remain unchanged
when barge-in is disabled.

ESPHome speaker playback starts at `INTENT_PROGRESS` for interruptible streams,
using the token supplied at `RUN_START`. This drains live audio while the
conversation is still producing text and avoids waiting until `TTS_END`.

### Remote satellite

The satellite must:

- capture and transmit microphone audio while speaker audio is playing;
- run AEC so assistant audio is not mistaken for user speech;
- stop and purge old playback when the interruption boundary arrives;
- accept the new stream-start boundary and replacement audio;
- keep the overall Assist pipeline alive across the interruption; and
- report playback completion only for the final response segment.

## Microphone and echo cancellation

Capture and playback must run in separate tasks or threads. Receiving assistant
audio must never pause or close microphone capture.

Send 16 kHz, 16-bit, mono PCM for the complete pipeline run. A microphone
packet duration of roughly 20–40 ms is recommended; larger packets add directly
to interruption latency.

Without effective AEC, provider VAD can hear the assistant's speaker output and
cause repeated self-interruptions. Feed the exact rendered speaker samples into
the AEC reference path and account for hardware, operating-system, decoder, and
speaker buffering. Noise suppression and automatic gain control can help after
AEC, but do not replace it.

For hardware without reliable AEC, use headphones, a push-to-interrupt control,
or leave barge-in disabled.

## Playback behavior

When local speech is detected during playback, a capable satellite may stop and
purge playback immediately rather than waiting for the provider round trip. It
must continue transmitting microphone audio.

When the Core interruption boundary arrives, the satellite must:

1. stop the audio device;
2. clear device/DMA, decoder, jitter, and application playback queues;
3. reset timestamps and decoder state for the new segment;
4. continue microphone transmission; and
5. accept replacement audio after the new start event.

The underlying Home Assistant TTS result remains open. The stop/start events
divide that result into playback segments; they do not end the Assist pipeline.

These are standard transport events used in a new mid-response pattern, not new
protocol event types. Existing satellite firmware that treats every stop/end as
the end of the entire Assist response must be updated or configured to recognize
the immediately following start event.

## Wyoming satellites

For Wyoming, an interruption is represented as:

```text
AudioStop(timestamp=<old segment duration>)
AudioStart(rate=<same rate>, width=<same width>, channels=<same channels>, timestamp=0)
AudioChunk(... replacement audio ...)
```

Core sends `AudioStop` and `AudioStart` as one locked sequence, so an
`AudioChunk` cannot be interleaved between the two boundary events.

A compatible Wyoming satellite must:

- purge the old playback queue on the intermediate `AudioStop`;
- treat the immediately following `AudioStart` as a new segment of the same
  Assist response;
- reset its playback timestamp to zero for the new segment;
- continue sending microphone `AudioChunk` events throughout; and
- avoid sending `Played` for an interrupted segment.

Send `Played` only after the final `AudioStop` has actually completed playback.
An early `Played` makes Home Assistant mark the entire TTS response as finished,
even though replacement audio is still coming.

Because Wyoming does not yet carry a distinct interruption event or response
identifier, the back-to-back `AudioStop`/`AudioStart` pair is the interruption
marker. A client that normally sends `Played` immediately after every
`AudioStop` needs a short look-ahead/grace period so it can recognize the
following `AudioStart` and suppress the intermediate completion.

## ESPHome voice assistants

For ESPHome, Core sends:

```text
VOICE_ASSISTANT_TTS_STREAM_END
VOICE_ASSISTANT_TTS_STREAM_START
<replacement TTS audio>
```

Core also resets its stream timing and duration accounting when the new segment
starts.

The ESPHome voice assistant must purge buffered audio on the intermediate
`TTS_STREAM_END`, accept the following `TTS_STREAM_START`, and continue
microphone capture. It must not report the overall announcement or Assist
response as finished for the intermediate end event; completion belongs to the
final stream end.

## Suggested satellite state machine

```text
LISTENING
    user utterance ends
        -> PROCESSING

PROCESSING
    first assistant audio/start event arrives
        -> SPEAKING

SPEAKING
    local user speech starts
        -> optionally purge playback immediately
        -> keep transmitting microphone audio
        -> LISTENING

SPEAKING or LISTENING
    Core interruption stop/end arrives
        -> purge old playback
    Core interruption start arrives
        -> prepare a clean replacement segment
        -> PROCESSING

PROCESSING
    replacement audio arrives
        -> SPEAKING

SPEAKING
    final stop/end and playback completion
        -> IDLE or LISTENING, according to pipeline continuation
```

The local state machine may react before Core, but the Core signal is the
authoritative boundary between old and replacement assistant audio.

## Buffering and latency

Suggested starting targets:

| Component | Target |
|---|---:|
| Microphone packet duration | 20–40 ms |
| Local playback queue | 40–100 ms |
| Speech-start debounce | 40–100 ms |
| Local playback purge after speech start | Under 50 ms |
| Integration response-audio buffer | About 200 ms |

The integration deliberately bounds its response-audio queue. On interruption,
it clears that queue, while Core clears its own active TTS consumer queues. The
satellite boundary then clears audio already delivered to the remote device.

Smaller buffers reduce audible overrun after interruption but increase
sensitivity to network jitter and scheduling delays. Measure latency on the
actual satellite hardware.

## Compatibility behavior

| Barge-in | Core interruption API | Result |
|---|---|---|
| Disabled | Missing or present | Normal legacy TTS behavior |
| Enabled | Present | Full interruption propagation to supported satellites |
| Enabled | Missing | Configuration is rejected and setup is blocked |

The current custom Core image and patch are release-specific. Do not copy the
patched Core files into a different Home Assistant version without rebasing and
testing the patch against that release.

## Failure handling

- On network loss, stop playback, close capture, and return to an error or idle
  state.
- If replacement audio does not arrive within the normal response timeout, end
  the interaction instead of remaining in `PROCESSING` indefinitely.
- If AEC diverges or speaker loopback repeatedly interrupts the model, disable
  automatic interruption and provide a push-to-interrupt fallback.
- If the satellite stops microphone transmission during playback, barge-in is
  unavailable for that turn; ordinary playback should still work.

## Validation checklist

Test with response transcription both enabled and disabled.

- Barge-in cannot be enabled on an unpatched Core.
- With barge-in disabled, behavior is unchanged on patched and unpatched Core.
- Microphone packets continue while assistant audio is playing.
- Speaker loopback alone does not trigger interruption.
- Real user speech interrupts the provider response.
- The integration discards its queued old-response audio.
- Core discards audio queued for active TTS consumers.
- Wyoming receives an intermediate `AudioStop` followed by `AudioStart`.
- ESPHome receives an intermediate `TTS_STREAM_END` followed by
  `TTS_STREAM_START`.
- The satellite purges old buffered playback at the interruption boundary.
- No intermediate Wyoming `Played` or ESPHome completion is reported.
- Replacement audio starts with reset timing and is not clipped.
- The final stop/end produces the normal playback-completed indication.
- Enabling barge-in does not enable response transcription.
- Repeated interruptions do not leak tasks, callbacks, buffers, or response
  state.

For an end-to-end test, request a long answer and interrupt it after several
words. Verify that old speech stops, microphone upload continues, the transport
emits the expected stop/start boundary, replacement speech plays, and the same
live-model session retains its conversation context.
