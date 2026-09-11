# Implementing Barge-In on a Remote Voice Satellite

This document describes the remote-device behavior required to use the
optional barge-in mode in `ha-gemini-live`.

Barge-in is more than keeping the microphone enabled. A good implementation
must capture microphone audio while assistant audio is playing, prevent the
speaker signal from being mistaken for user speech, stop old playback quickly,
and keep the display state synchronized with what the user is doing.

## Responsibilities

The integration and live-model provider handle:

- continuously receiving microphone PCM while the model speaks;
- detecting user activity with provider-side VAD;
- cancelling the interrupted model response;
- discarding assistant audio still queued inside Home Assistant; and
- streaming replacement response audio through the existing TTS stream.

The remote satellite must handle:

- simultaneous microphone upload and speaker playback;
- acoustic echo cancellation (AEC);
- immediate local playback cancellation when the user starts speaking;
- discarding locally buffered or in-flight audio from the old response; and
- its local listening, processing, and speaking display states.

The response-transcription option is independent of barge-in. Do not require or
display the assistant's response transcript merely because barge-in is enabled.
The user's input transcript is still expected and may be sent to Home Assistant
as the STT result.

## Required audio behavior

### Keep microphone capture active

Do not pause or close microphone capture when an `AudioStart`, audio chunk, TTS,
or speaking event is received. Continue sending 16 kHz, 16-bit, mono PCM to
Home Assistant for the complete pipeline run.

Capture and playback must run in separate tasks or threads. Neither direction
may block the other.

Recommended microphone packet duration is 20–40 ms. Large packets add directly
to interruption latency.

### Use acoustic echo cancellation

Without AEC, provider VAD may interpret the assistant's own speaker output as a
new user utterance and repeatedly interrupt itself.

Feed the exact rendered speaker samples into the AEC reference path. Account
for speaker, operating-system, and hardware buffering so the reference is time
aligned with microphone capture. Noise suppression and automatic gain control
are useful after echo cancellation, but they are not substitutes for AEC.

For devices without effective AEC, use headphones, a push-to-interrupt button,
or disable barge-in.

## Recommended satellite state machine

Use local speech activity for immediate UI and playback reactions. Waiting for
the provider's interruption acknowledgement adds a network round trip and, with
current Home Assistant transports, that acknowledgement is not exposed as a
dedicated satellite protocol event.

```text
LISTENING
    user utterance ends
        -> PROCESSING

PROCESSING
    first assistant audio arrives
        -> SPEAKING

SPEAKING
    local user speech starts
        -> purge playback buffers
        -> gate/drop incoming assistant audio
        -> LISTENING

LISTENING (during barge-in)
    local user speech ends
        -> PROCESSING

PROCESSING (after barge-in)
    replacement assistant audio arrives
        -> open playback gate
        -> SPEAKING

SPEAKING
    normal AudioStop/playback completion
        -> IDLE or LISTENING, according to conversation continuation
```

Use a short speech-start debounce to avoid clicks causing interruption, but keep
it small enough that the interaction still feels immediate. The provider
remains authoritative about whether generation is actually cancelled.

## Playback cancellation

On local speech start while speaking:

1. Stop the audio device immediately.
2. Clear the device/DMA buffer, decoder buffer, jitter buffer, and application
   playback queue.
3. Continue microphone transmission.
4. Drop incoming assistant audio while the user remains active.
5. Change the display to `LISTENING`.

Do not close the Home Assistant TTS stream. The replacement response is sent on
that same stream. Treat the interruption as a discontinuity within one stream,
not as the end of the pipeline.

After local speech ends, enter `PROCESSING`. Resume playback when replacement
audio begins arriving. A small post-speech guard window can discard old packets
already in transit, but keep it short and configurable because an excessive
window clips the replacement response.

## Current Home Assistant transport limitation

The standard Home Assistant streaming-TTS path contains raw audio, not an
explicit mid-stream `interrupt` marker. `AudioStop` is sent only when the entire
TTS stream ends. Therefore a generic remote cannot determine with certainty
whether an arbitrary audio gap represents interruption, network jitter, or
normal model pacing.

For current clients, local AEC/VAD is the practical trigger for stopping
playback and changing the display. The integration clears audio that Home
Assistant has not consumed, but it cannot recall audio already buffered on the
network or device.

A future protocol extension should carry explicit events similar to:

```text
assistant_output_interrupted(response_id)
assistant_output_started(response_id)
```

Those events would let the satellite purge only the interrupted response and
resume on a positively identified replacement response.

## Wyoming implementations

A Wyoming satellite normally receives `AudioStart`, `AudioChunk`, and
`AudioStop` for one streaming response. In the current Home Assistant path,
barge-in replacement audio remains inside the same `AudioStart`/`AudioStop`
pair.

Consequently, a Wyoming client that supports this integration should:

- continue emitting microphone `AudioChunk` events during received playback;
- run local AEC and speech detection during playback;
- purge and gate its playback queue on local speech start;
- not send microphone `AudioStop` merely because assistant playback began;
- keep accepting server audio while gated, dropping it until safe to resume;
- resume the existing playback stream for replacement audio; and
- emit its normal playback-completed/`Played` indication only after the final
  `AudioStop` has actually played.

Do not emit a false `Played` event for the interrupted partial response. Home
Assistant treats that as completion of the whole TTS response.

## Buffering and latency targets

Suggested starting targets:

| Component | Target |
|---|---:|
| Microphone packet duration | 20–40 ms |
| Local playback queue | 40–100 ms |
| Speech-start debounce | 40–100 ms |
| Playback purge after speech start | Under 50 ms |
| Post-speech stale-audio guard | 0–100 ms, tune experimentally |

Smaller buffers improve interruption responsiveness but increase sensitivity to
network jitter and scheduling delays. Measure on the actual hardware rather
than assuming desktop behavior applies to an embedded device.

## Failure handling

- If the network disconnects, stop playback, close capture, and return to an
  error or idle state.
- If no replacement audio arrives within the normal response timeout, end the
  active interaction rather than remaining in `PROCESSING` forever.
- If AEC reports divergence or repeated self-interruptions occur, disable local
  automatic interruption and expose a push-to-interrupt fallback.
- If the user stops transmitting microphone audio during playback, barge-in is
  unavailable for that turn; playback should otherwise continue normally.

## Validation checklist

Test with response transcription both enabled and disabled.

- Microphone packets continue while assistant audio is playing.
- Speaker loopback alone does not trigger interruption.
- Real speech during playback changes the display to `LISTENING` immediately.
- Old queued playback is inaudible after interruption.
- Microphone transmission continues throughout the interruption.
- The display changes to `PROCESSING` when the user stops speaking.
- Replacement audio changes the display back to `SPEAKING`.
- Replacement audio is not clipped by the stale-audio guard.
- Only final playback completion produces the normal completed/`Played` event.
- Disabling barge-in preserves existing half-duplex behavior.
- Enabling barge-in does not enable or display response transcripts.
- Repeated interruptions in one session do not leak tasks, audio buffers, or
  response state.

For an end-to-end test, use a long assistant response, interrupt it after
several words, and verify that the old response stops, the display follows the
states above, and the replacement answer plays without opening a new provider
session.
