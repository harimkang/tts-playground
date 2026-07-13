# Optional-Transcript and Ephemeral Voice Clone Design

Date: 2026-07-13
Status: Approved in conversation; awaiting review of this written specification

## 1. Purpose

Extend the existing Qwen3-TTS voice-clone API in two compatible ways:

1. Make the reference transcript optional for registered voices. When supplied, it enables the
   higher-accuracy text-assisted clone path; when omitted or blank, the service uses Qwen's
   speaker-embedding-only path.
2. Add a one-request clone API for callers that retain their own reference audio and do not want
   the service to persist it.

The existing registered `voice_id` workflow remains supported without breaking current clients or
stored voices.

## 2. Public API contract

### 2.1 Persistent voice registration

`POST /v1/audio/voices` remains a multipart request with the existing fields:

- `name`: required.
- `audio`: required WAV or FLAC reference audio.
- `transcript`: optional, at most 4,000 characters.
- `language`: required.
- `consent_confirmed`: required and must be `true`.

An omitted, empty, or whitespace-only `transcript` is normalized to `null`. A non-empty value is
trimmed only for the empty-value decision; its supplied content is otherwise preserved because it
must correspond exactly to the spoken reference.

The response and subsequent JSON synthesis contract remain unchanged: registration returns an
opaque `voice_*` identifier, and callers use it with `model=qwen3-tts-1.7b-clone` on
`POST /v1/audio/speech`.

### 2.2 One-request voice cloning

Add an authenticated multipart endpoint:

```http
POST /v1/audio/speech/clone
Content-Type: multipart/form-data
```

Fields:

| Field | Type | Required | Rules |
|---|---|---:|---|
| `input` | string | yes | 1-2,000 Unicode characters |
| `audio` | file | yes | WAV or FLAC; existing reference-audio limits apply |
| `transcript` | string | no | 1-4,000 characters when non-blank; blank means absent |
| `language` | string | no | Supported Qwen language code; defaults to `auto` |
| `response_format` | string | no | Must be `wav`; defaults to `wav` |
| `consent_confirmed` | boolean | yes | Must be `true` |

Success returns the same WAV body and safe response headers as the existing synthesis endpoint.
The endpoint does not accept `voice`, `instructions`, `speed`, or `steps`.

Keeping this endpoint separate from the JSON synthesis endpoint preserves the stable existing JSON
contract, keeps multipart validation explicit, and avoids an ambiguous endpoint that accepts two
unrelated content types.

## 3. Clone-mode selection

Both persistent and one-request clones use the same rule:

- Non-empty transcript: call `create_voice_clone_prompt` with the reference text and
  `x_vector_only_mode=False`.
- Missing or blank transcript: call it with `ref_text=None` and
  `x_vector_only_mode=True`.

The text-assisted mode is documented as the preferred path when an accurate transcript is
available. Speaker-embedding-only mode is a convenience and privacy option and may have lower
similarity, pronunciation accuracy, or prosody fidelity.

## 4. Data lifecycle

### 4.1 Registered voices

Registered voices continue to store normalized reference audio and JSON metadata under the
configured `TTS_DATA_DIR`. Metadata stores `transcript` as either a string or JSON `null`. Existing
metadata containing a string remains compatible. Clone prompts remain memory-only and are rebuilt
after restart as before.

### 4.2 One-request voices

The gateway streams the bounded multipart request to the private Qwen worker and does not persist
the upload. The worker may hold bounded request bytes in memory and writes only a request-scoped
normalized reference file beneath the container's `/tmp` tmpfs because Qwen's prompt API requires
a file path.

The temporary audio and derived clone prompt are not entered into the voice registry, are not
cached, and are removed or released in a `finally` path after success, validation failure, model
failure, timeout, or caller cancellation. No reference artifact is written beneath `TTS_DATA_DIR`.

The implementation must use unpredictable request-scoped paths, must not incorporate the client
filename, and must not return internal paths.

## 5. Worker and concurrency integration

One-request clones use the existing Qwen Base checkpoint and pass through the same bounded FIFO
gate, single-active-request rule, model-switch lifecycle, VRAM guard, OOM retry policy, cooldown,
and cancellation ordering as registered clones.

The model manager gains a transient-clone operation that builds a request-local prompt instead of
looking up or caching a registered voice prompt. GPU work must finish before the request releases
the gate or removes resources used by an in-flight blocking call.

The existing Qwen internal registration endpoint accepts optional transcript metadata. A separate
private multipart synthesis endpoint handles the one-request path so persistent registration is
never used as an implementation shortcut.

## 6. Validation and errors

Both clone workflows share the current reference-audio validation:

- Maximum encoded upload: 20 MB.
- Formats: WAV or FLAC only.
- Duration: 3-30 seconds.
- At most two channels.
- Sample rate at most 192 kHz.
- At most 6,000,000 decoded samples.
- Metadata is checked before full decode.

The one-request endpoint also enforces the existing Qwen language allowlist and synthesis-text
limit before GPU admission. Invalid multipart fields return the established controlled error
envelope with a request ID. Unsupported media, excessive size, corrupt audio, unsupported
language, missing consent, queue timeout, VRAM shortage, and model failure keep their existing
stable error codes where applicable.

The gateway and workers must not log synthesis input, transcript, audio bytes, client filenames,
bearer tokens, generated audio, or exception messages that may include those values.

## 7. Discovery, documentation, and operations

Model discovery continues to advertise `voice_cloning=true` for Qwen Base and additionally exposes
that both registered and ephemeral reference modes are supported. README examples cover:

- Registered clone with transcript.
- Registered clone without transcript.
- One-request clone with transcript.
- One-request clone without transcript.
- Explicit persistence and deletion behavior for registered voices.
- Guaranteed non-persistence and request-scoped cleanup for one-request voices.

No new host port, container, volume, model checkpoint, API key, or external service is introduced.
The gateway remains the only published service on port 8020.

## 8. Verification

Development follows regression-first tests. Coverage must include:

- Registration accepts omitted, empty, and whitespace-only transcripts and persists `null`.
- Existing non-empty transcripts remain compatible and select text-assisted mode.
- Stored `null` metadata reloads safely after worker restart.
- One-request multipart validation, authentication, consent, language, audio, and text limits.
- Exact adapter mapping for `x_vector_only_mode=True/False` and `ref_text=None/string`.
- One-request prompts are never cached or added to the voice registry.
- Temporary files are absent after success, validation failure, backend failure, timeout, and
  cancellation.
- Privacy markers from input, transcript, filenames, and model exceptions do not appear in logs or
  public errors.
- Existing registered `voice_id`, CustomVoice, VoiceDesign, Supertonic, queue, VRAM, and OOM tests
  remain green.

After the software gate passes, rebuild affected Docker images and run live acceptance for all of
the following:

1. Registered clone with transcript returns a valid WAV.
2. Registered clone without transcript returns a valid WAV.
3. One-request clone with transcript returns a valid WAV and leaves no registry/data artifact.
4. One-request clone without transcript returns a valid WAV and leaves no registry/data artifact.
5. All TTS containers remain healthy, only port 8020 is published, privacy markers are absent, and
   the existing llama.cpp PID, health, and VRAM allocation remain unchanged.

## 9. Compatibility and out of scope

This change is additive except that registration no longer rejects a missing transcript. Existing
voice IDs, stored metadata, synthesis requests, authentication, model identifiers, and response
formats remain compatible.

Out of scope:

- Persisting an ephemeral reference automatically.
- Returning or downloading stored reference audio.
- Streaming synthesis output.
- Supporting URL-based reference audio.
- Prompt caching across ephemeral requests.
- Automatic speech-to-text generation of a missing transcript.
