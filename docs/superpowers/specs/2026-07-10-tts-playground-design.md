# TTS Playground API Server Design

Date: 2026-07-10  
Status: Approved in conversation; awaiting review of this written specification

## 1. Purpose

Build an internal HTTP service for comparing two local text-to-speech tiers on the current host:

- Qwen3-TTS 1.7B for higher-cost, higher-quality synthesis.
- Supertonic 3 for lower-cost, medium-quality synthesis.

The service is initially an evaluation playground with one concurrent Qwen request. Its public API must remain stable when the deployment later expands to a small shared service or an always-on multi-GPU service.

## 2. Environment and constraints

The current host has:

- Ubuntu Linux on x86_64.
- One NVIDIA GeForce RTX 5090 with 32 GB VRAM.
- 60 GB system RAM.
- Docker 29 and Docker Compose 5.
- An existing `llama-server` process using about 14.4 GB VRAM and substantial GPU compute.
- `/srv` on a separate volume with about 50 GB free.
- `/var`, where Docker stores image layers, with about 71 GB free.
- Existing private connectivity from the office network and AWS to the host's llama.cpp API.

The TTS deployment must not stop, restart, or reconfigure the existing llama.cpp process. GPU-memory pressure in the TTS service must fail the Qwen request without propagating failure to llama.cpp.

## 3. Scope

### Included

- Qwen3-TTS CustomVoice using `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- Qwen3-TTS VoiceDesign using `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- Qwen3-TTS voice cloning using `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- Supertonic 3 using `Supertone/supertonic-3` and the official Python SDK/server.
- A single authenticated, OpenAI-shaped HTTP entry point.
- Persistent registration of reference voices for Qwen voice cloning.
- Health, readiness, structured logs, and Prometheus metrics.
- Docker Compose deployment and repeatable API smoke/benchmark scripts.

### Not included initially

- MeloTTS.
- Internet-public access.
- Streaming audio responses.
- MP3, OGG, or FLAC output from the public endpoint.
- Multiple simultaneous Qwen generations.
- Training or fine-tuning.
- A browser UI.
- Automatic control of the existing llama.cpp service.

## 4. Architecture

The deployment contains three services on one private Docker network:

```text
Office network / AWS
          |
          v
tts-gateway :8020
    |             |
    v             v
qwen-worker    supertonic-worker
GPU, lazy      CPU, always loaded
```

Only `tts-gateway` publishes a host port. Worker ports use Docker `expose` and are unreachable directly from the office network or AWS.

### 4.1 tts-gateway

Responsibilities:

- Authenticate the caller with a bearer token.
- Validate the public request schema and model-specific options.
- Normalize ISO language codes for each backend.
- Route requests by the stable public model identifier.
- Forward voice registration and deletion to the Qwen worker.
- Enforce body-size and upstream timeouts.
- Return a uniform audio response or error envelope.
- Emit request metrics and privacy-safe structured logs.

The gateway does not load ML models and does not persist uploaded audio.

### 4.2 qwen-worker

Responsibilities:

- Own CUDA access for Qwen3-TTS.
- Keep no more than one of the three Qwen checkpoints loaded.
- Serialize all model loads and inference.
- Unload an idle checkpoint after a configurable interval.
- Reject a model load when available VRAM is below a configurable guard.
- Persist reference audio and metadata for voice cloning.
- Cache derived clone prompts only in process memory.

The worker loads Qwen models with BF16. FlashAttention 2 is preferred when the installed build supports the RTX 5090; the implementation must have an explicit tested fallback to PyTorch SDPA rather than silently selecting an unknown attention backend.

### 4.3 supertonic-worker

Supertonic runs on CPU through the official Python package and ONNX Runtime. It stays loaded so that the low-cost path remains available while Qwen is unloaded, loading, switching, or in an error state.

The gateway calls Supertonic's native `/v1/tts` endpoint so that language, step count, and speed remain explicit. Supertonic requests are serialized per worker because the underlying ONNX session is serialized.

## 5. Public API

### 5.1 Authentication

All `/v1/*` endpoints require:

```http
Authorization: Bearer <internal-api-key>
```

`/healthz` remains unauthenticated for container health checks. `/readyz` and `/metrics` are authenticated when reached through the gateway.

### 5.2 Synthesis

```http
POST /v1/audio/speech
Content-Type: application/json
```

Request fields:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `model` | string | yes | Stable model identifier listed below |
| `input` | string | yes | Text to synthesize |
| `voice` | string | by model | Built-in speaker or registered clone voice ID |
| `language` | string | yes | ISO code such as `ko` or `en`; `auto` where supported |
| `instructions` | string | by model | Style or voice-design instruction |
| `response_format` | string | no | Must be `wav`; defaults to `wav` |
| `speed` | number | no | Supported by Supertonic; defaults to `1.0` |
| `steps` | integer | no | Supertonic quality/cost setting from 5 to 12; defaults to `8` |

Stable public model identifiers:

| Identifier | Backend | Rules |
|---|---|---|
| `supertonic-3` | Supertonic | `voice` is `M1`-`M5` or `F1`-`F5`; `speed` and `steps` are supported |
| `qwen3-tts-1.7b-custom` | Qwen CustomVoice | `voice` is a Qwen built-in speaker; `instructions` is optional |
| `qwen3-tts-1.7b-design` | Qwen VoiceDesign | `instructions` is required; `voice` is rejected |
| `qwen3-tts-1.7b-clone` | Qwen Base | `voice` must be a registered `voice_*` identifier |

Unsupported fields are rejected with HTTP 422. They are never silently ignored. In particular, Qwen requests reject `speed` values other than `1.0` and reject `steps` in the initial release.

The gateway accepts synthesis text from 1 to 2,000 Unicode characters and instructions from 1 to 1,000 characters. Voice names are limited to 100 characters and reference transcripts to 4,000 characters. These limits are checked before any backend call.

Success returns complete WAV bytes:

```http
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Request-ID: req_...
X-TTS-Model: supertonic-3
X-Queue-Ms: 0
X-Processing-Ms: 184
X-Audio-Duration: 4.21
```

### 5.3 Voice registration

```http
POST /v1/audio/voices
Content-Type: multipart/form-data
```

Fields:

- `name`: human-readable internal name.
- `audio`: direct WAV or FLAC upload.
- `transcript`: exact transcript of the reference speech.
- `language`: ISO language code.
- `consent_confirmed`: must be `true`.

The gateway and worker reject:

- Files larger than 20 MB.
- Audio shorter than 3 seconds or longer than 30 seconds.
- Unsupported or corrupt audio.
- URL-based audio references.
- Missing transcript or consent confirmation.

The worker creates an opaque UUID-based identifier prefixed with `voice_`. It stores normalized reference audio and a JSON metadata file under `/data/voices/<voice_id>/`. Client filenames are never used as filesystem paths.

Derived Qwen clone prompts remain in RAM. After restart, the first use of a voice rebuilds the prompt from the persisted audio and transcript. No Python pickle or device-specific tensor is persisted.

Voice list and detail responses contain the voice ID, name, language, duration, creation time, and consent flag. They do not return the reference transcript or filesystem path.

Additional endpoints:

- `GET /v1/audio/voices` lists registered voice metadata without audio bytes.
- `GET /v1/audio/voices/{voice_id}` returns one metadata record.
- `DELETE /v1/audio/voices/{voice_id}` deletes persisted files and any in-memory prompt.

### 5.4 Discovery and health

- `GET /v1/models` returns model capabilities, supported speakers, worker state, and which Qwen checkpoint is loaded.
- `GET /healthz` reports that the gateway process is alive.
- `GET /readyz` reports gateway and worker reachability. An unloaded Qwen model is ready; Qwen readiness means the worker can accept a load request, not that a checkpoint is resident. It returns HTTP 503 when either required worker is unreachable or Qwen is in its error-recovery cooldown, even though Supertonic synthesis may still remain available.
- `GET /metrics` exposes Prometheus text metrics.

## 6. Qwen model lifecycle

The worker state machine is:

```text
UNLOADED -> LOADING -> READY
    ^          |         |
    |          v         v
    +-------- ERROR <- SWITCHING
```

Initial defaults:

- Maximum active Qwen generations: 1.
- Maximum waiting Qwen requests: 3.
- Queue wait timeout: 180 seconds.
- End-to-end Qwen upstream timeout: 360 seconds.
- Idle unload: 600 seconds.
- Minimum free VRAM before loading a checkpoint: 10 GiB.
- Error-recovery cooldown: 30 seconds.

For a request targeting a different checkpoint, the worker:

1. Acquires the inference/model-management lock.
2. Releases references to the current model.
3. Runs Python garbage collection and releases unused CUDA cache.
4. Checks free VRAM.
5. Loads the requested checkpoint.
6. Runs inference.
7. Records the last-used time for idle unloading.

If loading or inference raises CUDA OOM, the worker unloads its own model and clears its own cache, then retries the request once. A second failure returns HTTP 503 and leaves the worker in `ERROR`. During the 30-second recovery cooldown, new Qwen requests return HTTP 503 without attempting a load. The first Qwen request after the cooldown transitions the worker to `UNLOADED` and makes one clean load attempt. It never sends signals or commands to llama.cpp.

## 7. Storage and deployment configuration

Host storage:

- Qwen Hugging Face cache: `/srv/tts-cache/huggingface`.
- Supertonic cache: `/srv/tts-cache/supertonic`.
- Registered reference voices: `/srv/tts-data/voices`.

Container paths:

- Qwen cache: `/cache/huggingface` through `HF_HOME`.
- Supertonic cache: the SDK's expected cache directory mapped from `/srv/tts-cache/supertonic`.
- Voice data: `/data/voices`.

The API key is stored in a local file ignored by Git and mounted read-only as a Compose secret. It is not embedded in an image, committed file, or command-line argument.

Deployment configuration is supplied through a checked-in `.env.example` and a local ignored `.env`. At minimum it controls the bind host, public port, cache/data paths, Qwen idle timeout, queue limit, timeout, and free-VRAM guard.

## 8. Network and security

- Publish only the gateway on `0.0.0.0:8020` by default.
- Restrict port 8020 at the host/network firewall to the existing office and AWS source ranges used for llama.cpp.
- Reuse the existing encrypted private path. If that path is not encrypted, terminate TLS at the existing internal reverse proxy.
- Do not publish worker ports.
- Accept clone audio only as a direct upload to prevent SSRF.
- Sniff and decode audio rather than trusting filename extensions or content type.
- Do not log request text, instructions, transcripts, reference audio, bearer tokens, or generated audio.
- Record consent confirmation and creation time in voice metadata.

The repository must retain Qwen's Apache-2.0 notices and document that Supertonic code is MIT while its model is OpenRAIL-M. Operators are responsible for ensuring the submitted reference voice may lawfully be cloned.

## 9. Errors

All JSON errors use:

```json
{
  "error": {
    "code": "model_switch_timeout",
    "message": "Qwen model switch timed out.",
    "request_id": "req_..."
  }
}
```

Status mapping:

- 400: malformed request audio or invalid input.
- 401: missing or invalid bearer token.
- 404: unknown model or voice ID.
- 413: request or reference audio too large.
- 422: model-specific validation failure.
- 429: bounded queue is full.
- 503: insufficient VRAM, model-load failure, or unavailable worker.
- 504: queue, model-switch, or synthesis timeout.

## 10. Observability

Logs are JSON and include request ID, model identifier, character count, queue time, processing time, status, error code, audio duration, and current Qwen checkpoint. They exclude content and credentials.

Metrics include:

- Request and error counters by model and status.
- Queue and processing latency histograms.
- Generated audio duration and real-time factor.
- Current Qwen queue depth.
- Qwen checkpoint load/switch count and duration.
- Current Qwen state and loaded checkpoint.
- GPU free VRAM at load time.
- Worker readiness.

## 11. Testing and evaluation

### Automated tests

- Unit tests for authentication, schema validation, language mapping, routing, error mapping, queue limits, and privacy-safe logging.
- Gateway integration tests using fake backend services.
- Qwen worker lifecycle tests with a fake model adapter, including switching, idle unload, queue timeout, VRAM guard, retry, and cleanup.
- Voice registry tests for upload validation, safe paths, persistence, reload, and deletion.
- Container smoke tests for health and readiness.
- Hardware smoke tests for all three Qwen checkpoints and Supertonic on Korean text.

### Evaluation corpus

Use fixed short, medium, and long Korean passages containing:

- Numbers, dates, currency, and units.
- Mixed Korean and English.
- Proper nouns.
- Emotional delivery instructions.
- One consented reference voice for clone similarity.

Run each sample three times and record end-to-end latency, model-switch time, real-time factor, peak VRAM/RAM, output duration, pronunciation accuracy, naturalness, and clone speaker similarity.

### Acceptance criteria

1. The same authenticated API succeeds from the office network and the existing AWS environment.
2. Supertonic remains available while Qwen is unloaded, loading, switching, or failed.
3. All three Qwen checkpoints can be selected through the same public endpoint.
4. Qwen OOM or load failure does not stop or restart llama.cpp.
5. A registered clone voice remains usable after service restart.
6. Logs contain no synthesis text, transcript, audio, or credentials.
7. The benchmark script produces comparable latency, RTF, and resource records for Qwen and Supertonic.

## 12. Expansion path

The public API and model identifiers remain unchanged as the deployment grows.

- Small shared service: run additional CPU Supertonic replicas and move each Qwen checkpoint to a dedicated worker where GPU capacity permits.
- Always-on service: place Qwen workers on separate GPUs or hosts, add gateway load balancing and per-model concurrency controls, and remove model-switch delays.
- Streaming: add a capability flag and streaming response without changing existing non-streaming behavior.
- Additional formats: add explicit gateway transcoding after WAV behavior is stable.

No Redis, durable job queue, Kubernetes, or external database is introduced in the initial playground because a single-user synchronous evaluation service does not require them.
