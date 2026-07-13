# TTS Playground

TTS Playground is an authenticated internal API for comparing two local speech tiers behind one stable HTTP contract:

- `tts-gateway` is the only published service. It validates and authenticates requests, then routes them to private workers.
- `supertonic-worker` keeps Supertonic 3 on CPU for the low-cost path.
- `qwen-worker` uses one NVIDIA GPU and lazily keeps at most one Qwen3-TTS 1.7B checkpoint resident. Switching public Qwen model IDs unloads and reloads that checkpoint.

The current target host is Ubuntu x86_64 with one RTX 5090 (32 GiB VRAM), 60 GiB RAM, Docker 29, Docker Compose 5, about 50 GiB free under `/srv`, and an existing `llama-server` using roughly 14.4 GiB VRAM. This deployment must not stop, restart, or reconfigure that process. The Qwen worker checks free VRAM and confines cleanup to its own process.

## Prerequisites

- Ubuntu Linux on x86_64 with the target cache/data capacity under `/srv`.
- A working NVIDIA driver compatible with the CUDA 12.8 container image.
- Docker Engine with the Compose plugin.
- NVIDIA Container Toolkit configured for Docker. Confirm `docker info` reports the NVIDIA runtime and that an approved CUDA container can run `nvidia-smi` before deployment.
- `curl`, `openssl`, and standard POSIX shell utilities.
- `uv` with Python 3.12 available. The checked-in `uv.lock` is authoritative; `uv sync --frozen --extra dev` creates/updates the repository `.venv` without changing the lock.
- Network access to the container registries, Python package indexes recorded by `uv.lock`, Hugging Face, and the Supertonic model host during build/prefetch/first startup.

Run commands from the repository root. Do not launch the stack until the host firewall and TLS boundary described below are in place.

## Deploy and operate

Create the local environment file, review every value, and load its directory/user settings into the preparation shell:

```sh
cp .env.example .env
set -a
. ./.env
set +a
./scripts/prepare-host.sh
./scripts/generate-api-key.sh
```

`prepare-host.sh` creates the configured cache and data directories with the numeric `TTS_UID:TTS_GID`. If it reports that `/srv` requires administrator access, use this complete command after loading `.env` as shown above; it explicitly passes every ownership and path setting through `sudo`:

```sh
sudo env \
  TTS_UID="$TTS_UID" \
  TTS_GID="$TTS_GID" \
  TTS_HF_CACHE_DIR="$TTS_HF_CACHE_DIR" \
  TTS_SUPERTONIC_CACHE_DIR="$TTS_SUPERTONIC_CACHE_DIR" \
  TTS_DATA_DIR="$TTS_DATA_DIR" \
  ./scripts/prepare-host.sh
```

`generate-api-key.sh` creates `secrets/tts_api_key.txt` once, with mode `0600`, and refuses to overwrite it.

Build every image from the frozen lock, then prefetch the exact three Qwen repositories into the mounted Hugging Face cache:

```sh
docker compose --env-file .env build
docker compose --env-file .env run --rm --no-deps qwen-worker \
  uv run --no-sync python /app/scripts/prefetch-models.py
```

The prefetch command stops before downloading if `/cache/huggingface` has less than 20 GiB free. It prints repository names and final snapshot paths, but does not print Hugging Face credentials. A private-model token, if ever required, must be supplied through the standard Hugging Face container environment or secret mechanism—not placed in a command.

Start the stack only after build and prefetch complete:

```sh
docker compose --env-file .env up -d
docker compose --env-file .env ps
docker compose --env-file .env logs --tail=100 tts-gateway qwen-worker supertonic-worker
```

Load the API key into the shell without displaying it, then check the public gateway:

```sh
export TTS_BASE_URL="http://127.0.0.1:8020"
export TTS_API_KEY="$(tr -d '\r\n' < secrets/tts_api_key.txt)"
curl --fail --silent --show-error "$TTS_BASE_URL/healthz"
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  "$TTS_BASE_URL/readyz"
```

Run the fast Supertonic smoke check; opt into the slower Qwen CustomVoice load check explicitly:

```sh
./scripts/smoke.sh
RUN_QWEN_SMOKE=1 ./scripts/smoke.sh
```

The smoke script calls `/healthz`, authenticated `/readyz` and `/v1/models`, and synthesis. It rejects output that is no larger than a WAV header or does not begin with `RIFF`.

For the fixed Korean benchmark, prepare the locked local `.venv`, keep the bearer token in its file, and run the CLI through `uv`:

```sh
uv sync --frozen --extra dev
TTS_BASE_URL="$TTS_BASE_URL" \
TTS_API_KEY_FILE=secrets/tts_api_key.txt \
uv run --frozen python scripts/benchmark.py
```

The benchmark validates `benchmarks/corpus.ko.json`, runs Supertonic `F1` and Qwen CustomVoice `Sohee` three times per case in serial order, and creates a timestamped directory under `results/`. Each run contains WAV files plus complete `results.jsonl` and `results.csv` records. HTTP/request failures receive result rows too. The key is never included in console output or result files. Use `--corpus`, `--output-root`, `--repeats`, `--base-url`, and `--timeout-seconds` for controlled experiments; record any non-default flags with the results.

Inspect current state and logs, or stop the deployment without deleting host caches and voice data:

```sh
docker compose --env-file .env ps
docker compose --env-file .env logs --since=10m tts-gateway qwen-worker supertonic-worker
docker compose --env-file .env down
```

Logs intentionally exclude bearer tokens, input text, instructions, transcripts, reference audio, and generated audio. Still treat operational logs as internal data.

## Synthesis API examples

All synthesis calls return WAV bytes. These examples assume `TTS_BASE_URL`, `TTS_API_KEY`, and (for clone use) `VOICE_ID` are already set.

Supertonic 3, with its supported speed and step controls:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"supertonic-3","input":"안녕하세요.","voice":"F1","language":"ko","speed":1.0,"steps":8,"response_format":"wav"}' \
  -o supertonic.wav \
  "$TTS_BASE_URL/v1/audio/speech"
```

Qwen CustomVoice:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-custom","input":"차분하게 안내를 시작하겠습니다.","voice":"Sohee","language":"ko","instructions":"차분하고 명확하게 말해 주세요.","response_format":"wav"}' \
  -o qwen-custom.wav \
  "$TTS_BASE_URL/v1/audio/speech"
```

Qwen VoiceDesign:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-1.7b-design","input":"새로운 탐험이 지금 시작됩니다.","language":"ko","instructions":"따뜻하고 자신감 있는 젊은 내레이터 목소리","response_format":"wav"}' \
  -o qwen-design.wav \
  "$TTS_BASE_URL/v1/audio/speech"
```

Qwen registered voice clone:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-1.7b-clone\",\"input\":\"등록된 음성으로 읽는 문장입니다.\",\"voice\":\"$VOICE_ID\",\"language\":\"ko\",\"response_format\":\"wav\"}" \
  -o qwen-clone.wav \
  "$TTS_BASE_URL/v1/audio/speech"
```

Qwen rejects `steps` and non-default `speed`; VoiceDesign requires `instructions` and rejects `voice`.

## Voice clone workflows and lifecycle

`POST /v1/audio/voices` registers a reusable reference, while `POST /v1/audio/speech/clone` uses one only for the current request. Both clone workflows accept a direct WAV or FLAC reference between 3 and 30 seconds, no larger than 20 MB, with at most 2 channels, a sample rate no higher than 192 kHz, and at most 6,000,000 decoded samples total across channels. URL inputs are not accepted. `transcript` is optional: include an exact reference transcript for text-assisted cloning, which generally gives better pronunciation and similarity, or omit it for speaker-embedding-only cloning. Internally, omission enables `x_vector_only_mode`; it is convenient when no reliable transcript exists but may reduce clone quality.

Persistent registration, text-assisted:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'name=approved-reference' -F 'language=ko' \
  -F 'transcript=정확한 기준 음성 녹취' -F 'consent_confirmed=true' \
  -F 'audio=@reference.wav;type=audio/wav' \
  "$TTS_BASE_URL/v1/audio/voices"
```

Persistent registration, speaker-embedding-only (omit `transcript`):

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'name=approved-reference' -F 'language=ko' \
  -F 'consent_confirmed=true' \
  -F 'audio=@reference.wav;type=audio/wav' \
  "$TTS_BASE_URL/v1/audio/voices"
```

Copy the returned opaque `voice_...` ID into `VOICE_ID`. A registered reference and its private transcript persist in the registry data volume until explicitly deleted. List registrations with `GET /v1/audio/voices`, use the ID with the `qwen3-tts-1.7b-clone` example above, and delete it with `DELETE /v1/audio/voices/{voice_id}`:

```sh
export VOICE_ID='voice_replace_with_returned_id'
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  "$TTS_BASE_URL/v1/audio/voices"
curl --fail --silent --show-error -X DELETE \
  -H "Authorization: Bearer $TTS_API_KEY" \
  "$TTS_BASE_URL/v1/audio/voices/$VOICE_ID"
```

`POST /v1/audio/speech/clone` performs a one-request clone without creating a voice ID. Ephemeral clone, text-assisted:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'input=이번 요청에서만 이 목소리를 사용합니다.' -F 'language=ko' \
  -F 'transcript=정확한 기준 음성 녹취' -F 'consent_confirmed=true' \
  -F 'audio=@reference.wav;type=audio/wav' \
  -o ephemeral-clone.wav \
  "$TTS_BASE_URL/v1/audio/speech/clone"
```

Ephemeral clone, speaker-embedding-only (omit `transcript`):

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'input=이번 요청에서만 이 목소리를 사용합니다.' -F 'language=ko' \
  -F 'consent_confirmed=true' -F 'audio=@reference.wav;type=audio/wav' \
  -o ephemeral-clone.wav \
  "$TTS_BASE_URL/v1/audio/speech/clone"
```

Ephemeral references are not persisted: they are never added to the registry or data volume, and their private `qwen-clone-*` working directory is removed from `/tmp` after every exit path, including validation, backend failure, timeout, and cancellation.

**Legal/consent warning:** `consent_confirmed=true` records only the caller's attestation. It is not proof of consent and is not a substitute for copyright, publicity, privacy, employment, contractual, or other legal authority to clone or use a person's voice. The operator and caller remain responsible for lawful collection and use.

## Errors and Qwen switching

JSON errors contain `error.code`, a safe message, and `request_id`. The public codes implemented by the gateway and the errors it safely forwards from its private workers are:

| HTTP | Code | Operator meaning |
| ---: | --- | --- |
| 400 | `invalid_audio` | Reference audio cannot be decoded or is not WAV/FLAC. Replace the upload. |
| 401 | `unauthorized` | The bearer token is missing, malformed, or incorrect. |
| 404 | `not_found` | The public route/resource does not exist. |
| 404 | `voice_not_found` | The supplied registered voice ID does not exist. List voices and retry with a current ID. |
| 405 | `method_not_allowed` | The route exists but does not accept that HTTP method. |
| 413 | `audio_too_large` | Reference audio exceeds 20 MB. |
| 413 | `request_too_large` | A synthesis JSON or complete multipart request exceeds its bounded size. |
| 422 | `validation_error` | JSON/form data or model-specific fields fail public schema validation. |
| 422 | `unsupported_language` | The selected backend does not support the requested language code. |
| 422 | `consent_required` | Voice registration or ephemeral cloning did not attest `consent_confirmed=true`. |
| 422 | `invalid_audio_duration` | Reference audio is shorter than 3 seconds or longer than 30 seconds. |
| 422 | `invalid_multipart` | Voice-registration or ephemeral-clone multipart content type, boundary, or byte stream is malformed. |
| 429 | `queue_full` | Three Qwen requests are already waiting by default; retry with backoff. |
| 500 | `internal_error` | The Qwen worker encountered an unexpected internal failure; inspect its privacy-safe logs. |
| 503 | `backend_not_configured` | A gateway was started without a backend router; this indicates deployment/configuration failure. |
| 503 | `backend_unavailable` | A required private worker cannot be reached; check container status and networks. |
| 503 | `backend_invalid_response` | A worker returned malformed discovery/voice JSON or a non-WAV synthesis payload. Inspect worker logs. |
| 503 | `worker_not_ready` | The Qwen process is alive but its runtime manager/voice registry is not ready yet. |
| 503 | `insufficient_vram` | Free VRAM is below `QWEN_MIN_FREE_VRAM_BYTES`; leave the existing GPU workload untouched and retry later. |
| 503 | `model_load_failed` | The requested Qwen checkpoint could not be loaded; inspect Qwen logs/cache integrity. |
| 503 | `model_unavailable` | Qwen is inside its error-recovery cooldown; wait for the configured cooldown. |
| 503 | `cuda_out_of_memory` | Qwen exhausted GPU memory after its one cleanup/retry cycle. Reduce competing VRAM pressure without stopping `llama-server`. |
| 504 | `queue_timeout` | A Qwen request waited longer than `QWEN_QUEUE_TIMEOUT_SECONDS`. |
| 504 | `backend_timeout` | The gateway's private-worker request exceeded its configured timeout. |
| Upstream status | `backend_error` | A worker returned an error without a valid structured envelope; use `request_id` to correlate gateway/worker logs. |
| Other framework status | `http_error` | A framework HTTP error did not match the explicit 404/405 mappings. |

The first Qwen call loads its checkpoint. A call for a different Qwen public model ID synchronously unloads the resident checkpoint and loads the requested one, so it can be much slower than a warm call. The default queue wait is 180 seconds and gateway Qwen timeout is 360 seconds. Do not interpret an unloaded Qwen state as unhealthy: `/readyz` means it can accept a load request.

## Limits and stable API

Operational limits can change without changing clients or public model IDs. Edit `.env`, then recreate the affected containers with `docker compose --env-file .env up -d`:

- `QWEN_MAX_WAITING`: bounded number of waiting Qwen requests.
- `QWEN_QUEUE_TIMEOUT_SECONDS`: maximum queue wait.
- `QWEN_IDLE_UNLOAD_SECONDS`: warm-checkpoint idle lifetime.
- `QWEN_MIN_FREE_VRAM_BYTES`: free-VRAM load guard.
- `QWEN_ERROR_COOLDOWN_SECONDS`: recovery cooldown.
- `QWEN_ATTENTION_IMPLEMENTATION`: explicit supported attention backend; `sdpa` is the safe default.
- `TTS_BIND_HOST` and `TTS_PORT`: gateway bind address and published port.
- `TTS_QWEN_TIMEOUT_SECONDS` and `TTS_SUPERTONIC_TIMEOUT_SECONDS`: private-worker request timeouts.
- `TTS_MAX_SYNTHESIS_BODY_BYTES`: pre-authentication JSON body limit; keep it large enough for the public text and instruction limits.
- `TTS_HF_CACHE_DIR`, `TTS_SUPERTONIC_CACHE_DIR`, and `TTS_DATA_DIR`: persistent host paths.

Keep one active Qwen generation on this single-GPU playground. Raising queue capacity does not add GPU concurrency, and lowering the VRAM guard can interfere with the existing GPU workload.

## Network and security boundary

The default bind publishes gateway port 8020 on all host interfaces; workers remain on private Docker networks. Before startup, restrict the host and network firewall to the same approved office and AWS source ranges already used for the internal `llama-server` API. Do not make this service internet-public.

Bearer authentication does not encrypt traffic. Reuse the existing encrypted private network path. If any office/AWS path is not encrypted end-to-end, terminate TLS at the existing internal reverse proxy and expose only that TLS endpoint to callers. Never publish either worker port.

## Licensing

See [LICENSES.md](LICENSES.md) for the Qwen Apache-2.0 and Supertonic code MIT/model OpenRAIL-M notices and upstream license links. Operators must review those terms for their use case and preserve required notices when redistributing images, code, or model artifacts.
