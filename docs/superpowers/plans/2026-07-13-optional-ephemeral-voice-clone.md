# Optional-Transcript and Ephemeral Voice Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support both persistent `voice_id` cloning and request-scoped voice cloning, with optional reference transcripts selecting Qwen's text-assisted or speaker-embedding-only mode.

**Architecture:** Keep the existing JSON synthesis and persistent voice APIs backward compatible, then add a dedicated authenticated multipart `/v1/audio/speech/clone` endpoint. The gateway bounds and streams the multipart body to a private Qwen endpoint; the worker normalizes reference audio into an unpredictable `/tmp` path, runs it through the existing serialized Qwen Base lifecycle, and deletes it on every exit path without touching the registry.

**Tech Stack:** Python 3.12, FastAPI/Starlette, HTTPX, python-multipart, Pydantic v2, soundfile, Qwen3-TTS, Prometheus client, pytest/pytest-asyncio/respx, Ruff, mypy, uv, Docker Compose.

## Global Constraints

- Preserve `POST /v1/audio/speech` and the registered `voice_id` workflow without breaking existing clients or stored string transcripts.
- `POST /v1/audio/voices` accepts an omitted, empty, or whitespace-only `transcript` and persists it as JSON `null`; a non-empty transcript remains unchanged.
- Non-empty transcript calls `create_voice_clone_prompt(ref_text=<text>, x_vector_only_mode=False)`; absent transcript calls `create_voice_clone_prompt(ref_text=None, x_vector_only_mode=True)`.
- Add authenticated multipart `POST /v1/audio/speech/clone` with required `input`, `audio`, and `consent_confirmed=true`; optional `transcript`, `language=auto`, and `response_format=wav`.
- Apply the existing 20 MB, WAV/FLAC, 3-30 second, two-channel, 192 kHz, and 6,000,000 decoded-sample limits before GPU admission.
- Ephemeral audio may exist only in bounded request memory and an unpredictable request-scoped directory beneath the Qwen container's `/tmp` tmpfs; never write it beneath `TTS_DATA_DIR`.
- Remove ephemeral files and release request-local prompts after success, validation failure, backend failure, timeout, or cancellation; never cache them or add them to the voice registry.
- Use the existing Qwen FIFO gate, one-active-request invariant, VRAM guard, OOM retry, cooldown, model switching, and cancellation ordering.
- Do not log or return synthesis input, reference transcript, audio bytes, client filename, bearer token, generated audio, unsafe exception text, or internal paths.
- Keep only gateway port 8020 published and do not stop, restart, signal, or reconfigure the existing llama.cpp process.
- Manage Python environments and dependencies only with uv, `.venv`, `pyproject.toml`, and `uv.lock`; do not introduce pip, Poetry, or Conda flows.

---

### Task 1: Optional transcript and shared reference-audio normalization

**Files:**
- Create: `src/tts_playground/qwen/reference_audio.py`
- Modify: `src/tts_playground/qwen/voice_registry.py`
- Modify: `src/tts_playground/qwen/adapter.py`
- Modify: `src/tts_playground/qwen/real_adapter.py`
- Modify: `src/tts_playground/qwen/app.py`
- Test: `tests/unit/test_voice_registry.py`
- Test: `tests/unit/test_real_adapter_mapping.py`
- Test: `tests/integration/test_qwen_app.py`

**Interfaces:**
- Produces: `normalize_optional_transcript(value: str | None) -> str | None`.
- Produces: `normalize_reference_audio(audio: bytes, destination: Path) -> float`, which writes PCM-16 WAV and returns validated duration.
- Changes: `VoiceInternal.transcript` and `VoiceRegistry.register(..., transcript=...)` to `str | None`.
- Changes: `QwenAdapter.build_clone_prompt(..., transcript: str | None) -> object`.

- [ ] **Step 1: Add failing transcript normalization and persistence tests**

Add parameterized cases to `tests/unit/test_voice_registry.py` that register `None`, `""`, and `"   "`, then assert both `registry.get_internal(voice_id).transcript is None` and `metadata.json` contains `"transcript": null`. Add a reload case with JSON `null`, and retain a non-empty case asserting the exact text survives registration and reload.

```python
@pytest.mark.parametrize("transcript", [None, "", "   "])
def test_registration_normalizes_missing_or_blank_transcript_to_none(
    tmp_path: Path,
    transcript: str | None,
) -> None:
    registry = VoiceRegistry(tmp_path / "voices")
    public = registry.register(
        name="optional transcript",
        audio=_wav_bytes(),
        transcript=transcript,
        language="ko",
        consent_confirmed=True,
    )

    assert registry.get_internal(public.id).transcript is None
    metadata = json.loads(
        (tmp_path / "voices" / public.id / "metadata.json").read_text()
    )
    assert metadata["transcript"] is None
    assert VoiceRegistry(tmp_path / "voices").get_internal(public.id).transcript is None
```

- [ ] **Step 2: Run the registry tests and verify RED**

Run: `uv run --no-sync pytest -q tests/unit/test_voice_registry.py`

Expected: FAIL because `register` and `VoiceInternal` require `str`, blank values are not normalized, and `_load_voice` rejects JSON `null`.

- [ ] **Step 3: Add failing adapter mode-selection tests**

Extend `tests/unit/test_real_adapter_mapping.py` to prove both exact Qwen calls:

```python
@pytest.mark.parametrize(
    ("transcript", "x_vector_only_mode"),
    [("정확한 녹취", False), (None, True)],
)
def test_build_clone_prompt_selects_mode_from_transcript(
    adapter: QwenTTSAdapter,
    transcript: str | None,
    x_vector_only_mode: bool,
) -> None:
    model = Mock()
    audio_path = Path("/tmp/reference.wav")

    adapter.build_clone_prompt(model, audio_path, transcript)

    model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(audio_path),
        ref_text=transcript,
        x_vector_only_mode=x_vector_only_mode,
    )
```

- [ ] **Step 4: Run the adapter test and verify RED**

Run: `uv run --no-sync pytest -q tests/unit/test_real_adapter_mapping.py`

Expected: FAIL for the `None` case because the implementation always uses `x_vector_only_mode=False`.

- [ ] **Step 5: Extract reference normalization and implement optional transcript**

Create `qwen/reference_audio.py` with the existing constants and metadata-first validation moved from `voice_registry.py`. The central entry point must reject size before `sf.info`, validate metadata before `sf.read`, write a normalized WAV to the caller-supplied destination, and return duration:

```python
def normalize_optional_transcript(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value


def normalize_reference_audio(audio: bytes, destination: Path) -> float:
    if len(audio) > MAX_AUDIO_BYTES:
        raise _audio_too_large()
    audio_stream = io.BytesIO(audio)
    try:
        audio_info = sf.info(audio_stream)
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError):
        raise invalid_audio() from None
    duration_seconds = validate_audio_metadata(audio_info)
    try:
        audio_stream.seek(0)
        samples, sample_rate = sf.read(audio_stream, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError):
        raise invalid_audio() from None
    if sample_rate != audio_info.samplerate:
        raise invalid_audio()
    sf.write(destination, samples, sample_rate, format="WAV", subtype="PCM_16")
    return duration_seconds
```

Import/re-export the existing constants from `voice_registry.py` so current imports remain compatible. Replace its inline decode/write with `normalize_reference_audio`, retain fsync and atomic publication, normalize the transcript before metadata creation, and accept `str | None` while loading either a string or JSON null.

Update the adapter protocol, observed adapter, fake adapters, and `_QwenModel.create_voice_clone_prompt` protocol to accept `str | None`. Implement:

```python
return self._with_oom_conversion(
    qwen_model.create_voice_clone_prompt,
    ref_audio=str(audio_path),
    ref_text=transcript,
    x_vector_only_mode=transcript is None,
)
```

Change the existing registration form field to accept omission and blank values, then pass the
normalized value into the registry:

```python
transcript: Annotated[str | None, Form(max_length=4000)] = None
```

Place this defaulted parameter after the required form parameters. Keep `name`, `audio`,
`language`, and `consent_confirmed` requirements unchanged. Add integration
cases for omitted, empty, whitespace-only, and non-empty registration through
`/internal/v1/voices`; the public voice response must continue to omit transcript and paths.

- [ ] **Step 6: Run focused tests and type/lint checks**

Run:

```bash
uv run --no-sync pytest -q tests/unit/test_voice_registry.py tests/unit/test_real_adapter_mapping.py tests/unit/test_qwen_manager.py tests/integration/test_qwen_app.py
uv run --no-sync ruff check src/tts_playground/qwen tests/unit/test_voice_registry.py tests/unit/test_real_adapter_mapping.py
uv run --no-sync mypy
```

Expected: all selected tests pass; Ruff and mypy report no issues.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/tts_playground/qwen/reference_audio.py src/tts_playground/qwen/voice_registry.py src/tts_playground/qwen/adapter.py src/tts_playground/qwen/real_adapter.py src/tts_playground/qwen/app.py tests/unit/test_voice_registry.py tests/unit/test_real_adapter_mapping.py tests/unit/test_qwen_manager.py tests/integration/test_qwen_app.py
git commit -m "feat: make clone reference transcripts optional"
```

### Task 2: Request-local clone operation in the Qwen manager

**Files:**
- Modify: `src/tts_playground/qwen/manager.py`
- Test: `tests/unit/test_qwen_manager.py`

**Interfaces:**
- Consumes: `QwenAdapter.build_clone_prompt(model, audio_path, transcript: str | None)` from Task 1.
- Produces: `QwenModelManager.synthesize_ephemeral_clone(request, audio_path, transcript) -> SynthesisResult`.
- Preserves: `QwenModelManager.synthesize(request)` and registered prompt caching.

- [ ] **Step 1: Add failing request-local prompt and no-cache tests**

Extend the fake adapter to record `(audio_path, transcript)` build calls. Add tests that invoke the new operation twice with the same path and assert two prompt builds, no registry lookup, the Qwen Base model, and two normal syntheses.

```python
async def test_ephemeral_clone_builds_request_local_prompt_without_cache(
    manager: QwenModelManager,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"normalized")
    request = InternalSpeechRequest(
        model=ModelID.QWEN_CLONE,
        text="새 문장",
        language="Korean",
        voice=None,
        instructions=None,
    )

    await manager.synthesize_ephemeral_clone(request, reference, None)
    await manager.synthesize_ephemeral_clone(request, reference, None)

    assert fake_adapter.build_calls == [(reference, None), (reference, None)]
```

Also add an OOM-retry case proving the local prompt is rebuilt after unload, a cancellation case proving blocking GPU work finishes before cancellation escapes, and a gate case proving ephemeral and registered synthesis share one active slot.

- [ ] **Step 2: Run focused manager tests and verify RED**

Run: `uv run --no-sync pytest -q tests/unit/test_qwen_manager.py`

Expected: FAIL because `synthesize_ephemeral_clone` does not exist.

- [ ] **Step 3: Implement the transient source through the existing lifecycle**

Add an immutable private source type and thread it through `_synthesize_locked` and `_clone_prompt`:

```python
@dataclass(frozen=True, slots=True)
class _EphemeralCloneSource:
    audio_path: Path
    transcript: str | None


async def synthesize_ephemeral_clone(
    self,
    request: InternalSpeechRequest,
    audio_path: Path,
    transcript: str | None,
) -> SynthesisResult:
    if request.model != ModelID.QWEN_CLONE or request.voice is not None:
        raise ValueError("Ephemeral clone requires Qwen clone without a voice ID")
    source = _EphemeralCloneSource(audio_path, transcript)
    async with self._gate.slot():
        checkpoint = _CancellationCheckpoint()
        return await self._synthesize_locked(request, checkpoint, source)
```

When a source is present, `_clone_prompt` calls `build_clone_prompt` directly and never reads or writes `_clone_prompts`. When absent, preserve the registered lookup/cache path verbatim. Keep the source alive through both OOM attempts and let `_CancellationCheckpoint` finish blocking work before returning or raising.

- [ ] **Step 4: Run manager and existing lifecycle tests**

Run:

```bash
uv run --no-sync pytest -q tests/unit/test_qwen_manager.py tests/unit/test_qwen_gate.py
uv run --no-sync mypy
```

Expected: all tests pass and mypy is clean.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/tts_playground/qwen/manager.py tests/unit/test_qwen_manager.py
git commit -m "feat: add transient Qwen clone lifecycle"
```

### Task 3: Private multipart clone endpoint and temporary-file cleanup

**Files:**
- Modify: `src/tts_playground/common/models.py`
- Modify: `src/tts_playground/qwen/app.py`
- Test: `tests/integration/test_qwen_app.py`
- Test: `tests/unit/test_language_mapping.py`

**Interfaces:**
- Consumes: Task 1 reference normalization and Task 2 `synthesize_ephemeral_clone`.
- Produces: private `POST /internal/v1/speech/clone` multipart endpoint.
- Produces: shared `QWEN_LANGUAGES: dict[str, str]` mapping for gateway and worker.

- [ ] **Step 1: Add failing endpoint contract tests**

Add integration tests for:

- Valid audio with transcript: 200 WAV and adapter build call `(temporary_path, "exact text")`.
- Valid audio with omitted, empty, and whitespace transcript: 200 WAV and `None` build argument.
- Default `language=auto` reaches inference as `Auto`.
- Unsupported language, missing/false consent, invalid response format, invalid audio, and oversized audio return controlled 4xx envelopes before adapter build.
- The temporary path has no client filename, exists during prompt build, and is absent after success.
- The temporary path is absent after adapter build failure, synthesis failure, and cancelled request.
- Registry list and on-disk registry directory remain unchanged.
- Input, transcript, filename, and injected exception markers do not appear in public responses, caplog, or captured stderr/stdout.

Use a blocking fake adapter plus an async client task for cancellation so the assertion proves the file remains until the blocking call finishes and is removed before cancellation reaches the caller.

- [ ] **Step 2: Run the Qwen app tests and verify RED**

Run: `uv run --no-sync pytest -q tests/integration/test_qwen_app.py`

Expected: FAIL with 404 for `/internal/v1/speech/clone`.

- [ ] **Step 3: Centralize Qwen language names**

Move the current exact gateway map into `common.models`:

```python
QWEN_LANGUAGES: Final = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
```

Import it from both worker and gateway while preserving the existing public import in
`gateway.backends` for compatibility with current tests.

- [ ] **Step 4: Implement private multipart validation and request-scoped cleanup**

Add a route with explicit form constraints and blank normalization:

```python
@app.post("/internal/v1/speech/clone")
async def ephemeral_clone(
    request: Request,
    input: Annotated[str, Form(min_length=1, max_length=2000)],
    audio: Annotated[UploadFile, File()],
    consent_confirmed: Annotated[bool, Form()],
    transcript: Annotated[str | None, Form(max_length=4000)] = None,
    language: Annotated[str, Form(min_length=2, max_length=10)] = "auto",
    response_format: Annotated[Literal["wav"], Form()] = "wav",
) -> Response:
    if not consent_confirmed:
        raise ServiceError(422, "consent_required", "Consent confirmation is required.")
    qwen_language = QWEN_LANGUAGES.get(language)
    if qwen_language is None:
        raise ServiceError(422, "unsupported_language", "Qwen does not support this language.")
    audio_bytes = await _read_bounded_upload(audio)
    normalized_transcript = normalize_optional_transcript(transcript)
    with tempfile.TemporaryDirectory(prefix="qwen-clone-", dir="/tmp") as directory:
        audio_path = Path(directory) / "reference.wav"
        await asyncio.to_thread(normalize_reference_audio, audio_bytes, audio_path)
        internal = InternalSpeechRequest(
            model=ModelID.QWEN_CLONE,
            text=input,
            language=qwen_language,
            voice=None,
            instructions=None,
        )
        result = await _manager(request).synthesize_ephemeral_clone(
            internal, audio_path, normalized_transcript
        )
        return _synthesis_response(result, ModelID.QWEN_CLONE, started_at)
```

Factor common Qwen metrics/headers from the JSON speech route into a helper so both routes record
success, stable errors, and `internal_error` consistently. Include private `X-Input-Chars` with a
bounded decimal value for gateway telemetry. Do not echo filenames, transcript, language errors,
or model exception strings.

The `TemporaryDirectory` scope must surround the complete awaited manager call; do not schedule
cleanup in a background task. The manager's cancellation checkpoint guarantees the blocking GPU
call stops using the path before this scope exits.

- [ ] **Step 5: Run worker endpoint, language, privacy, and cancellation tests**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_qwen_app.py tests/unit/test_language_mapping.py tests/unit/test_qwen_manager.py tests/unit/test_voice_registry.py
uv run --no-sync ruff check src/tts_playground/common/models.py src/tts_playground/qwen tests/integration/test_qwen_app.py
uv run --no-sync mypy
```

Expected: all selected tests pass; Ruff and mypy are clean.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/tts_playground/common/models.py src/tts_playground/qwen/app.py tests/integration/test_qwen_app.py tests/unit/test_language_mapping.py
git commit -m "feat: add private ephemeral clone endpoint"
```

### Task 4: Public gateway endpoint, streaming proxy, discovery, and telemetry

**Files:**
- Modify: `src/tts_playground/gateway/app.py`
- Modify: `src/tts_playground/gateway/backends.py`
- Modify: `src/tts_playground/gateway/routes.py`
- Test: `tests/integration/test_gateway_speech.py`
- Test: `tests/integration/test_gateway_contract.py`
- Test: `tests/integration/test_gateway_management.py`

**Interfaces:**
- Produces: authenticated public `POST /v1/audio/speech/clone`.
- Produces: `BackendRouter.synthesize_ephemeral(...) -> AudioResult`.
- Changes: `AudioResult` carries an optional bounded private `input_chars` value for safe logging;
  it is never exposed as a public response header.
- Extends Qwen clone discovery capabilities with `registered_reference`, `ephemeral_reference`, and `reference_text_optional`.

- [ ] **Step 1: Add failing public proxy and authentication tests**

Add gateway tests proving:

- Missing/bad bearer token returns 401 before reading the multipart stream or calling Qwen.
- Valid multipart is forwarded once to `/internal/v1/speech/clone` with request ID and content type,
  and returns valid WAV plus the existing public synthesis headers.
- Content-Length over `max_upload_bytes + 1 MiB`, chunked audio over 20 MB, oversized non-audio
  multipart, malformed boundary/header count/header size, and missing final boundary return stable
  controlled errors.
- A worker 4xx/5xx code is preserved but message and worker request ID are sanitized.
- Timeout/unavailable worker increments Qwen backend-error metrics.
- Successful and failed ephemeral calls increment Qwen clone request metrics and privacy-safe logs;
  request input/transcript/audio/filename markers remain absent.
- The JSON body limiter still applies only to exact `/v1/audio/speech` and does not pre-buffer the
  multipart clone request.

- [ ] **Step 2: Run gateway tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py tests/integration/test_gateway_management.py
```

Expected: FAIL because the public endpoint and backend method do not exist.

- [ ] **Step 3: Reuse the bounded multipart proxy for registration and clone synthesis**

Extract the current registration streaming block into a private method that returns the worker
response only after `_MultipartAudioLimiter.finalize()` succeeds:

```python
async def _post_bounded_multipart(
    self,
    *,
    path: str,
    content: AsyncIterator[bytes],
    content_type: str | None,
    content_length: int | None,
    request_id: str,
) -> httpx.Response:
    maximum_request_bytes = self._settings.max_upload_bytes + _MAX_MULTIPART_NON_AUDIO_BYTES
    limiter = _MultipartAudioLimiter(
        content_type,
        maximum_audio_bytes=self._settings.max_upload_bytes,
        maximum_request_bytes=maximum_request_bytes,
    )
    if content_length is not None and content_length > maximum_request_bytes:
        raise _multipart_request_too_large()

    async def bounded_content() -> AsyncIterator[bytes]:
        async for chunk in content:
            limiter.write(chunk)
            yield chunk
        limiter.finalize()

    return await self._request(
        "POST",
        f"{self._settings.qwen_url.rstrip('/')}{path}",
        request_id=request_id,
        timeout_seconds=self._settings.qwen_timeout_seconds,
        headers={"Content-Type": limiter.content_type},
        content=bounded_content(),
    )
```

Preserve valid Content-Length when present. Use the helper for existing registration and new
`synthesize_ephemeral`. Validate worker success as RIFF/WAVE, sanitize failures through
`_raise_upstream_error`, normalize timing/duration headers, and parse only bounded decimal
`X-Input-Chars` from the private worker.

- [ ] **Step 4: Add the public route and shared synthesis response accounting**

Extend `SynthesisRouter` and `_RuntimeBackendRouter` with the streaming method. Add an authenticated
route that passes `request.stream()`, content headers, and request ID without invoking FastAPI's
multipart parser in the gateway. Factor the existing response/metrics/logging code so both JSON and
multipart calls use the same safe path; the ephemeral model is always
`qwen3-tts-1.7b-clone` and its log `input_chars` comes only from the bounded private numeric header.

Do not add the multipart route to `_SynthesisBodyLimitMiddleware`; `_MultipartAudioLimiter` is its
streaming body limit.

- [ ] **Step 5: Advertise both clone reference modes**

Extend `_QWEN_CLONE_CAPABILITIES` without changing the existing fields:

```python
_QWEN_CLONE_CAPABILITIES = {
    "instructions": False,
    "speed": False,
    "steps": False,
    "voice": True,
    "voice_cloning": True,
    "voice_design": False,
    "registered_reference": True,
    "ephemeral_reference": True,
    "reference_text_optional": True,
}
```

Update the capability type annotation from `dict[str, bool]` only where required; all values remain
booleans.

- [ ] **Step 6: Run all gateway and cross-boundary tests**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py tests/integration/test_gateway_management.py tests/integration/test_qwen_app.py
uv run --no-sync ruff check src/tts_playground/gateway tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py tests/integration/test_gateway_management.py
uv run --no-sync mypy
```

Expected: all tests pass; Ruff and mypy are clean.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/tts_playground/gateway/app.py src/tts_playground/gateway/backends.py src/tts_playground/gateway/routes.py tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py tests/integration/test_gateway_management.py
git commit -m "feat: expose ephemeral voice clone API"
```

### Task 5: Documentation, full regression, Docker rebuild, and live acceptance

**Files:**
- Modify: `README.md`
- Modify: `tests/deployment/test_operator_docs.py`
- Modify if required by implementation: `compose.yaml`
- Modify if required by implementation: `tests/deployment/test_compose_config.py`

**Interfaces:**
- Documents both public clone workflows and exact persistence behavior.
- Verifies the complete feature against rebuilt gateway and Qwen images without changing llama.cpp.

- [ ] **Step 1: Add failing operator-documentation assertions**

Require README to contain the new endpoint, transcript-optional behavior, both clone modes,
`x_vector_only_mode` quality guidance in user-facing language, registered persistence/delete
semantics, ephemeral `/tmp` cleanup semantics, all audio limits, and four curl examples.

```python
def test_docs_cover_registered_and_ephemeral_clone_lifecycles() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for required in (
        "POST /v1/audio/speech/clone",
        "transcript",
        "speaker-embedding-only",
        "text-assisted",
        "not persisted",
        "DELETE /v1/audio/voices/{voice_id}",
    ):
        assert required in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `uv run --no-sync pytest -q tests/deployment/test_operator_docs.py`

Expected: FAIL because the ephemeral endpoint and optional transcript are not documented.

- [ ] **Step 3: Update README examples and operational guidance**

Document these exact request families:

```bash
# Persistent registration, text-assisted
curl -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'name=approved-reference' -F 'language=ko' \
  -F 'transcript=정확한 기준 음성 녹취' -F 'consent_confirmed=true' \
  -F 'audio=@reference.wav;type=audio/wav' \
  "$TTS_BASE_URL/v1/audio/voices"

# Persistent registration, speaker embedding only: omit transcript
# Ephemeral clone, text-assisted: include transcript
# Ephemeral clone, speaker embedding only: omit transcript
curl -H "Authorization: Bearer $TTS_API_KEY" \
  -F 'input=이번 요청에서만 이 목소리를 사용합니다.' -F 'language=ko' \
  -F 'consent_confirmed=true' -F 'audio=@reference.wav;type=audio/wav' \
  -o ephemeral-clone.wav \
  "$TTS_BASE_URL/v1/audio/speech/clone"
```

State explicitly that registered references persist until DELETE, while ephemeral references are
never placed in the registry or data volume and are removed from `/tmp` after every exit path.

- [ ] **Step 4: Run the complete software gate**

Run:

```bash
uv sync --frozen --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src scripts tests docker/supertonic_privacy/sitecustomize.py
uv run --no-sync mypy
uv lock --check
git diff --check
docker compose --env-file .env.example config --quiet
```

Expected: all tests pass; Ruff, mypy, lock, diff, and Compose checks exit 0.

- [ ] **Step 5: Record the protected llama baseline**

Record the llama-server PID, `GET /health`, and per-process VRAM before rebuilding TTS. Do not send
signals or Compose commands to that process.

```bash
pgrep -af llama-server
curl --fail --silent http://127.0.0.1:8090/health
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

Expected: the established llama process is healthy; retain exact values for after-comparison.

- [ ] **Step 6: Rebuild and recreate only affected TTS services**

From the feature worktree, copy the ignored `.env` and API-key file from the main checkout if they
are not already present, without printing their contents. Then run:

```bash
docker compose -p tts-implementation --env-file .env build tts-gateway qwen-worker
docker compose -p tts-implementation --env-file .env up -d --force-recreate --wait --wait-timeout 180 qwen-worker tts-gateway
```

Do not recreate Supertonic unless Compose requires it as a dependency, and never operate on the
separate llama process.

- [ ] **Step 7: Run four live clone acceptance cases**

Using consented reference audio and privacy-safe synthetic text, verify valid RIFF/WAVE responses
for registered-with-transcript, registered-without-transcript, ephemeral-with-transcript, and
ephemeral-without-transcript. Before and after ephemeral requests, compare:

```bash
find "$TTS_DATA_DIR/voices" -maxdepth 2 -type f -printf '%P\n' | sort
docker exec tts-implementation-qwen-worker-1 sh -lc 'find /tmp -maxdepth 2 -name "qwen-clone-*" -print'
```

Expected: only explicitly registered voices change the registry listing; no ephemeral temp path
remains after either request. Validate `Content-Type: audio/wav`, RIFF/WAVE bytes, request ID, model
header, and nonnegative timing/duration headers.

- [ ] **Step 8: Run live failure/privacy/isolation checks**

Exercise unauthenticated multipart, false consent, omitted audio, malformed audio, unsupported
language, oversized Content-Length, oversized chunked audio, and a marker-bearing model failure in
a safe in-container fake-adapter process. Confirm controlled errors and zero input, transcript,
filename, exception, or API-key markers in TTS container logs.

Verify only `0.0.0.0:8020` is published, both workers have no host ports, all three TTS containers
are healthy, and normal Supertonic synthesis still succeeds.

- [ ] **Step 9: Recheck llama and disk state**

Repeat the exact baseline commands. Expected: identical llama PID, healthy response, and unchanged
process VRAM allocation. Confirm adequate `/srv` and Docker disk headroom with `df -hT` and scoped
`du -x`.

- [ ] **Step 10: Commit Task 5**

```bash
git add README.md tests/deployment/test_operator_docs.py compose.yaml tests/deployment/test_compose_config.py
git commit -m "docs: document both voice clone workflows"
```

Only add Compose files if they actually changed; do not commit `.env`, API keys, reference audio,
generated WAVs, caches, benchmark output, or live logs.

### Final review gate

- [ ] Generate a review package from the feature branch merge base through HEAD.
- [ ] Dispatch an independent whole-branch reviewer against the design and this plan.
- [ ] Fix and re-review all Critical or Important findings.
- [ ] Re-run the complete software gate after the final fix wave.
- [ ] Use `superpowers:finishing-a-development-branch` to merge, push/PR, keep, or discard according to the user's choice.
