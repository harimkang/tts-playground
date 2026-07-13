# Public Clone OpenAPI and Safe Validation Details Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the exact multipart/WAV contract for `POST /v1/audio/speech/clone` and return privacy-safe field-level 422 details while preserving authenticated bounded raw streaming.

**Architecture:** OpenAPI metadata is attached manually to the public route so FastAPI never gains runtime `Form`/`File` dependencies in the gateway. The private Qwen worker produces a small shared validation-detail type, and the gateway forwards it only after strict shape, field, reason, and count validation; all other upstream text remains sanitized.

**Tech Stack:** Python 3.12, FastAPI/Starlette, HTTPX, Pydantic v2, python-multipart, pytest/pytest-asyncio/respx, Ruff, mypy, uv, Docker Compose.

## Global Constraints

- Preserve authentication-before-body-read and the existing bounded raw-streaming gateway path; do not add public `Form`, `File`, `UploadFile`, or body-model dependencies.
- Public multipart fields are exactly `input`, `audio`, `consent_confirmed`, `transcript`, `language`, and `response_format`; required fields are `input`, `audio`, and `consent_confirmed`.
- Public OpenAPI documents `additionalProperties: false`, exact string limits/defaults, binary `audio`, and binary `audio/wav` HTTP 200 without `application/json`.
- Validation reasons are exactly `required`, `invalid`, `duplicate`, and `unknown`.
- Field identifiers are at most 64 ASCII characters matching `[A-Za-z_][A-Za-z0-9_]{0,63}`; unsafe identifiers become the literal `unknown`.
- Return at most 16 validation issues and never return or log form values, input text, transcript, audio, client filenames, bearer tokens, internal paths, worker request IDs, exception text, Pydantic messages, contexts, or backend-controlled prose.
- Forward details only for a normalized upstream `validation_error`; discard the complete details payload if any entry has an invalid shape, field, or reason.
- Existing error envelopes remain backward compatible because `details` is optional; non-validation backend responses remain fixed `Backend request failed.` without details.
- Preserve voice registry, temp cleanup, cancellation, queue/GPU lifecycle, metrics, and model routing behavior.
- Keep only gateway port 8020 published and do not stop, restart, signal, or reconfigure llama.cpp.
- Manage Python only with uv, `.venv`, `pyproject.toml`, and `uv.lock`; do not introduce pip, Poetry, or Conda.

---

### Task 1: Shared safe validation details and worker production

**Files:**
- Create: `src/tts_playground/common/validation.py`
- Modify: `src/tts_playground/common/errors.py`
- Modify: `src/tts_playground/qwen/app.py`
- Test: `tests/unit/test_validation_details.py`
- Test: `tests/integration/test_qwen_app.py`

**Interfaces:**
- Produces: `ValidationIssue(field: str, reason: ValidationReason)` immutable dataclass.
- Produces: `sanitize_field_identifier(value: object) -> str`.
- Produces: `parse_validation_issues(value: object) -> tuple[ValidationIssue, ...] | None` for untrusted payloads.
- Produces: `validation_issues_to_public(issues) -> list[dict[str, str]]`.
- Changes: `ServiceError.details: tuple[ValidationIssue, ...] | None = None`.

- [ ] **Step 1: Write failing shared sanitizer tests**

Create `tests/unit/test_validation_details.py` with exact boundary and rejection cases:

```python
def test_field_identifier_keeps_only_bounded_ascii_names() -> None:
    assert sanitize_field_identifier("audio") == "audio"
    assert sanitize_field_identifier("a" * 64) == "a" * 64
    for unsafe in ("", "1audio", "a-b", "한글", "a" * 65, None, 7):
        assert sanitize_field_identifier(unsafe) == "unknown"


def test_untrusted_issue_parser_is_all_or_nothing_and_bounded() -> None:
    assert parse_validation_issues(
        [{"field": "audio", "reason": "required"}]
    ) == (ValidationIssue("audio", "required"),)
    assert parse_validation_issues(
        [{"field": "audio", "reason": "private-reason"}]
    ) is None
    assert parse_validation_issues([{"field": "audio", "reason": "required"}] * 17) is None
```

Also reject extra keys, missing keys, non-list payloads, non-string reasons, and unsafe incoming field
identifiers rather than silently repairing them in the untrusted gateway parser.

- [ ] **Step 2: Run sanitizer tests and verify RED**

Run: `uv run --no-sync pytest -q tests/unit/test_validation_details.py`

Expected: collection fails because `common.validation` does not exist.

- [ ] **Step 3: Implement the bounded shared model**

Create the focused module:

```python
import re
from dataclasses import dataclass
from typing import Final, Literal, cast

ValidationReason = Literal["required", "invalid", "duplicate", "unknown"]
MAX_VALIDATION_ISSUES: Final = 16
_FIELD_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_REASONS: Final = frozenset({"required", "invalid", "duplicate", "unknown"})


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    field: str
    reason: ValidationReason


def sanitize_field_identifier(value: object) -> str:
    return value if isinstance(value, str) and _FIELD_PATTERN.fullmatch(value) else "unknown"
```

`parse_validation_issues` accepts only a non-empty list of at most 16 dictionaries whose exact key
set is `{field, reason}`, whose field already matches the safe pattern, and whose reason is in the
four-item set. `validation_issues_to_public` serializes only those two keys. Extend `ServiceError`
after `backend_failure` with a defaulted details field so every current named or positional call
remains compatible.

- [ ] **Step 4: Add failing worker validation-detail tests**

Extend Qwen integration coverage for:

- Missing each required field returns `reason=required` with the declared field name.
- Invalid `input`, `consent_confirmed`, `language`, `response_format`, and transcript length return
  `reason=invalid` without Pydantic `msg`, `input`, or `ctx` data.
- Duplicate each allowed field returns `duplicate`.
- Known forbidden and arbitrary safe unknown names return `unknown` with the safe field identifier.
- Malicious/oversized/non-ASCII field names return field `unknown` and never appear in response,
  caplog, stdout, or stderr.
- Existing public error keys remain and non-validation errors have no `details`.

Example assertion:

```python
assert response.json() == {
    "error": {
        "code": "validation_error",
        "message": "Request validation failed.",
        "request_id": request_id,
        "details": [{"field": "audio", "reason": "required"}],
    }
}
```

- [ ] **Step 5: Run worker tests and verify RED**

Run: `uv run --no-sync pytest -q tests/integration/test_qwen_app.py -k 'validation_detail or duplicate_singular or unknown_fields'`

Expected: FAIL because worker envelopes contain no details.

- [ ] **Step 6: Produce safe worker details**

Update the Qwen `ServiceError` and `RequestValidationError` handlers to pass optional details into
`_error_response`, which conditionally inserts `details` only when present.

For `RequestValidationError`, inspect only `error.errors()` locations and types when
`request.url.path == "/internal/v1/speech/clone"`:

```python
def _request_validation_issues(error: RequestValidationError) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for item in error.errors()[:MAX_VALIDATION_ISSUES]:
        location = item.get("loc")
        field = location[-1] if isinstance(location, tuple | list) and location else "unknown"
        reason: ValidationReason = "required" if item.get("type") == "missing" else "invalid"
        issues.append(ValidationIssue(sanitize_field_identifier(field), reason))
    return tuple(issues)
```

Do not read `msg`, `input`, `ctx`, or exception objects. For all other private worker paths retain
the existing fixed envelope without details.

Change `_validate_ephemeral_clone_fields` to count cached `request.form().multi_items()`, emit one
unknown issue per distinct non-allowlisted name and one duplicate issue per repeated allowed name,
cap deterministically at 16, and raise one fixed `ServiceError(validation_error, details=...)` before
normalization/manager work.

- [ ] **Step 7: Run Task 1 verification**

Run:

```bash
uv run --no-sync pytest -q tests/unit/test_validation_details.py tests/integration/test_qwen_app.py tests/unit/test_qwen_manager.py
uv run --no-sync ruff check src/tts_playground/common src/tts_playground/qwen tests/unit/test_validation_details.py tests/integration/test_qwen_app.py
uv run --no-sync mypy
```

Expected: selected tests pass; Ruff and mypy are clean.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/tts_playground/common/validation.py src/tts_playground/common/errors.py src/tts_playground/qwen/app.py tests/unit/test_validation_details.py tests/integration/test_qwen_app.py
git commit -m "feat: add safe clone validation details"
```

### Task 2: Gateway validation-detail allowlist and public propagation

**Files:**
- Modify: `src/tts_playground/gateway/app.py`
- Modify: `src/tts_playground/gateway/backends.py`
- Test: `tests/integration/test_gateway_speech.py`
- Test: `tests/integration/test_gateway_contract.py`

**Interfaces:**
- Consumes: `ServiceError.details` and `parse_validation_issues` from Task 1.
- Produces: optional public `error.details` only for validated upstream `validation_error`.
- Preserves: fixed message/no-details behavior for every other upstream error.

- [ ] **Step 1: Add failing real-boundary detail tests**

Extend the existing in-process public gateway-to-Qwen tests to assert exact safe details for missing,
invalid, duplicate, forbidden, and unknown fields. The gateway must replace the worker request ID
with the caller's public ID and use `Request validation failed.` rather than `Backend request
failed.` only when safe validation details survive.

Add respx cases where a fake worker sends:

- valid details on `validation_error` — forwarded;
- valid details on another error code — discarded;
- 17 issues, bad reason, unsafe field, extra keys, non-list details — discarded in full;
- marker-bearing message/value/context/request ID — absent from response and logs.

- [ ] **Step 2: Run gateway tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py -k 'validation_detail or worker_validation or malicious_detail'
```

Expected: FAIL because `_raise_upstream_error` discards all details and always returns `Backend request failed.`.

- [ ] **Step 3: Strictly parse details at the backend boundary**

In `_raise_upstream_error`, normalize the code first. Only for `validation_error`, call
`parse_validation_issues(error.get("details"))`. When it returns a non-empty tuple, raise:

```python
raise ServiceError(
    status_code=response.status_code,
    code="validation_error",
    message="Request validation failed.",
    backend_failure=True,
    details=details,
)
```

For missing/malformed details and every other error code, preserve the current fixed message and no
details. Do not forward the worker message, request ID, or any unrecognized key.

Update the gateway ServiceError handler and `_error_response` to serialize only typed details via
`validation_issues_to_public`. Keep details out of metrics/log calls.

- [ ] **Step 4: Run gateway and privacy verification**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py tests/integration/test_qwen_app.py
uv run --no-sync ruff check src/tts_playground/gateway tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py
uv run --no-sync mypy
```

Expected: tests pass; Ruff and mypy are clean.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/tts_playground/gateway/app.py src/tts_playground/gateway/backends.py tests/integration/test_gateway_speech.py tests/integration/test_gateway_contract.py
git commit -m "feat: expose safe clone field errors"
```

### Task 3: Accurate public multipart and WAV OpenAPI

**Files:**
- Create: `src/tts_playground/gateway/openapi.py`
- Modify: `src/tts_playground/gateway/app.py`
- Modify: `src/tts_playground/gateway/schemas.py`
- Test: `tests/integration/test_gateway_contract.py`
- Test: `tests/deployment/test_operator_docs.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `EPHEMERAL_CLONE_OPENAPI_EXTRA` request-body metadata.
- Produces: `EPHEMERAL_CLONE_RESPONSES` with binary `audio/wav` 200 and typed error envelopes.
- Preserves: public route runtime signature `(request: Request, authentication dependency)`.

- [ ] **Step 1: Add failing OpenAPI contract tests**

Fetch `/openapi.json` from the gateway app and assert:

```python
operation = schema["paths"]["/v1/audio/speech/clone"]["post"]
multipart = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
assert multipart["additionalProperties"] is False
assert multipart["required"] == ["input", "audio", "consent_confirmed"]
assert set(multipart["properties"]) == {
    "input", "audio", "consent_confirmed", "transcript", "language", "response_format"
}
assert multipart["properties"]["audio"] == {"type": "string", "format": "binary"}
assert set(operation["responses"]["200"]["content"]) == {"audio/wav"}
assert "application/json" not in operation["responses"]["200"]["content"]
```

Assert every exact min/max/default/enum and that 401/413/422/500 responses reference the public
error envelope with optional details. Assert the 200 response documents `X-Request-ID`,
`X-TTS-Model`, `X-Audio-Duration`, `X-Queue-Ms`, and `X-Processing-Ms` with bounded string schemas.
Inspect the route dependency metadata or run the existing raw ASGI auth-before-read test to prove
no runtime body parameter was added.

- [ ] **Step 2: Run OpenAPI tests and verify RED**

Run: `uv run --no-sync pytest -q tests/integration/test_gateway_contract.py -k openapi`

Expected: FAIL because public `requestBody` is absent and 200 advertises JSON.

- [ ] **Step 3: Define documentation-only schemas**

Create `gateway/openapi.py` constants containing the exact inline multipart schema from the design.
Define Pydantic documentation models in `gateway/schemas.py`:

```python
class ValidationDetail(BaseModel):
    field: str
    reason: Literal["required", "invalid", "duplicate", "unknown"]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[ValidationDetail] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
```

Attach metadata without runtime parsing:

```python
@app.post(
    "/v1/audio/speech/clone",
    response_class=Response,
    responses=EPHEMERAL_CLONE_RESPONSES,
    openapi_extra=EPHEMERAL_CLONE_OPENAPI_EXTRA,
)
async def ephemeral_clone(
    request: Request,
    _authenticated: Annotated[None, Depends(authenticate)],
) -> Response:
    ...
```

Use `ErrorEnvelope` only as documentation metadata. Confirm generated OpenAPI, rather than assuming
FastAPI merge behavior, because `response_class` and `responses` can otherwise retain a default
JSON content entry.

- [ ] **Step 4: Document the discoverable contract**

Update README to point clients at `/docs` and `/openapi.json`, list the exact multipart fields, and
show the safe `details` shape. State that values and backend text are intentionally omitted.
Extend operator-doc tests for these exact fragments.

- [ ] **Step 5: Run Task 3 verification**

Run:

```bash
uv run --no-sync pytest -q tests/integration/test_gateway_contract.py tests/deployment/test_operator_docs.py tests/integration/test_gateway_speech.py
uv run --no-sync ruff check src/tts_playground/gateway tests/integration/test_gateway_contract.py tests/deployment/test_operator_docs.py README.md
uv run --no-sync mypy
```

If Ruff does not accept Markdown, omit only `README.md` from the Ruff invocation; `git diff --check`
still validates it. Expected: tests pass; generated OpenAPI assertions, Ruff, mypy, and diff check are clean.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/tts_playground/gateway/openapi.py src/tts_playground/gateway/app.py src/tts_playground/gateway/schemas.py tests/integration/test_gateway_contract.py tests/deployment/test_operator_docs.py README.md
git commit -m "docs: publish clone multipart OpenAPI"
```

### Task 4: Full regression, Docker rebuild, and live acceptance

**Files:**
- Modify only if a discovered regression requires it: files already in Tasks 1-3.
- Test evidence/report remains ignored under `.superpowers/sdd/`.

**Interfaces:**
- Verifies the exact committed public OpenAPI and safe worker-to-gateway detail contract in rebuilt containers.

- [ ] **Step 1: Run the complete software gate**

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

Expected: all tests pass and every static/config command exits 0.

- [ ] **Step 2: Record protected llama baseline**

Read only:

```bash
pgrep -af llama-server
curl --fail --silent http://127.0.0.1:8090/health
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

Record the exact established PID, `ok` health, and process VRAM. Never send it a signal or include
it in Compose commands.

- [ ] **Step 3: Rebuild and recreate only gateway and Qwen**

Use the existing ignored worktree `.env` and secret without printing them:

```bash
docker compose -p tts-implementation --env-file .env build tts-gateway qwen-worker
docker compose -p tts-implementation --env-file .env up -d --force-recreate --no-deps --wait --wait-timeout 180 qwen-worker tts-gateway
```

Expected: gateway/Qwen healthy; Supertonic container not recreated.

- [ ] **Step 4: Verify live public OpenAPI**

Fetch `http://127.0.0.1:8020/openapi.json` and use `jq` to assert exact multipart required/properties,
`additionalProperties=false`, binary audio, constraints/defaults, binary `audio/wav` 200, and no
200 JSON content.

- [ ] **Step 5: Verify live safe 422 details**

Through public port 8020, test missing audio, duplicate input, a safe unknown field, and a malicious
field identifier. Assert status 422, public request ID, fixed message, correct bounded details,
absence of every marker/value/backend request ID, unchanged registry, and no `qwen-clone-*` temp
residue.

- [ ] **Step 6: Verify valid and compatibility paths**

Using the existing consent-safe synthetic reference, verify one normal ephemeral clone returns a
non-empty RIFF/WAVE with safe headers and no registry/temp residue. Verify existing JSON Supertonic
synthesis and persistent voice management remain functional. Confirm all three containers healthy,
only 8020 published, and workers private.

- [ ] **Step 7: Verify privacy and llama state**

Scan gateway/Qwen/Supertonic logs since the marked requests for field values, input, transcript,
filename, API key, worker request ID, Pydantic text, and exception markers; expect zero matches.
Repeat the exact llama and disk checks; PID, health, and VRAM must match baseline.

- [ ] **Step 8: Run a final fresh gate and commit any required Task 4 fix**

If live validation found no tracked defect, create no empty commit. If a defect required a TDD fix,
commit only that fix after rerunning its focused tests and the complete software gate. Never commit
`.env`, keys, WAVs, audio, logs, caches, reports, or live scripts.

### Final review gate

- [ ] Generate a merge-base-to-HEAD review package.
- [ ] Dispatch an independent whole-branch reviewer against the design, plan, and live evidence.
- [ ] Fix and re-review every Critical or Important finding in one final fix wave.
- [ ] Run the complete software gate on exact final HEAD.
- [ ] Use `superpowers:finishing-a-development-branch` for merge/push/keep/discard.
