# Public Clone OpenAPI and Safe Validation Details Design

Date: 2026-07-13
Status: Approved in conversation; awaiting review of this written specification

## 1. Purpose

Make `POST /v1/audio/speech/clone` self-describing to external clients without changing its
raw-streaming runtime architecture. The public OpenAPI document must describe the exact multipart
request and WAV response, and controlled 422 responses must identify invalid fields without
exposing field values, reference text, audio, credentials, internal paths, or backend exception
text.

## 2. Root cause and constraints

The gateway route intentionally accepts only `Request` plus authentication and streams the raw
multipart body to the private Qwen worker. FastAPI therefore sees no public body parameters and
currently emits no `requestBody`; it also documents the generic `Response` incorrectly as JSON.

Adding `Form` or `File` parameters to the public gateway route is not acceptable because it would
make the gateway parse/spool the multipart upload and undermine the existing authentication-before-
body-read and bounded raw-streaming design.

The current `Backend request failed.` text proves worker-side failures reached the gateway, but the
gateway deliberately replaces all worker-controlled messages. That privacy boundary remains; only
new strictly validated field metadata may cross it.

## 3. Public OpenAPI contract

The gateway declares OpenAPI metadata separately from runtime request binding. The generated
`/openapi.json` entry for `POST /v1/audio/speech/clone` must include:

```yaml
requestBody:
  required: true
  content:
    multipart/form-data:
      schema:
        type: object
        additionalProperties: false
        required: [input, audio, consent_confirmed]
        properties:
          input:
            type: string
            minLength: 1
            maxLength: 2000
          audio:
            type: string
            format: binary
          consent_confirmed:
            type: boolean
          transcript:
            type: string
            maxLength: 4000
          language:
            type: string
            minLength: 2
            maxLength: 10
            default: auto
          response_format:
            type: string
            enum: [wav]
            default: wav
```

The documented success response is `audio/wav` with a binary string schema and the existing safe
response headers. It must not advertise `application/json` for HTTP 200. Controlled error
responses document the existing error envelope plus optional field details.

The runtime handler continues to receive `Request`, authenticate before stream iteration, and call
the existing bounded multipart proxy unchanged.

## 4. Safe field-level validation details

The controlled error envelope gains an optional `details` array only for safe validation issues:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "request_id": "req_...",
    "details": [
      {"field": "audio", "reason": "required"}
    ]
  }
}
```

Allowed reasons are exactly:

- `required`
- `invalid`
- `duplicate`
- `unknown`

Field identifiers are bounded to 64 ASCII characters matching
`[A-Za-z_][A-Za-z0-9_]{0,63}`. Invalid, oversized, or non-ASCII identifiers are replaced with the
literal `unknown`; they are never copied verbatim. At most 16 issues are returned.

Worker-side FastAPI validation is converted from `RequestValidationError.errors()` using only the
location and error type. Messages, input values, contexts, filenames, and exception objects are
discarded. Known missing fields use `required`; all other declared-field schema failures use
`invalid`.

The explicit multipart allowlist produces `duplicate` for repeated allowed fields and `unknown`
for non-allowlisted fields. It does not echo form values.

The gateway accepts details only when all of the following hold:

- The upstream error code is `validation_error`.
- The payload is a list no longer than 16.
- Every entry contains only a sanitized field identifier and one allowed reason.

Malformed or unexpected detail payloads are discarded in full. Other backend errors remain the
existing fixed `Backend request failed.` response and never carry details.

## 5. Components

### 5.1 Common error model

Extend `ServiceError` with optional immutable validation details. Existing construction sites and
responses remain backward compatible because the new field defaults to `None`.

Provide a small shared validation-detail type and sanitizer so worker production, gateway parsing,
and response rendering use one bounded contract rather than duplicating security rules.

### 5.2 Qwen worker

The private worker:

- Emits safe details for missing/invalid declared multipart fields.
- Emits safe details for unknown and duplicate fields detected by the exact allowlist.
- Keeps error messages fixed.
- Performs these checks before reference normalization, manager admission, or GPU work.

### 5.3 Gateway

The gateway:

- Adds manual OpenAPI metadata without adding runtime `Form`/`File` dependencies.
- Parses and forwards only allowlisted validation details from the private worker.
- Uses the fixed public `Request validation failed.` message when safe details are accepted.
- Preserves current status, request ID replacement, metrics, logging, body limits, and privacy
  behavior.

## 6. Testing

Regression-first coverage must prove:

- Public OpenAPI contains the exact multipart properties, required fields, constraints, defaults,
  `additionalProperties: false`, and binary `audio/wav` 200 response.
- The route signature still contains no public `Form`, `File`, or body model dependency and an
  unauthenticated request is rejected before body reads.
- Missing required fields produce safe `required` details.
- Invalid known fields produce safe `invalid` details.
- Duplicates produce `duplicate`; forbidden/unknown fields produce `unknown` without values.
- Malicious field identifiers are bounded/replaced and do not appear in responses or logs.
- Arbitrary worker messages, detail shapes, reasons, values, request IDs, and exception text are
  not forwarded.
- Existing non-validation worker errors remain unchanged.
- A valid multipart request still reaches the worker and returns WAV.
- Existing JSON synthesis, registration, metrics, cancellation, temp cleanup, and privacy suites
  remain green.

After the software gate, rebuild and recreate only gateway and Qwen, then verify live OpenAPI,
safe 422 details for missing/duplicate/unknown fields, a normal ephemeral WAV, worker isolation,
privacy markers, and unchanged llama PID/health/VRAM.

## 7. Compatibility and out of scope

This is additive for public clients: existing error envelope keys remain unchanged and `details` is
optional. Existing callers that ignore unknown JSON properties continue to work. Runtime multipart
processing, model routing, reference persistence, queueing, GPU lifecycle, and API authentication
do not change.

Out of scope:

- Echoing invalid field values or original backend messages.
- Publishing private worker endpoints.
- Moving multipart parsing into the gateway.
- Generating OpenAPI dynamically from the private worker at runtime.
- Changing the persistent voice registration OpenAPI in this task.
