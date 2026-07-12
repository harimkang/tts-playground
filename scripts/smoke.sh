#!/bin/sh
set -eu

if [ -z "${TTS_BASE_URL:-}" ]; then
    echo "TTS_BASE_URL is required." >&2
    exit 2
fi
if [ -z "${TTS_API_KEY:-}" ]; then
    echo "TTS_API_KEY is required." >&2
    exit 2
fi

base_url=${TTS_BASE_URL%/}
temporary_directory=$(mktemp -d)
cleanup() {
    rm -rf -- "$temporary_directory"
}
trap cleanup 0 1 2 15

authenticated_get() {
    curl --silent --show-error --fail \
        --header "Authorization: Bearer $TTS_API_KEY" \
        --output /dev/null \
        "$base_url$1"
}

verify_wav() {
    audio_file=$1
    label=$2
    response_bytes=$(wc -c < "$audio_file")
    if [ "$response_bytes" -le 44 ]; then
        echo "$label smoke response is too small to be a WAV file." >&2
        exit 1
    fi
    magic=$(dd if="$audio_file" bs=1 count=4 2>/dev/null)
    if [ "$magic" != "RIFF" ]; then
        echo "$label smoke response does not begin with RIFF." >&2
        exit 1
    fi
}

curl --silent --show-error --fail \
    --output /dev/null \
    "$base_url/healthz"
authenticated_get /readyz
authenticated_get /v1/models

supertonic_audio="$temporary_directory/supertonic.wav"
curl --silent --show-error --fail \
    --request POST \
    --header "Authorization: Bearer $TTS_API_KEY" \
    --header "Content-Type: application/json" \
    --data '{"model":"supertonic-3","input":"안녕하세요. 음성 합성 상태를 확인합니다.","voice":"F1","language":"ko","response_format":"wav","steps":8}' \
    --output "$supertonic_audio" \
    "$base_url/v1/audio/speech"
verify_wav "$supertonic_audio" "Supertonic"
echo "Supertonic smoke passed."

if [ "${RUN_QWEN_SMOKE:-0}" = "1" ]; then
    qwen_audio="$temporary_directory/qwen-custom.wav"
    curl --silent --show-error --fail \
        --request POST \
        --header "Authorization: Bearer $TTS_API_KEY" \
        --header "Content-Type: application/json" \
        --data '{"model":"qwen3-tts-1.7b-custom","input":"안녕하세요. 큐원 음성 합성 상태를 확인합니다.","voice":"Sohee","language":"ko","response_format":"wav"}' \
        --output "$qwen_audio" \
        "$base_url/v1/audio/speech"
    verify_wav "$qwen_audio" "Qwen CustomVoice"
    echo "Qwen CustomVoice smoke passed."
fi
