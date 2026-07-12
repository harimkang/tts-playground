#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
environment_file=${1:-$repository_root/.env.example}
temporary_directory=$(mktemp -d)
project_name="tts-gateway-probe-$$"
override_file="$temporary_directory/compose.probe.yaml"
key_file="$temporary_directory/tts_api_key.txt"

cleanup() {
    docker compose \
        --project-name "$project_name" \
        --env-file "$environment_file" \
        -f "$repository_root/compose.yaml" \
        -f "$override_file" \
        down --remove-orphans >/dev/null 2>&1 || true
    rm -rf -- "$temporary_directory"
}
trap cleanup EXIT HUP INT TERM

umask 077
printf '%s\n' 'task-8-container-probe-key' > "$key_file"
cat > "$override_file" <<EOF
secrets:
  tts_api_key:
    file: $key_file
EOF

docker compose --project-name "$project_name" --env-file "$environment_file" \
    -f "$repository_root/compose.yaml" -f "$override_file" \
    run --no-deps --rm tts-gateway \
    uv run --no-sync python -c \
    'import tts_playground.gateway.app; print("gateway-hardened-import-ok")'
