#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
key_file="$script_dir/../secrets/tts_api_key.txt"

if [ -e "$key_file" ]; then
    echo "Refusing to overwrite existing API key: $key_file" >&2
    exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to generate the API key." >&2
    exit 1
fi

umask 077
openssl rand -hex 32 > "$key_file"
chmod 0600 "$key_file"
echo "Created $key_file with mode 0600."
