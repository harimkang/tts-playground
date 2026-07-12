#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
secrets_directory="$script_dir/../secrets"
key_file="$secrets_directory/tts_api_key.txt"

if [ -e "$key_file" ]; then
    echo "Refusing to overwrite existing API key: $key_file" >&2
    exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to generate the API key." >&2
    exit 1
fi

umask 077
temporary_key_file=$(mktemp "$secrets_directory/.tts_api_key.tmp.XXXXXX")
cleanup() {
    rm -f -- "$temporary_key_file"
}
trap cleanup 0 1 2 15

openssl rand -hex 32 > "$temporary_key_file"
chmod 0600 "$temporary_key_file"

if ! ln "$temporary_key_file" "$key_file" 2>/dev/null; then
    echo "Refusing to overwrite existing API key: $key_file" >&2
    exit 1
fi

rm -f -- "$temporary_key_file"
trap - 0 1 2 15
echo "Created $key_file with mode 0600."
