#!/bin/sh
set -eu

TTS_UID=${TTS_UID:-1007}
TTS_GID=${TTS_GID:-1007}
TTS_HF_CACHE_DIR=${TTS_HF_CACHE_DIR:-/srv/tts-cache/huggingface}
TTS_SUPERTONIC_CACHE_DIR=${TTS_SUPERTONIC_CACHE_DIR:-/srv/tts-cache/supertonic}
TTS_DATA_DIR=${TTS_DATA_DIR:-/srv/tts-data}

case "$TTS_UID:$TTS_GID" in
    *[!0-9:]* | :* | *: | *:*:*)
        echo "TTS_UID and TTS_GID must be numeric." >&2
        exit 2
        ;;
esac

rerun_with_sudo() {
    echo "Unable to prepare directories under /srv as the current user." >&2
    echo "Rerun this script under sudo with TTS_UID and TTS_GID preserved." >&2
    exit 1
}

prepare_directory() {
    directory=$1
    if [ ! -d "$directory" ] && ! mkdir -p "$directory" 2>/dev/null; then
        rerun_with_sudo
    fi

    owner_uid=$(stat -c '%u' "$directory")
    owner_gid=$(stat -c '%g' "$directory")
    if [ "$owner_uid:$owner_gid" != "$TTS_UID:$TTS_GID" ]; then
        if [ "$(id -u)" -ne 0 ]; then
            rerun_with_sudo
        fi
        chown "$TTS_UID:$TTS_GID" "$directory"
    fi
    chmod 0750 "$directory"
}

verify_directory() {
    directory=$1
    if [ "$(id -u)" -eq "$TTS_UID" ]; then
        test -w "$directory" || {
            echo "$directory is not writable by UID $TTS_UID." >&2
            exit 1
        }
    elif [ "$(id -u)" -eq 0 ]; then
        if ! command -v setpriv >/dev/null 2>&1; then
            echo "setpriv is required to verify numeric-user access; install util-linux." >&2
            exit 1
        fi
        if ! setpriv --reuid "$TTS_UID" --regid "$TTS_GID" --clear-groups \
            test -w "$directory"; then
            echo "$directory is not writable by UID:GID $TTS_UID:$TTS_GID." >&2
            exit 1
        fi
    else
        rerun_with_sudo
    fi
}

prepare_directory "$TTS_HF_CACHE_DIR"
prepare_directory "$TTS_SUPERTONIC_CACHE_DIR"
prepare_directory "$TTS_DATA_DIR"

verify_directory "$TTS_HF_CACHE_DIR"
verify_directory "$TTS_SUPERTONIC_CACHE_DIR"
verify_directory "$TTS_DATA_DIR"

echo "Host cache and data directories are ready for UID:GID $TTS_UID:$TTS_GID."
