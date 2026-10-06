#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Separate saves preserve the game running from main during QA.
export SANTONI_DATA_DIR="${SANTONI_DATA_DIR:-$PWD/.preview-data/all-powers}"
export SANTONI_PREVIEW_LABEL="Test des 55 pouvoirs"
exec ./start_native.sh "$@"
