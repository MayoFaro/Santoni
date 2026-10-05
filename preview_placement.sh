#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Separate saves protect the production game during QA of this branch.
export SANTONI_DATA_DIR="${SANTONI_DATA_DIR:-$PWD/.preview-data}"
export SANTONI_PREVIEW_LABEL="Test placement"
exec ./start_native.sh "$@"
