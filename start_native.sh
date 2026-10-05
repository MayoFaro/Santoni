#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
engine_library="native_engine/target/release/libsantoni_engine.so"
needs_build=false
if [[ ! -f "$engine_library" ]]; then needs_build=true; fi
for source_file in native_engine/src/*.rs native_engine/Cargo.toml; do
    if [[ "$source_file" -nt "$engine_library" ]]; then needs_build=true; fi
done
if $needs_build; then
    cargo_executable="${CARGO:-cargo}"
    if ! command -v "$cargo_executable" >/dev/null 2>&1; then
        cargo_executable="$HOME/.cargo/bin/cargo"
    fi
    "$cargo_executable" build --release --manifest-path native_engine/Cargo.toml
fi
exec python3 -m santorini "$@"
