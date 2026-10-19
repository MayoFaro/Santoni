"""Expose `native_engine::mcts::encode_planes` (Rust) to arbitrary
`santorini.engine.Position` objects from Python.

The encoding itself only exists in Rust (`native_engine/src/mcts.rs`,
`encode_planes`) -- it's the same function the self-play worker already
uses internally to build inference requests, and this project's
zero-new-dependency rule on the Rust side means there is no FFI boundary
already wired for a single-call encode. Rather than reimplement the plane
layout a second time in Python (and risk the two silently drifting), this
module shells out to the `selfplay_worker` binary's `--encode-planes` CLI
mode, which decodes a raw `State`, calls the real `encode_planes`, and
writes the result back out -- exactly the same subprocess/binary-file
convention already used by `tools.mcts_az_vs_reference_campaign.candidate_move`
for `--single-move`.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER_BINARY = ROOT / 'native_engine' / 'target' / 'release' / 'selfplay_worker'
PLANE_BYTES = 150

sys.path.insert(0, str(ROOT))
from santorini.native import encode  # noqa: E402


@lru_cache(maxsize=8192)
def compute_planes(position: "santorini.engine.Position") -> bytes:
    """Raw `encode_planes` bytes for `position`.

    Memoized: `Position` is a frozen (hashable) dataclass and the returned
    `bytes` are immutable, so sharing one result across callers is safe --
    and worth it, since every call here costs a process spawn and the
    validation gate re-encodes the same held-out positions once per round
    per network, every round."""
    state = encode(position)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        input_path = scratch_dir / 'state.bin'
        output_path = scratch_dir / 'planes.bin'
        input_path.write_bytes(bytes(state))
        cmd = [str(WORKER_BINARY), '--encode-planes', '--input-state', str(input_path),
               '--output-planes', str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f'encode-planes worker failed: {result.stderr}')
        raw = output_path.read_bytes()
    if len(raw) != PLANE_BYTES:
        raise ValueError(f'encode-planes output malformed: {len(raw)} bytes, expected {PLANE_BYTES}')
    return raw
