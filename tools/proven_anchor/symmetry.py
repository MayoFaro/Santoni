"""The 8-element dihedral symmetry group of the 5x5 no-power board.

`santorini.engine` indexes cells as `cell = y*5 + x` with `x = cell % 5`,
`y = cell // 5` (see `neighbors()` and `border()` in `santorini/engine.py`).
The no-power subgame has no gods/powers active, so every cell behaves
identically under rotation and reflection -- there is no hidden per-cell
asymmetry to worry about (confirmed by reading `Position.validate()` and
the adjacency/movement logic, which only ever consult `x`/`y` derived from
`cell` this same way).

Each transform below remaps `heights`, `domes` and `workers` -- the only
fields that encode board geometry -- and leaves every other field
(`player`, `powers`, `winner`, `reason`, `extra`, `resume`, `hero_used`,
`athena_lock`, `adonis`) untouched, since those describe game state that
has no spatial component in the no-power subgame.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from santorini.engine import Position

CellTransform = Callable[[int, int], tuple[int, int]]


def _identity(x: int, y: int) -> tuple[int, int]:
    return x, y


def _rotate90(x: int, y: int) -> tuple[int, int]:
    return y, 4 - x


def _flip(x: int, y: int) -> tuple[int, int]:
    return 4 - x, y


def _compose(f: CellTransform, g: CellTransform) -> CellTransform:
    """Return the transform that applies `f` first, then `g`."""
    def composed(x: int, y: int) -> tuple[int, int]:
        fx, fy = f(x, y)
        return g(fx, fy)
    return composed


def _remap_cell(cell: int, transform_fn: CellTransform) -> int:
    x, y = cell % 5, cell // 5
    nx, ny = transform_fn(x, y)
    return ny * 5 + nx


def _make_position_transform(transform_fn: CellTransform) -> Callable[[Position], Position]:
    def transform(position: Position) -> Position:
        new_heights = [0] * 25
        new_domes = 0
        for cell in range(25):
            new_cell = _remap_cell(cell, transform_fn)
            new_heights[new_cell] = position.heights[cell]
            if position.domes & (1 << cell):
                new_domes |= 1 << new_cell
        workers = tuple(
            tuple(_remap_cell(w, transform_fn) if w >= 0 else w for w in ws)
            for ws in position.workers
        )
        return replace(position, heights=tuple(new_heights), domes=new_domes, workers=workers)
    return transform


_rotations = (_identity, _rotate90, _compose(_rotate90, _rotate90),
              _compose(_compose(_rotate90, _rotate90), _rotate90))

_CELL_TRANSFORMS: tuple[CellTransform, ...] = _rotations + tuple(
    _compose(_flip, r) for r in _rotations
)

BOARD_SYMMETRIES: tuple[Callable[[Position], Position], ...] = tuple(
    _make_position_transform(fn) for fn in _CELL_TRANSFORMS
)
