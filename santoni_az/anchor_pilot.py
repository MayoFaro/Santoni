"""Live anchor pilot orchestrator: runs the training+eval loop of a
candidate MCTS+NN network that learns by playing against a live reference
engine, starting every game from one fixed anchor position (Tasks 5/6).

`AnchorPilotOrchestrator` is deliberately a thin composition over the
exact same `InferenceServer`/`Trainer` pair `santoni_az.orchestrator
.Orchestrator` already uses -- see that class, which this one mirrors
closely -- plus two pools of the compiled `selfplay_worker --vs-reference`
binary instead of its one plain self-play pool:

- a training pool (no `--deterministic`) writing into `run_dir/games`,
  the only directory `self.trainer` is ever pointed at;
- an eval pool (`--deterministic`, for reproducible eval games) writing
  into `run_dir/eval_games`, a directory `Trainer` never reads.

This directory split is the entire mechanism that keeps eval data from
leaking into training (spec Sec 3.3): no filtering logic is needed anywhere
because `Trainer` structurally only ever looks at `games/`.
"""
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

from .inference_server import InferenceServer
from .stats import clopper_pearson_lower_bound
from .trainer import Trainer

# Mirrors `santoni_az.orchestrator.MIN_CHECKPOINT_INTERVAL_S` exactly: same
# train loop, same rationale (decouple checkpoint-write cadence from the
# 0.5s polling cadence of that loop without meaningfully widening the
# worst-case loss on a hard stop -- completed games are never at risk,
# only the last few seconds of gradient steps).
MIN_CHECKPOINT_INTERVAL_S = 5.0

# Number of most-recently-modified eval games `status()` reads to compute
# `eval_ci_lower`. A fixed-size recent window (same `stat().st_mtime` sort
# `Trainer._rebuild_buffer_from_disk` already uses for its replay buffer)
# rather than every eval game ever played: the win rate should track the
# candidate's *current* strength against the reference, not be diluted by
# games played against a much weaker, long-superseded checkpoint earlier in
# the run.
EVAL_WINDOW = 40


class AnchorPilotOrchestrator:
    def __init__(self, run_dir, anchor_state_path, worker_binary_path,
                 train_worker_count, eval_worker_count, simulations_per_move=32,
                 reference_seconds=5.0):
        self.run_dir = Path(run_dir)
        self.anchor_state_path = Path(anchor_state_path)
        self.worker_binary_path = Path(worker_binary_path)
        self.games_dir = self.run_dir / "games"
        self.eval_games_dir = self.run_dir / "eval_games"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.eval_games_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = str(self.run_dir / "infer.sock")
        self.train_worker_count = train_worker_count
        self.eval_worker_count = eval_worker_count
        self.simulations_per_move = simulations_per_move
        self.reference_seconds = reference_seconds
        self.trainer = Trainer(str(self.games_dir), str(self.checkpoint_dir))
        self._server = None
        self._server_thread = None
        self._train_workers = []
        self._eval_workers = []
        self._train_thread = None
        self._running = False
        # Same semantics as `Orchestrator`: `None` until the first real
        # save/gradient step happens in this process.
        self._last_save = None
        self._last_loss = None
        # Guards reads of self.trainer.generation / the worker lists against
        # the background `_train_loop` thread, exactly like `Orchestrator`.
        self._status_lock = threading.Lock()

    def _spawn_worker(self, out_dir, deterministic):
        args = [
            str(self.worker_binary_path), "--vs-reference",
            "--input-state", str(self.anchor_state_path),
            "--out-dir", str(out_dir),
            "--games", "100000",
            "--simulations-per-move", str(self.simulations_per_move),
            "--inference-socket", self.socket_path,
            "--reference-seconds", str(self.reference_seconds),
        ]
        if deterministic:
            args.append("--deterministic")
        return subprocess.Popen(args)

    def start(self):
        # This single call gives both: a brand-new `run_dir` has no
        # checkpoints, so `self.trainer.net` stays at its fresh
        # `net_factory()` initialization; a relaunched `run_dir` loads its
        # own latest checkpoint. No special-casing needed to tell the two
        # situations apart -- exactly like `Orchestrator.start()`.
        self.trainer.load_latest_checkpoint()
        self._server = InferenceServer(self.socket_path, self.trainer.net)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        time.sleep(0.1)
        for _ in range(self.train_worker_count):
            self._train_workers.append(self._spawn_worker(self.games_dir, deterministic=False))
        for _ in range(self.eval_worker_count):
            self._eval_workers.append(self._spawn_worker(self.eval_games_dir, deterministic=True))
        self._running = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()

    def _train_loop(self):
        """Verbatim reuse of `Orchestrator._train_loop`'s checkpointing
        logic: train only from `games/` (never `eval_games/`), save only
        when a real gradient step happened (`loss != 0.0`) and at least
        `MIN_CHECKPOINT_INTERVAL_S` elapsed since the last save."""
        while self._running:
            self.trainer.consume_new_games()
            loss = self.trainer.train_step()
            trained = loss != 0.0
            if trained:
                with self._status_lock:
                    self._last_loss = loss
            now = time.monotonic()
            due = self._last_save is None or now - self._last_save >= MIN_CHECKPOINT_INTERVAL_S
            if trained and due:
                with self._status_lock:
                    self.trainer.save_checkpoint()
                self._last_save = now
            time.sleep(0.5)

    def stop(self):
        self._running = False
        if self._train_thread is not None:
            self._train_thread.join(timeout=10)
        all_workers = self._train_workers + self._eval_workers
        for proc in all_workers:
            proc.send_signal(signal.SIGINT)
        for proc in all_workers:
            # Same bound as `Orchestrator.stop()`: the worker only notices
            # SIGINT between games, never mid-game.
            proc.wait(timeout=60)
        if self._server:
            self._server.stop()

    def _recent_eval_outcomes(self):
        """Read the `outcome` field of the `EVAL_WINDOW` most-recently
        -modified eval game files on disk. Same sort convention as
        `Trainer._rebuild_buffer_from_disk` (`stat().st_mtime`, since eval
        game filenames carry a per-worker-process counter that isn't
        comparable across distinct worker processes). A corrupted or
        not-yet-fully-written file (same crash/partial-write window
        `Trainer.consume_new_games` already tolerates) is skipped rather
        than failing `status()`."""
        files = sorted(
            self.eval_games_dir.glob("game-*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:EVAL_WINDOW]
        outcomes = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                outcomes.append(data["outcome"])
            except (OSError, ValueError, KeyError):
                continue
        return outcomes

    def status(self):
        # Same split as `Orchestrator.status()`: worker liveness is read
        # without the lock (the worker lists are only ever appended to,
        # once, by `start()`, never mutated by the background
        # `_train_loop` thread), only the fields `_train_loop` itself
        # mutates concurrently are read under `_status_lock`.
        train_alive = sum(1 for p in self._train_workers if p.poll() is None)
        eval_alive = sum(1 for p in self._eval_workers if p.poll() is None)
        with self._status_lock:
            generation = self.trainer.generation
            last_loss = self._last_loss
        outcomes = self._recent_eval_outcomes()
        eval_n = len(outcomes)
        eval_wins = sum(1 for o in outcomes if o == 1)
        # `clopper_pearson_lower_bound` requires `n > 0` (it raises
        # `ValueError` on `n <= 0`); until at least one eval game has
        # landed on disk there is nothing to compute a bound from, so this
        # reports `None` -- the same "nothing real yet" convention
        # `last_loss` already uses before the first real gradient step.
        eval_ci_lower = clopper_pearson_lower_bound(eval_wins, eval_n) if eval_n > 0 else None
        return {
            "generation": generation,
            "last_loss": last_loss,
            "games_available": len(list(self.games_dir.glob("game-*.json"))),
            "eval_games_available": len(list(self.eval_games_dir.glob("game-*.json"))),
            "workers_alive": train_alive + eval_alive,
            "eval_wins": eval_wins,
            "eval_n": eval_n,
            "eval_ci_lower": eval_ci_lower,
        }
