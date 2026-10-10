import signal
import subprocess
import threading
import time
from pathlib import Path

from .inference_server import InferenceServer
from .trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
WORKER_BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


class Orchestrator:
    def __init__(self, run_dir, worker_count=1):
        self.run_dir = Path(run_dir)
        self.games_dir = self.run_dir / "games"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = str(self.run_dir / "infer.sock")
        self.worker_count = worker_count
        # Deliberately lower than a "real" training run might use. The
        # worker (Task 8) only checks the SIGINT-set interrupt flag between
        # games, never between individual simulations/moves (see the comment
        # above `INTERRUPTED` in native_engine/src/bin/selfplay_worker.rs),
        # and every simulation is its own request/response round trip
        # through InferenceServer (Task 7) rather than a batched call.
        # Measured empirically against the real server in this environment:
        # each round trip costs on the order of 100-200ms (dominated by
        # InferenceServer._accept_loop's 0.1s accept() poll interval, which
        # this task must not change -- Task 7 is already approved), so a
        # single game at e.g. 32 simulations/move can take 20-30s to reach
        # the next interrupt check. stop() would then have to block that
        # long before a SIGINT takes effect. Using a shallower search here
        # keeps worst-case shutdown latency low and predictable while still
        # exercising real MCTS + real inference end to end; this is a
        # process-wiring choice local to this file, not a change to any
        # approved component.
        self.simulations_per_move = 8
        self.trainer = Trainer(str(self.games_dir), str(self.checkpoint_dir))
        self._server = None
        self._server_thread = None
        self._workers = []
        self._train_thread = None
        self._running = False
        # Guards reads of self.trainer.generation / self._workers against
        # the background _train_loop thread so status() never observes a
        # torn mid-mutation state (e.g. Trainer.save_checkpoint()
        # incrementing self.generation while status() reads it).
        self._status_lock = threading.Lock()

    def start(self):
        self.trainer.load_latest_checkpoint()
        self._server = InferenceServer(self.socket_path, self.trainer.net)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        time.sleep(0.1)
        for _ in range(self.worker_count):
            proc = subprocess.Popen([
                str(WORKER_BINARY), "--games", "100000", "--out-dir", str(self.games_dir),
                "--simulations-per-move", str(self.simulations_per_move), "--inference-socket", self.socket_path,
            ])
            self._workers.append(proc)
        self._running = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()

    def _train_loop(self):
        while self._running:
            self.trainer.consume_new_games()
            self.trainer.train_step()
            with self._status_lock:
                self.trainer.save_checkpoint()
            time.sleep(0.5)

    def stop(self):
        self._running = False
        if self._train_thread is not None:
            self._train_thread.join(timeout=10)
        for proc in self._workers:
            proc.send_signal(signal.SIGINT)
        for proc in self._workers:
            # The worker only notices SIGINT between games (never mid-game,
            # see the comment on `simulations_per_move` above), and a single
            # game's worth of real inference round trips was measured at up
            # to ~28s in this environment. 60s leaves a comfortable margin
            # over that worst case rather than a tight bound that would make
            # shutdown flaky under load.
            proc.wait(timeout=60)
        if self._server:
            self._server.stop()

    def status(self):
        alive = sum(1 for p in self._workers if p.poll() is None)
        with self._status_lock:
            generation = self.trainer.generation
        return {
            "generation": generation,
            "games_available": len(list(self.games_dir.glob("game-*.json"))),
            "workers_alive": alive,
        }
