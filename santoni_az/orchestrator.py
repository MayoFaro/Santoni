import signal
import subprocess
import threading
import time
from pathlib import Path

from .inference_server import InferenceServer
from .trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
WORKER_BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"

# Intervalle minimal entre deux checkpoints (secondes de temps mural).
# La boucle d'entraînement tourne à 0,5 s par itération ; sauvegarder à
# chaque itération écrivait un checkpoint de plusieurs mégaoctets ~1,5 fois
# par seconde (mesure de la revue finale), le plus souvent quasi identique au
# précédent et très souvent sans qu'aucun pas de gradient n'ait eu lieu entre
# les deux. 5 s découple la cadence de sauvegarde de la cadence de scrutation
# sans allonger significativement la perte maximale en cas d'arrêt brutal :
# ce qui est en jeu, ce sont au pire les quelques pas de gradient des 5
# dernières secondes, jamais les parties terminées (écrites atomiquement par
# les workers, et réintégrées au prochain `consume_new_games`).
MIN_CHECKPOINT_INTERVAL_S = 5.0


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
        # Horodatage (temps monotone) de la dernière sauvegarde de
        # checkpoint ; `None` = aucune encore faite dans ce processus. Lu et
        # écrit uniquement par `_train_loop`.
        self._last_save = None
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
        """Boucle d'entraînement : consomme les parties, fait un pas de
        gradient, et ne sauvegarde que si quelque chose a réellement changé.

        Deux conditions, toutes deux nécessaires, pour appeler
        `save_checkpoint()` :

        1. `train_step()` a réellement entraîné. `Trainer.train_step` renvoie
           exactement `0.0` quand sa file d'exemples est vide (il sort avant
           tout calcul) ; dans tous les autres cas il renvoie la perte
           calculée. Une perte strictement nulle sur un vrai pas de gradient
           est numériquement hors d'atteinte ici (somme d'une entropie
           croisée sur 1250 classes et d'une MSE), donc `loss != 0.0` est un
           signal fiable de « un pas a eu lieu ». Sans cette condition, la
           boucle incrémentait le compteur de génération et écrivait un
           checkpoint identique au précédent même quand aucun worker n'avait
           encore livré la moindre partie.
        2. Au moins `MIN_CHECKPOINT_INTERVAL_S` depuis la dernière
           sauvegarde, pour découpler la cadence d'écriture de l'intervalle
           de scrutation de 0,5 s de cette boucle.

        La toute première sauvegarde après un pas de gradient n'est pas
        retardée (`self._last_save is None`), pour qu'un démarrage publie un
        checkpoint dès qu'il a quelque chose à publier.
        """
        while self._running:
            self.trainer.consume_new_games()
            loss = self.trainer.train_step()
            trained = loss != 0.0
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
