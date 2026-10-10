import os
import socket
import struct
import threading
import time

import torch

from .inference_protocol import pack_response, unpack_request, PLANE_BYTES, ACTION_SPACE
from .network import NUM_PLANES


class InferenceServer:
    def __init__(self, socket_path, net, flush_interval_s=0.02):
        # Force torch's own intra-op thread pool down to a single thread.
        # Forward passes run on the background flush thread (not the main
        # thread); with torch's default multi-threaded OpenMP pool, the
        # native worker threads can still be alive when the main thread
        # starts interpreter shutdown, which races with libgomp teardown
        # and aborts the process (observed: "terminate called without an
        # active exception", SIGABRT, intermittently ~1 run in 3). Batching
        # already gets the parallelism we need across requests, so losing
        # intra-op CPU parallelism here costs little.
        #
        # IMPORTANT: torch.set_num_threads() is PROCESS-WIDE, not scoped to
        # this InferenceServer instance -- PyTorch has no per-instance
        # thread pool. Constructing any InferenceServer silently downgrades
        # ALL torch code running in the same process (e.g. Task 10's
        # Trainer, if it shares a process with this server) to
        # single-threaded intra-op execution. This is a deliberate,
        # necessary tradeoff to avoid the SIGABRT above, not an oversight;
        # a caller that also needs multi-threaded torch CPU ops elsewhere
        # in the same process should run this server in a separate process
        # instead.
        torch.set_num_threads(1)
        self.socket_path = socket_path
        self.net = net
        self.net.eval()
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._pending = []  # list of (planes_bytes, connection)
        self._running = True
        if os.path.exists(socket_path):
            os.remove(socket_path)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(socket_path)
        self._listener.listen(64)
        self._listener.settimeout(0.1)
        self._flusher_thread = None

    def stop(self):
        with self._lock:
            self._running = False
        self._listener.close()
        # Join the flush thread before returning: it is the one that calls
        # into torch, and leaving it alive-but-unjoined when the process
        # proceeds to interpreter shutdown races with native OpenMP/CUDA
        # thread-pool teardown and can abort the process (see note in
        # __init__).
        #
        # NOTE for callers (e.g. Task 10's orchestrator): this join has a
        # BOUNDED timeout, so it is a best-effort wait, not a hard
        # guarantee that the flush thread has fully exited by the time
        # stop() returns. An unconditional (untimed) join was rejected
        # deliberately: if a forward pass + sendall were ever stuck (slow
        # client, GPU hang, etc.), an untimed join would hang the entire
        # orchestrator's shutdown forever, which is worse than the already
        # unlikely SIGABRT this join mostly eliminates. If a forward pass
        # legitimately takes longer than the timeout below, stop() will
        # return while that thread is still running and still touching
        # torch -- the original race, just much less likely to be hit in
        # practice.
        flusher = self._flusher_thread
        if flusher is not None:
            flusher.join(timeout=max(self.flush_interval_s * 10, 1.0))

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._listener.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn):
        header = conn.recv(4)
        if len(header) < 4:
            conn.close()
            return
        (count,) = struct.unpack("<I", header)
        body = b""
        while len(body) < count * PLANE_BYTES:
            body += conn.recv(count * PLANE_BYTES - len(body))
        planes_list = unpack_request(header + body)
        with self._lock:
            if not self._running:
                # Server is shutting down: the flush thread may already
                # have exited, so queuing here would leave this client
                # hanging on recv() forever. Close instead of stranding it.
                conn.close()
                return
            self._pending.append((planes_list, conn))

    def _drain_and_process_once(self):
        """Take whatever is currently in `_pending`, run one forward pass
        over it if non-empty, and respond to each waiting client. Shared
        by the periodic loop below and by the unconditional final pass in
        `_flush_loop`, so both go through the exact same batching logic."""
        with self._lock:
            batch, self._pending = self._pending, []
        if not batch:
            return
        all_planes = [p for planes_list, _ in batch for p in planes_list]
        tensor = torch.stack([
            torch.frombuffer(bytearray(p), dtype=torch.uint8).reshape(6, 5, 5).float()
            for p in all_planes
        ])
        with torch.no_grad():
            policy_logits, values = self.net(tensor)
        policies = policy_logits.tolist()
        values = [v[0] for v in values.tolist()]
        offset = 0
        for planes_list, conn in batch:
            n = len(planes_list)
            response = pack_response(policies[offset:offset + n], values[offset:offset + n])
            try:
                conn.sendall(response)
            finally:
                conn.close()
            offset += n

    def _flush_loop(self):
        while self._running:
            time.sleep(self.flush_interval_s)
            self._drain_and_process_once()
        # The while condition above is checked BEFORE each sleep, so the
        # loop's last iteration already drains whatever had accumulated
        # during that final sleep -- but there is still a narrow window
        # between that last drain (the `with self._lock: batch, ... = ...`
        # swap) and this loop re-checking `self._running` and exiting:
        # a request appended to `_pending` in exactly that window would
        # otherwise sit there forever with no one left to flush it,
        # stranding its client on recv(). One more unconditional drain
        # right before this thread exits closes that window. (A client
        # that connects only after stop() has fully returned is a
        # different, caller-ordering concern -- the listener socket is
        # already closed by then -- and is out of scope here.)
        self._drain_and_process_once()

    def serve_forever(self):
        self._flusher_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher_thread.start()
        self._accept_loop()
