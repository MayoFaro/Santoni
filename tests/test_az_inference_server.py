import socket
import struct
import threading
import time

import torch

from santoni_az.network import PolicyValueNet, NUM_PLANES
from santoni_az.inference_protocol import pack_request, unpack_response, PLANE_BYTES, ACTION_SPACE
from santoni_az.inference_server import InferenceServer


def test_two_concurrent_clients_get_correct_shaped_responses(tmp_path):
    socket_path = str(tmp_path / "infer.sock")
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.05)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    def client(results, index):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        planes = bytes(PLANE_BYTES)
        sock.sendall(pack_request([planes]))
        header = sock.recv(4)
        (count,) = struct.unpack("<I", header)
        body = b""
        needed = count * (ACTION_SPACE * 4 + 4)
        while len(body) < needed:
            body += sock.recv(needed - len(body))
        policies, values = unpack_response(header + body, batch_size=count)
        results[index] = (policies, values)
        sock.close()

    results = {}
    t1 = threading.Thread(target=client, args=(results, 0))
    t2 = threading.Thread(target=client, args=(results, 1))
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)
    server.stop()

    assert len(results) == 2
    for policies, values in results.values():
        assert len(policies[0]) == ACTION_SPACE
        assert -1 <= values[0] <= 1


def test_flush_loop_final_drain_catches_request_queued_at_shutdown(tmp_path):
    """Regression test for a shutdown race: a request that lands in
    `_pending` after the flush thread's last timed drain, but before
    `stop()` has fully returned, must still get a response -- not strand
    its connection on `recv()` forever. The window in real time is a few
    microseconds wide and not reliably reproducible by racing real
    threads/sockets, so we reproduce it deterministically instead: seed
    `_pending` directly and set `_running = False` *before* calling
    `_flush_loop()`, which forces the while-loop body to run zero times
    and exercises exactly the trailing unconditional drain added to close
    this race."""
    socket_path = str(tmp_path / "infer_drain.sock")
    net = PolicyValueNet(channels=8, blocks=1)
    # A large flush_interval_s is irrelevant here since we call
    # _flush_loop() directly rather than letting it sleep in a thread.
    server = InferenceServer(socket_path, net, flush_interval_s=10.0)

    server_side, client_side = socket.socketpair()
    planes_list = [bytes(PLANE_BYTES)]
    server._pending.append((planes_list, server_side))

    # Simulate "stop() already flipped _running" happening strictly before
    # _flush_loop notices it, without going through the real stop()/thread
    # machinery (which is what the other test already covers under real
    # timing).
    server._running = False
    server._flush_loop()  # while-loop body runs 0 times; only the trailing
                           # unconditional drain can process the seeded entry.

    client_side.settimeout(2.0)
    header = client_side.recv(4)
    (count,) = struct.unpack("<I", header)
    needed = count * (ACTION_SPACE * 4 + 4)
    body = b""
    while len(body) < needed:
        body += client_side.recv(needed - len(body))
    policies, values = unpack_response(header + body, batch_size=count)
    client_side.close()

    assert len(policies[0]) == ACTION_SPACE
    assert -1 <= values[0] <= 1

    server.stop()  # cleanup; _flusher_thread is None here, so this is just
                    # closing the listener socket.
