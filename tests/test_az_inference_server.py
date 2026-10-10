import socket
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
        import struct
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
