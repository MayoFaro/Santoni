import struct

PLANE_BYTES = 150
ACTION_SPACE = 1250


def pack_request(planes_batch: list[bytes]) -> bytes:
    header = struct.pack("<I", len(planes_batch))
    body = b"".join(planes_batch)
    return header + body


def unpack_request(data: bytes) -> list[bytes]:
    (count,) = struct.unpack_from("<I", data, 0)
    offset = 4
    result = []
    for _ in range(count):
        result.append(data[offset:offset + PLANE_BYTES])
        offset += PLANE_BYTES
    return result


def pack_response(policies: list[list[float]], values: list[float]) -> bytes:
    parts = [struct.pack("<I", len(values))]
    for policy, value in zip(policies, values):
        parts.append(struct.pack(f"<{ACTION_SPACE}f", *policy))
        parts.append(struct.pack("<f", value))
    return b"".join(parts)


def unpack_response(data: bytes, batch_size: int) -> tuple[list[list[float]], list[float]]:
    (count,) = struct.unpack_from("<I", data, 0)
    assert count == batch_size
    offset = 4
    policies, values = [], []
    item_size = ACTION_SPACE * 4
    for _ in range(count):
        policies.append(list(struct.unpack_from(f"<{ACTION_SPACE}f", data, offset)))
        offset += item_size
        (value,) = struct.unpack_from("<f", data, offset)
        values.append(value)
        offset += 4
    return policies, values
