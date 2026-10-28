# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small versioned NumPy/msgpack wire format for split Flexiv deployment."""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_ARRAY_EXT = 1
_DTYPES = {"uint8", "float32", "float64", "bool"}


def _default(value: Any) -> msgpack.ExtType:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported wire value: {type(value)!r}")
    array = np.ascontiguousarray(value)
    if str(array.dtype) not in _DTYPES:
        raise ValueError(f"Unsupported wire array dtype: {array.dtype}")
    payload = msgpack.packb(
        (str(array.dtype), array.shape, array.tobytes()), use_bin_type=True
    )
    return msgpack.ExtType(_ARRAY_EXT, payload)


def _ext_hook(code: int, payload: bytes) -> Any:
    if code != _ARRAY_EXT:
        return msgpack.ExtType(code, payload)
    dtype, shape, raw = msgpack.unpackb(payload, raw=False)
    if dtype not in _DTYPES or len(shape) > 4 or any(int(n) < 0 for n in shape):
        raise ValueError("Invalid wire array metadata")
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if expected != len(raw):
        raise ValueError("Wire array byte count does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    """Send one bounded message; caller owns socket ordering."""
    payload = msgpack.packb(message, default=_default, use_bin_type=True)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Flexiv wire message exceeds 16 MiB")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _read_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = connection.recv(count - len(chunks))
        if not part:
            raise ConnectionError("Flexiv inference server disconnected")
        chunks.extend(part)
    return bytes(chunks)


def receive_message(connection: socket.socket) -> dict[str, Any]:
    """Read and validate one framed message."""
    size = struct.unpack("!I", _read_exact(connection, 4))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid Flexiv wire message length")
    value = msgpack.unpackb(
        _read_exact(connection, size), raw=False, ext_hook=_ext_hook
    )
    if not isinstance(value, dict) or value.get("version") != PROTOCOL_VERSION:
        raise ValueError("Flexiv wire protocol version mismatch")
    return value
