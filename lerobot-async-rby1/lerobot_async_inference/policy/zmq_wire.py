"""OpenPI-compatible msgpack transport for NumPy arrays (no robot imports)."""

import io
from typing import Any

import msgpack
import numpy as np


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(
            data, object_hook=MsgSerializer._decode, raw=False, strict_map_key=False
        )

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Non-serializable type: {type(obj)}")

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj
