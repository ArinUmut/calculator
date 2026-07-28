from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def pack_signed(codes: Tensor, bits: int, *, chunk_elements: int = 8_000_000) -> bytes:
    """Pack signed two's-complement codes without a Python element loop."""
    if bits < 1 or bits > 8:
        raise ValueError("bits out of range")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    signed = codes.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
    count = signed.size
    mask = (1 << bits) - 1
    unsigned = np.bitwise_and(signed, mask).astype(np.uint16, copy=False)
    output = np.zeros((count * bits + 7) // 8, dtype=np.uint8)

    for start in range(0, count, chunk_elements):
        stop = min(count, start + chunk_elements)
        positions = np.arange(start, stop, dtype=np.int64) * bits
        byte_index = positions >> 3
        shifts = positions & 7
        values = unsigned[start:stop].astype(np.uint16, copy=False)
        low = ((values << shifts) & 0xFF).astype(np.uint8, copy=False)
        np.bitwise_or.at(output, byte_index, low)
        crossing = shifts + bits > 8
        if np.any(crossing):
            high = (values[crossing] >> (8 - shifts[crossing])).astype(np.uint8, copy=False)
            np.bitwise_or.at(output, byte_index[crossing] + 1, high)
    return output.tobytes()


def unpack_signed(data: bytes | bytearray | memoryview, count: int, bits: int, *, chunk_elements: int = 8_000_000) -> Tensor:
    """Vectorized inverse of :func:`pack_signed`."""
    if bits < 1 or bits > 8:
        raise ValueError("bits out of range")
    if count < 0 or chunk_elements <= 0:
        raise ValueError("invalid count/chunk_elements")
    required = (count * bits + 7) // 8
    source = np.frombuffer(data, dtype=np.uint8)
    if source.size < required:
        raise ValueError(f"packed buffer too short: need {required}, got {source.size}")
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    output = np.empty(count, dtype=np.int16)

    for start in range(0, count, chunk_elements):
        stop = min(count, start + chunk_elements)
        positions = np.arange(start, stop, dtype=np.int64) * bits
        byte_index = positions >> 3
        shifts = positions & 7
        raw = (source[byte_index].astype(np.uint16) >> shifts).astype(np.uint16)
        crossing = shifts + bits > 8
        if np.any(crossing):
            high = np.left_shift(
                source[byte_index[crossing] + 1].astype(np.uint16),
                (8 - shifts[crossing]).astype(np.uint16),
            ).astype(np.uint16, copy=False)
            raw[crossing] = np.bitwise_or(raw[crossing], high)
        raw &= mask
        signed = raw.astype(np.int16)
        negative = (raw & sign) != 0
        signed[negative] -= 1 << bits
        output[start:stop] = signed
    return torch.from_numpy(output)
