"""E1: rewrite the ternary g128 PQ2_0 Bonsai 2 file at another scale group size, codes unchanged."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/claude-1000/-var-home-zero-maxwell/bd9c629a-079a-4a8a-afe8-084090787b3f/scratchpad/prism-ss/gguf-py")
import gguf  # noqa: E402

PQ = gguf.GGMLQuantizationType.PQ2_0
Q2 = gguf.GGMLQuantizationType.Q2_0


def split64(raw: np.ndarray) -> np.ndarray:
    """One g128 block (d, 32 qs) becomes two g64 Q2_0 blocks sharing d: lossless."""
    blocks = raw.reshape(-1, 34)
    out = np.empty((blocks.shape[0], 2, 18), np.uint8)
    out[:, :, :2] = blocks[:, None, :2]
    out[:, 0, 2:] = blocks[:, 2:18]
    out[:, 1, 2:] = blocks[:, 18:34]
    return out.reshape(raw.shape[0], -1)


def merge(raw: np.ndarray, group: int) -> np.ndarray:
    """Least-squares shared scale over group/128 consecutive blocks: d* = sum(c_i d_i) / sum(c_i), c_i = sum q^2."""
    m = group // 128
    rows = raw.shape[0]
    blocks = raw.reshape(rows, -1, m, 34).copy()
    d = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float64)
    qs = blocks[..., 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], -1).astype(np.int64) - 1
    c = (q * q).sum(axis=(-1, -2)).astype(np.float64)
    total = c.sum(-1, keepdims=True)
    shared = np.where(total > 0, (c * d).sum(-1, keepdims=True) / np.where(total > 0, total, 1), d.mean(-1, keepdims=True))
    blocks[..., :2] = np.broadcast_to(shared.astype(np.float16), d.shape)[..., None].view(np.uint8)
    return blocks.reshape(rows, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--group", type=int, required=True, choices=(64, 256, 512, 1024))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    reader = gguf.GGUFReader(a.src)
    writer = gguf.GGUFWriter(a.out, arch=reader.fields[gguf.Keys.General.ARCHITECTURE].contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        writer.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    writer.add_key_value("carl.e1.scale_group", a.group, gguf.GGUFValueType.UINT32)
    writer.add_key_value("carl.e1.scale_fit", "split-lossless" if a.group == 64 else "least-squares-shared-scale", gguf.GGUFValueType.STRING)

    def convert(t):
        if t.tensor_type != PQ:
            return t.data, t.tensor_type
        raw = np.asarray(t.data)
        if a.group == 64:
            return split64(raw), Q2
        return merge(raw, a.group), PQ

    for t in reader.tensors:
        if t.tensor_type != PQ:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            continue
        rows, width = t.data.shape
        shape = (rows, width // 34 * 36) if a.group == 64 else (rows, width)
        writer.add_tensor_info(t.name, shape, np.dtype(np.uint8), rows * shape[1], Q2 if a.group == 64 else PQ)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        data, _ = convert(t)
        writer.write_tensor_data(np.ascontiguousarray(data), tensor_endianess=reader.endianess)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
