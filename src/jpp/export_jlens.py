"""Export a fitted lens to the app's .jlens format (same writer as
src/step0_app_prep.py export_jlens), under out/jpp/.

Format (little-endian): magic 'JLNS', uint32 version=1, uint32 d_model,
uint32 n_layers, n_layers x uint32 layer index, then per layer J as fp32
row-major [d_model, d_model], rows = OUTPUT dims (transport = J @ h).

J++ maps carry an arbitrary positive scale (unit-L2 expert weights); readout
ranks, the app's displayed scores (1/rms after J) and swap directions (unit
normalized) are all invariant to it, but each layer is rescaled here to the
Frobenius norm of the shipped J at that layer so magnitudes look familiar.

Usage: python src/jpp/export_jlens.py <model> <variant file stem> <out name> <layers comma-sep>
  e.g. python src/jpp/export_jlens.py stories110M jpp_rlens_skip16_n1000 stories110M.jlens 4,5,6,7,8
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import JPP_OUT, load_existing_J  # noqa: E402


def write_jlens(J: dict[int, torch.Tensor], layers: list[int], out_path: Path) -> None:
    d = J[layers[0]].shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(b"JLNS")
        fh.write(struct.pack("<III", 1, d, len(layers)))
        for l in layers:
            fh.write(struct.pack("<I", l))
        for l in layers:
            M = J[l].to(torch.float32).contiguous()
            assert M.shape == (d, d) and torch.isfinite(M).all()
            fh.write(M.numpy().astype("<f4").tobytes())
    print(f"[export] {out_path}: d={d} layers={layers} ({out_path.stat().st_size / 1e6:.1f} MB)")


def read_jlens(path: Path) -> dict[int, torch.Tensor]:
    import numpy as np

    b = path.read_bytes()
    assert b[:4] == b"JLNS"
    ver, d, n = struct.unpack("<III", b[4:16])
    layers = struct.unpack(f"<{n}I", b[16 : 16 + 4 * n])
    off = 16 + 4 * n
    out = {}
    for l in layers:
        out[l] = torch.from_numpy(np.frombuffer(b, dtype="<f4", count=d * d, offset=off).reshape(d, d).copy())
        off += 4 * d * d
    assert off == len(b)
    return out


def main() -> None:
    model_key, stem, name, layers = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
    state = torch.load(JPP_OUT / model_key / f"{stem}.pt", map_location="cpu", weights_only=True)
    J = {int(l): M.float() for l, M in state["J"].items()}
    ref = load_existing_J(model_key, 1000)
    scaled = {}
    for l in layers:
        c = torch.nn.functional.cosine_similarity(J[l].flatten(), ref[l].flatten(), dim=0)
        assert c > 0, f"L{l}: J++ map anti-aligned with J (cos {c:.3f}); refusing to rescale"
        scaled[l] = J[l] * (ref[l].norm() / J[l].norm())
    out = JPP_OUT / "export" / name
    write_jlens(scaled, layers, out)
    back = read_jlens(out)
    assert all(torch.equal(back[l], scaled[l].float()) for l in layers), "round-trip mismatch"
    app = Path(__file__).resolve().parents[2] / "app" / name
    if app.exists():
        shipped = read_jlens(app)
        print(f"  shipped {app.name}: layers {sorted(shipped)}; export layers {sorted(back)}; "
              f"same shape: {all(shipped[l].shape == back[l].shape for l in back if l in shipped)}")


if __name__ == "__main__":
    main()
