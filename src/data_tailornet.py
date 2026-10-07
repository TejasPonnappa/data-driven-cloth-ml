"""Build a dataset from the public TailorNet data (real simulated T-shirt) in OUR format.

TailorNet (Patel et al., CVPR 2020) provides, for a garment, body shape and style,
the SMPL pose theta (72 numbers) and the simulated garment's "unposed" vertices.
We fix ONE shape and ONE style, so the task is exactly the paper's:

    pose (72)  ->  per-vertex offset of the simulated garment from its rest garment

Folder layout expected (what the TailorNet sample download unzips to):

    <root>/pose/<shape>_<style>/poses_000.npz         (thetas: N x 72)
    <root>/pose/<shape>_<style>/unposed_000.npy       (N x V x 3, metres)
    <root>/style_shape/beta<shape>_gamma<style>.npy   (V x 3, rest garment)

Usage:
    python -m src.data_tailornet --root path/to/t-shirt_female
Writes data/tailornet.npz with the same keys as the synthetic data, so
`python -m src.train --data data/tailornet.npz ...` works unchanged.
Units are converted from metres to cm (as in the rest of the project).

If the folder holds MANY <shape>_<style> combos (a bigger garment download), they are all
used: the network then also gets the body shape (first 10 betas) and garment style (4 gammas)
as inputs (72 + 10 + 4 = 86), and each offset is measured from that combo's own rest garment.
--max_samples caps the total number of poses so that training stays fast.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from src.data import filter_outliers
from src.utils import ROOT, ensure_dir

TN_PATH = ROOT / "data" / "tailornet.npz"


def list_combos(root):
    combos = sorted(p.name for p in (Path(root) / "pose").iterdir() if p.is_dir())
    if not combos:
        raise FileNotFoundError(f"no <shape>_<style> folders under {root}/pose")
    return combos


def load_combo(root, combo, take, rng, cond=True):
    """Return (thetas, unposed, rest, beta10, gamma) for up to `take` random poses of one combo."""
    root = Path(root)
    shape, style = combo.split("_")
    d = root / "pose" / combo
    pose_files = sorted(d.glob("poses_*.npz"))
    if not pose_files:
        raise FileNotFoundError(f"no poses_*.npz in {d}")
    thetas, disp = [], []
    for pf in pose_files:
        idx = re.search(r"poses_(\d+)", pf.name).group(1)
        th = np.load(pf)["thetas"]
        up = np.load(d / f"unposed_{idx}.npy", mmap_mode="r")
        thetas.append(th)
        disp.append(up)
    n_tot = sum(len(t) for t in thetas)
    pick = np.sort(rng.permutation(n_tot)[:take]) if take < n_tot else np.arange(n_tot)
    offs = np.cumsum([0] + [len(t) for t in thetas])
    th_out, up_out = [], []
    for i in pick:
        f = np.searchsorted(offs, i, side="right") - 1
        th_out.append(thetas[f][i - offs[f]])
        up_out.append(np.asarray(disp[f][i - offs[f]]))
    rest = np.load(root / "style_shape" / f"beta{shape}_gamma{style}.npy").astype(np.float32)
    if not cond:
        return np.array(th_out, np.float32), np.array(up_out, np.float32), rest, None, None
    beta = np.load(root / "shape" / f"beta_{shape}.npy").astype(np.float32)[:10]
    gamma = np.load(root / "style" / f"gamma_{style}.npy").astype(np.float32)
    return np.array(th_out, np.float32), np.array(up_out, np.float32), rest, beta, gamma


def build(root, out=TN_PATH, combo=None, seed=0, frac_train=0.6, frac_val=0.2,
          max_dist=None, max_samples=4000, max_combos=None):
    rng = np.random.default_rng(seed)
    combos = [combo] if combo else list_combos(root)
    if max_combos and len(combos) > max_combos:
        combos = sorted(rng.choice(combos, size=max_combos, replace=False).tolist())
    multi = len(combos) > 1
    per = int(np.ceil(max_samples / len(combos)))
    print(f"{len(combos)} combo(s), up to {per} poses each; "
          f"inputs = {'pose + shape + style (86)' if multi else 'pose only (72)'}")

    X, Y, R = [], [], []
    for c in combos:
        th, up, rest, beta, gamma = load_combo(root, c, per, rng, cond=multi)
        feats = np.concatenate([th, np.tile(np.concatenate([beta, gamma]), (len(th), 1))], axis=1) if multi else th
        X.append(feats)
        Y.append(((up - rest[None]) * 100.0).reshape(len(th), -1))  # offsets in cm
        R.append(np.broadcast_to((rest * 100.0)[None], up.shape))
    x = np.concatenate(X).astype(np.float32)
    y = np.concatenate(Y).astype(np.float32)
    r = np.concatenate(R).astype(np.float32)
    n, v = len(y), y.shape[1] // 3
    print(f"{n} poses, {v} garment vertices (= {3 * v} outputs), input dim {x.shape[1]}")
    mag = np.linalg.norm(y.reshape(n, -1, 3), axis=2)
    print(f"offset magnitude: mean per vertex {mag.mean():.3f} cm, max {mag.max():.2f} cm")

    keep, thr = filter_outliers(y, max_dist)
    print(f"outlier threshold {thr:.2f} cm -> removed {int((~keep).sum())}")
    x, y, r = x[keep], y[keep], r[keep]
    n = len(y)

    perm = rng.permutation(n)
    n_tr, n_va = int(round(frac_train * n)), int(round(frac_val * n))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    out = Path(out)
    ensure_dir(out.parent)
    extra = {"rest_test": r[te]} if multi else {}
    np.savez_compressed(
        out,
        q_train=x[tr], y_train=y[tr], q_val=x[va], y_val=y[va], q_test=x[te], y_test=y[te],
        rest=r[0], grid_shape=np.array([0, 0]), dist_threshold=thr, **extra,
    )
    print(f"split: train {len(tr)} / val {len(va)} / test {len(te)} -> saved to {out}")
    print(f"PCA can have at most {len(tr)} components (training poses)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. C:\\...\\t-shirt_female")
    ap.add_argument("--combo", default=None, help="use only this combo, e.g. 000_023 (default: all)")
    ap.add_argument("--max_samples", type=int, default=4000, help="cap on total poses (keeps training fast)")
    ap.add_argument("--max_combos", type=int, default=None, help="use at most this many combos")
    ap.add_argument("--out", default=str(TN_PATH))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.root, Path(args.out), args.combo, args.seed,
          max_samples=args.max_samples, max_combos=args.max_combos)