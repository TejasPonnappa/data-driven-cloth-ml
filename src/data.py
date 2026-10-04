"""Dataset: pose (14 joint quaternions = 56 numbers) -> cloth vertex offsets (V*3 numbers).

The paper's data came from Blender skinning + PhysBAM cloth simulation, which we
do not have. So we build a SYNTHETIC stand-in with the same shape and the same
pipeline:

    * a cloth panel = nx*ny grid of vertices (rest positions)
    * a fixed, smooth, NONLINEAR function of the pose gives each vertex's offset
      (offset = displacement of the simulated cloth from the skinned cloth)
    * a little noise (unpredictable part) + ~1% blown-up "bad simulation" samples
      so the outlier-removal step from the paper actually does something

The generating function is fixed (seed 1234), so train/val/test always come from
the same "physics". Only the sampled poses/noise depend on --seed.

If you find a real dataset later, just save it in the same .npz format
(see build_dataset) and the rest of the code works unchanged.

Usage:
    python -m src.data --n 10000 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np

from src.utils import DATA_PATH, ensure_dir

N_JOINTS = 14
PANEL_W, PANEL_H = 30.0, 40.0  # cm


def _rest_grid(nx, ny):
    xs = np.linspace(0, PANEL_W, nx)
    ys = np.linspace(0, PANEL_H, ny)
    X, Y = np.meshgrid(xs, ys)  # shape (ny, nx); vertex id = row * nx + col
    return np.stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)], axis=1)


def _random_quaternions(rng, n, max_angles):
    """Random joint rotations -> unit quaternions (w, x, y, z), shape (n, J*4)."""
    J = len(max_angles)
    axes = rng.normal(size=(n, J, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(0.0, 1.0, size=(n, J)) * max_angles[None, :]
    half = angles / 2.0
    q = np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axes], axis=-1)
    return q.reshape(n, J * 4)


def generate_synthetic(n=10000, nx=20, ny=25, seed=0, outlier_frac=0.01,
                       noise=0.05, target_std=1.5):
    world = np.random.default_rng(1234)  # fixed "physics"
    rng = np.random.default_rng(seed)  # sampled poses / noise

    in_dim = 4 * N_JOINTS
    max_angles = world.uniform(np.pi / 6, np.pi / 2, N_JOINTS)  # joint limits 30-90 deg
    rest = _rest_grid(nx, ny)

    # smooth spatial basis over the panel (low frequencies matter most)
    u, v = rest[:, 0] / PANEL_W, rest[:, 1] / PANEL_H
    modes, amps = [], []
    for a in range(8):
        for b in range(8):
            modes.append(np.cos(np.pi * a * u) * np.cos(np.pi * b * v))
            amps.append(1.0 / (1.0 + a + b) ** 1.5)
    B = np.stack(modes, axis=1)  # (V, M)
    amps = np.array(amps)
    M = B.shape[1]

    # fixed nonlinear pose -> mode-coefficient map (2 tanh layers)
    H1, L = 64, 48
    W1 = world.normal(size=(in_dim, H1)) * 2.5 / np.sqrt(in_dim)
    b1 = world.normal(size=H1) * 0.3
    W2 = world.normal(size=(H1, L)) * 1.5 / np.sqrt(H1)
    b2 = world.normal(size=L) * 0.3
    A = world.normal(size=(L, M * 3)) / np.sqrt(L) * np.repeat(amps, 3)[None, :]

    q = _random_quaternions(rng, n, max_angles)
    rest_q = np.tile([1.0, 0.0, 0.0, 0.0], N_JOINTS)
    p = (q - rest_q[None, :]) / 0.3  # features relative to rest pose
    h = np.tanh(np.tanh(p @ W1 + b1) @ W2 + b2)  # (n, L)
    coef = (h @ A).reshape(n, M, 3)
    off = np.einsum("vm,nmc->nvc", B, coef)  # (n, V, 3)

    off *= target_std / off.std()  # offsets of ~1-2 cm
    off += rng.normal(scale=noise, size=off.shape)  # unpredictable part

    # a few exploded simulations -> to be removed by the outlier filter
    bad = rng.random(n) < outlier_frac
    off[bad] *= rng.uniform(6.0, 12.0, size=(bad.sum(), 1, 1))

    return {
        "q": q.astype(np.float32),
        "y": off.reshape(n, -1).astype(np.float32),
        "rest": rest.astype(np.float32),
        "grid_shape": np.array([nx, ny]),
        "is_outlier": bad,
    }


def filter_outliers(y, max_dist=None, k_mad=5.0):
    """Drop examples whose largest vertex displacement is too big.

    max_dist=None -> automatic threshold: median + k_mad * (robust std) of the
    per-example max displacement.  Returns (keep_mask, threshold_used).
    """
    d = np.linalg.norm(y.reshape(len(y), -1, 3), axis=2).max(axis=1)
    if max_dist is None:
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826
        max_dist = med + k_mad * mad
    return d <= max_dist, float(max_dist)


def split_indices(n, n_train, n_val, seed):
    if n_train + n_val >= n:
        raise ValueError(f"Not enough examples ({n}) for {n_train}+{n_val} train/val.")
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def build_dataset(path=DATA_PATH, n=10000, seed=0, n_train=6000, n_val=2000,
                  max_dist=None, **gen_kwargs):
    d = generate_synthetic(n=n, seed=seed, **gen_kwargs)
    keep, thr = filter_outliers(d["y"], max_dist)
    removed = int((~keep).sum())
    caught = int((d["is_outlier"] & ~keep).sum())
    print(f"generated {n} | outlier threshold {thr:.2f} cm | removed {removed} "
          f"(injected bad sims caught: {caught}/{int(d['is_outlier'].sum())})")
    q, y = d["q"][keep], d["y"][keep]
    tr, va, te = split_indices(len(q), n_train, n_val, seed)
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        q_train=q[tr], y_train=y[tr],
        q_val=q[va], y_val=y[va],
        q_test=q[te], y_test=y[te],
        rest=d["rest"], grid_shape=d["grid_shape"], dist_threshold=thr,
    )
    print(f"split: train {len(tr)} / val {len(va)} / test {len(te)} -> saved to {path}")
    print(f"input dim {q.shape[1]}, output dim {y.shape[1]} ({y.shape[1] // 3} vertices)")


def load_dataset(path=DATA_PATH):
    try:
        z = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found. Generate it first with:  python -m src.data --n 10000"
        )
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA_PATH))
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--max_dist", type=float, default=None,
                    help="outlier threshold in cm (default: automatic)")
    args = ap.parse_args()
    build_dataset(Path(args.out), args.n, args.seed, args.n_train, args.n_val, args.max_dist)
