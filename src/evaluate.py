"""Evaluate trained models on the TEST set and make the report plots.

    python -m src.evaluate                      # default ablation set
    python -m src.evaluate --names pca128 linreg

Writes to results/: summary.csv, summary.json, explained_variance.png,
ablation.png, training_curves.png, per_vertex_error_<model>.png
"""
import argparse
import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_dataset
from src.model import load_checkpoint, n_trainable_params
from src.pca import PCA
from src.utils import CKPT_DIR, DATA_PATH, RESULTS_DIR, ensure_dir, load_json, save_json

DEFAULT_NAMES = ["linreg", "direct", "pca256", "pca128", "pca64"]


@torch.no_grad()
def predict(model, x, bs=1024):
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)]).numpy()


def vertex_errors(pred, y):
    """Euclidean error per vertex, shape (N, V), in cm."""
    return np.linalg.norm((pred - y).reshape(len(y), -1, 3), axis=2)


@torch.no_grad()
def time_ms(model, x, batch, runs=200):
    xb = x[:batch]
    for _ in range(20):  # warm-up
        model(xb)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(xb)
    return (time.perf_counter() - t0) / runs * 1000.0


def evaluate_model(name, xte, yte):
    model, ckpt = load_checkpoint(CKPT_DIR / f"{name}.pt")
    pred = predict(model, xte)
    err = vertex_errors(pred, yte)
    cfg = ckpt["config"]
    mse = float(np.mean((pred - yte) ** 2))
    row = dict(
        name=name, mode=cfg["mode"], k=cfg["k"] or "-", hidden=str(cfg["hidden"]),
        params=n_trainable_params(model), best_epoch=ckpt.get("epoch"),
        test_mse=mse, test_rmse_cm=mse ** 0.5,
        mean_vertex_err_cm=float(err.mean()),
        p95_vertex_err_cm=float(np.percentile(err, 95)),
        ms_per_pose_batch1=time_ms(model, xte, 1),
        ms_per_pose_batch256=time_ms(model, xte, 256, runs=50) / 256,
    )
    return row, err


def plot_explained_variance(ytr, yte, path):
    pca = PCA().fit(ytr)
    ks = np.arange(1, min(300, len(pca.explained_variance_)) + 1)
    cum = np.cumsum(pca.explained_variance_ratio_)[: len(ks)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, 100 * cum, color="#1f77b4")
    ax[0].set_xlabel("number of principal components k")
    ax[0].set_ylabel("cumulative explained variance (%)")
    ax[1].semilogy([1, 8, 16, 32, 64, 128, 256],
                   [pca.reconstruction_mse(yte, k) for k in [1, 8, 16, 32, 64, 128, 256]],
                   "o-", color="#d62728")
    ax[1].set_xlabel("number of principal components k")
    ax[1].set_ylabel("test reconstruction MSE (cm$^2$)")
    for a in ax:
        a.axvline(64, ls=":", c="gray")
        a.axvline(128, ls=":", c="gray")
        a.grid(alpha=0.3)
    ax[0].set_title("PCA of vertex offsets")
    ax[1].set_title("Reconstruction floor for a perfect predictor")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(rows, mean_mse, path):
    names = [r["name"] for r in rows]
    colors = plt.cm.tab10(np.arange(len(names)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bars = ax[0].bar(names, [r["test_mse"] for r in rows], color=colors)
    ax[0].bar_label(bars, fmt="%.3f", fontsize=8)
    ax[0].set_ylabel("test MSE (cm$^2$)")
    ax[0].set_title(f"Accuracy (predicting the train mean: {mean_mse:.2f})")
    bars = ax[1].bar(names, [r["ms_per_pose_batch1"] for r in rows], color=colors)
    ax[1].bar_label(bars, fmt="%.3f", fontsize=8)
    ax[1].set_ylabel("ms per pose (batch of 1, CPU)")
    ax[1].set_title("Inference time")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(names, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        p = RESULTS_DIR / f"{name}_history.json"
        if p.exists():
            h = load_json(p)
            if len(h["val_mse"]) > 1:
                ax.plot(np.arange(1, len(h["val_mse"]) + 1), h["val_mse"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (cm$^2$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vertex_error(err, grid_shape, name, path):
    nx, ny = grid_shape
    m = err.mean(axis=0).reshape(ny, nx)  # vertex id = row * nx + col
    fig, ax = plt.subplots(figsize=(4.5, 5))
    im = ax.imshow(m, origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="mean error (cm)")
    ax.set_title(f"Per-vertex test error: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--heatmap_model", default=None, help="default: best test MSE")
    ap.add_argument("--out_dir", default=str(RESULTS_DIR), help="where summary + plots go")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    data = load_dataset(args.data)
    xte, yte = torch.from_numpy(data["q_test"]), data["y_test"]

    names = [n for n in args.names if (CKPT_DIR / f"{n}.pt").exists()]
    missing = set(args.names) - set(names)
    if missing:
        print(f"skipping (no checkpoint yet): {sorted(missing)}")

    rows, errs = [], {}
    for n in names:
        row, err = evaluate_model(n, xte, yte)
        rows.append(row)
        errs[n] = err
        print(f"{n:10s} MSE {row['test_mse']:.5f} | RMSE {row['test_rmse_cm']:.4f} cm | "
              f"mean vertex err {row['mean_vertex_err_cm']:.4f} cm | "
              f"{row['ms_per_pose_batch1']:.3f} ms/pose")

    mean_mse = float(np.mean((yte - data["y_train"].mean(axis=0)) ** 2))
    print(f"reference: predicting the train mean gives MSE {mean_mse:.5f}")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows, "mean_predictor_mse": mean_mse}, out / "summary.json")

    plot_explained_variance(data["y_train"], yte, out / "explained_variance.png")
    plot_ablation(rows, mean_mse, out / "ablation.png")
    plot_curves(names, out / "training_curves.png")
    best = args.heatmap_model or min(rows, key=lambda r: r["test_mse"])["name"]
    if int(data["grid_shape"][0]) > 0:  # flat panel only; meshes without a grid skip this plot
        plot_vertex_error(errs[best], data["grid_shape"], best, out / f"per_vertex_error_{best}.png")
    print(f"saved summary + plots to {out}")


if __name__ == "__main__":
    main()