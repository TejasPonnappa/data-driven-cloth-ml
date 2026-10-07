"""Live demo: pick a test pose, show ground-truth cloth vs the model's prediction.

    python demo.py --model pca128            # window; use LEFT/RIGHT arrow keys to change pose
    python demo.py --model pca128 --save results/demo.png --no_show --index 7
Add --compare linreg to also show the linear-regression baseline.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_dataset
from src.model import load_checkpoint
from src.utils import CKPT_DIR, DATA_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pca128")
    ap.add_argument("--compare", default=None, help="second model to show, e.g. linreg")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--index", type=int, default=0, help="test pose index")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    data = load_dataset(args.data)
    nx, ny = (int(v) for v in data["grid_shape"])
    cloud = nx == 0  # real garment mesh (e.g. TailorNet): drawn as a point cloud
    rest = data["rest"]
    xte, yte = data["q_test"], data["y_test"]

    def rest_now():  # multi-combo TailorNet data stores each test pose's own rest garment
        return data["rest_test"][state["i"]] if "rest_test" in data else data["rest"]

    names = [args.model] + ([args.compare] if args.compare else [])
    models = {n: load_checkpoint(CKPT_DIR / f"{n}.pt")[0] for n in names}

    ncols = 2 + len(names)  # ground truth + each model + error of the main model
    fig = plt.figure(figsize=(3.6 * ncols, 7) if cloud else (4.2 * ncols, 5))
    axes = [fig.add_subplot(1, ncols, i + 1, **({} if cloud else {"projection": "3d"})) for i in range(ncols)]
    state = {"i": args.index % len(xte)}

    def surface(ax, offsets, title, color_vals, vmax, cmap):
        if cloud:  # clean 2D front view (x vs height), points coloured by how far they moved
            rest_i = rest_now()
            pos = rest_i + offsets.reshape(-1, 3)
            order = np.argsort(pos[:, 2])  # draw far points first
            ax.clear()
            ax.scatter(pos[order, 0], pos[order, 1], c=color_vals[order], cmap=cmap,
                       vmin=0, vmax=vmax, s=5, linewidths=0)
            lo, hi = rest_i.min(axis=0), rest_i.max(axis=0)
            padx, pady = 0.15 * (hi[0] - lo[0]), 0.05 * (hi[1] - lo[1])
            ax.set_xlim(lo[0] - padx, hi[0] + padx); ax.set_ylim(lo[1] - pady, hi[1] + pady)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            return
        pos = (rest + offsets.reshape(-1, 3)).reshape(ny, nx, 3)
        colors = plt.get_cmap(cmap)(np.clip(color_vals.reshape(ny, nx) / vmax, 0, 1))[:-1, :-1]
        ax.clear()
        ax.plot_surface(pos[..., 0], pos[..., 1], pos[..., 2], facecolors=colors,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_title(title, fontsize=10)
        ax.set_zlim(-8, 8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (cm)")
        ax.view_init(elev=25, azim=-60)

    def draw():
        i = state["i"]
        q = torch.from_numpy(xte[i:i + 1])
        gt = yte[i]
        gmag = np.linalg.norm(gt.reshape(-1, 3), axis=1)
        vmax = max(gmag.max(), 1e-6)
        surface(axes[0], gt, f"Real simulation (pose {i})", gmag, vmax, "viridis")
        preds = {}
        for k, n in enumerate(names):
            with torch.no_grad():
                preds[n] = models[n](q).numpy()[0]
            pmag = np.linalg.norm(preds[n].reshape(-1, 3), axis=1)
            surface(axes[1 + k], preds[n], f"Predicted: {n}", pmag, vmax, "viridis")
        err = np.linalg.norm((preds[names[0]] - gt).reshape(-1, 3), axis=1)
        surface(axes[-1], preds[names[0]], f"Error, {names[0]}\nmean {err.mean():.2f} cm",
                err, max(err.max(), 1e-6), "magma")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["i"] = (state["i"] + 1) % len(xte)
        elif event.key == "left":
            state["i"] = (state["i"] - 1) % len(xte)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()