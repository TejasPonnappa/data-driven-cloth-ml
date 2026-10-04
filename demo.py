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
    rest = data["rest"]
    xte, yte = data["q_test"], data["y_test"]

    names = [args.model] + ([args.compare] if args.compare else [])
    models = {n: load_checkpoint(CKPT_DIR / f"{n}.pt")[0] for n in names}

    ncols = 2 + len(names)  # ground truth + each model + error of the main model
    fig = plt.figure(figsize=(4.2 * ncols, 5))
    axes = [fig.add_subplot(1, ncols, i + 1, projection="3d") for i in range(ncols)]
    state = {"i": args.index % len(xte)}

    def surface(ax, offsets, title, color_vals, vmax, cmap):
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
        surface(axes[0], gt, f"Ground truth (test pose {i})", gmag, vmax, "viridis")
        preds = {}
        for k, n in enumerate(names):
            with torch.no_grad():
                preds[n] = models[n](q).numpy()[0]
            pmag = np.linalg.norm(preds[n].reshape(-1, 3), axis=1)
            surface(axes[1 + k], preds[n], f"Prediction: {n}", pmag, vmax, "viridis")
        err = np.linalg.norm((preds[names[0]] - gt).reshape(-1, 3), axis=1)
        surface(axes[-1], preds[names[0]], f"Error of {names[0]} (mean {err.mean():.3f} cm)",
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
