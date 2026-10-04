"""Train one model.

Examples (run from the repo root):
    # MLP (128,256) predicting 128 PCA coefficients
    python -m src.train --name pca128 --mode pca --k 128
    # MLP predicting the offsets directly (no PCA)
    python -m src.train --name direct --mode direct
    # linear regression baseline (closed-form least squares, no hidden layers)
    python -m src.train --name linreg --mode direct --hidden --closed_form
"""
import argparse
import time

import torch
import torch.nn.functional as F

from src.data import load_dataset
from src.model import build_model, fit_closed_form, n_trainable_params, save_checkpoint
from src.pca import PCA
from src.utils import CKPT_DIR, DATA_PATH, RESULTS_DIR, ensure_dir, get_device, save_json, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="used for checkpoints/<name>.pt")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--mode", choices=["pca", "direct"], default="pca")
    ap.add_argument("--k", type=int, default=128, help="number of PCA components (pca mode)")
    ap.add_argument("--hidden", type=int, nargs="*", default=[128, 256],
                    help="hidden layer sizes; give none for linear regression")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=0, help="early stop after N epochs w/o val improvement (0=off)")
    ap.add_argument("--closed_form", action="store_true", help="least-squares fit instead of SGD (needs no hidden layers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=5)
    return ap.parse_args()


@torch.no_grad()
def eval_mse(model, x, y, bs=1024):
    model.eval()
    total = 0.0
    for i in range(0, len(x), bs):
        total += ((model(x[i:i + bs]) - y[i:i + bs]) ** 2).sum().item()
    return total / y.numel()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dir(CKPT_DIR)
    ensure_dir(RESULTS_DIR)

    data = load_dataset(args.data)
    xtr, ytr = torch.from_numpy(data["q_train"]), torch.from_numpy(data["y_train"])
    xva, yva = torch.from_numpy(data["q_val"]), torch.from_numpy(data["y_val"])
    in_dim, out_dim = xtr.shape[1], ytr.shape[1]

    pca, mean, comps = None, None, None
    if args.mode == "pca":
        pca = PCA().fit(data["y_train"])  # TRAIN offsets only -> no leakage
        mean, comps = pca.get(args.k)
        print(f"PCA: {comps.shape[0]} components keep "
              f"{100 * pca.cumulative_variance(comps.shape[0]):.2f}% of train variance")

    cfg = dict(mode=args.mode, k=int(comps.shape[0]) if comps is not None else None,
               hidden=list(args.hidden), in_dim=in_dim, out_dim=out_dim)
    model = build_model(cfg, mean, comps)  # built on CPU first
    print(f"model '{args.name}': {cfg} | trainable params: {n_trainable_params(model):,} | device: {device}")

    hist = {"train_mse": [], "val_mse": []}
    t0 = time.time()

    if args.closed_form:
        target = torch.from_numpy(pca.transform(data["y_train"], cfg["k"])) if pca else ytr
        fit_closed_form(model, xtr, target)
        va = eval_mse(model, xva, yva)
        hist["train_mse"].append(eval_mse(model, xtr, ytr))
        hist["val_mse"].append(va)
        best, best_ep = va, 0
        save_checkpoint(CKPT_DIR / f"{args.name}.pt", model, cfg, epoch=0, val_mse=va)
        print(f"closed-form fit | val MSE {va:.5f}")
    else:
        model = model.to(device)
        xtr, ytr, xva, yva = xtr.to(device), ytr.to(device), xva.to(device), yva.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # PCA buffers are not parameters
        N = len(xtr)
        best, best_ep = float("inf"), 0
        for ep in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(N, device=device)
            running = 0.0
            for i in range(0, N, args.batch_size):
                idx = perm[i:i + args.batch_size]
                loss = F.mse_loss(model(xtr[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            tr, va = running / N, eval_mse(model, xva, yva)
            hist["train_mse"].append(tr)
            hist["val_mse"].append(va)
            if va < best:
                best, best_ep = va, ep
                save_checkpoint(CKPT_DIR / f"{args.name}.pt", model, cfg, epoch=ep, val_mse=va)
            if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
                print(f"epoch {ep:4d} | train MSE {tr:.5f} | val MSE {va:.5f} | best {best:.5f} (ep {best_ep})")
            if args.patience and ep - best_ep >= args.patience:
                print(f"early stop at epoch {ep}")
                break

    hist.update(best_val_mse=best, best_epoch=best_ep, train_seconds=time.time() - t0, config=cfg)
    save_json(hist, RESULTS_DIR / f"{args.name}_history.json")
    print(f"done in {hist['train_seconds']:.0f}s | best val MSE {best:.5f} at epoch {best_ep} "
          f"| checkpoint: checkpoints/{args.name}.pt")


if __name__ == "__main__":
    main()
