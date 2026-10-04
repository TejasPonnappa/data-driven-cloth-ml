"""MLP: pose quaternions -> PCA coefficients -> (fixed linear layer) -> vertex offsets.

mode="pca":    net outputs k coefficients; a FIXED (non-trainable) linear layer
               built from the PCA components turns them into offsets.
mode="direct": net outputs the offsets directly (no PCA).
hidden=()      zero hidden layers -> plain linear regression baseline.
"""
import numpy as np
import torch
import torch.nn as nn


class ClothNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 256), pca_mean=None, pca_components=None):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.use_pca = pca_components is not None
        k = pca_components.shape[0] if self.use_pca else out_dim
        layers.append(nn.Linear(d, k))
        self.net = nn.Sequential(*layers)
        if self.use_pca:
            # buffers = saved with the model but never trained
            self.register_buffer("pca_mean", torch.as_tensor(np.asarray(pca_mean), dtype=torch.float32))
            self.register_buffer("pca_components", torch.as_tensor(np.asarray(pca_components), dtype=torch.float32))

    def forward_coeffs(self, x):
        return self.net(x)

    def forward(self, x):
        z = self.net(x)
        if self.use_pca:
            z = z @ self.pca_components + self.pca_mean
        return z


def n_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, pca_mean=None, pca_components=None):
    """cfg: dict(mode, k, hidden, in_dim, out_dim)."""
    if cfg["mode"] == "direct":
        pca_mean = pca_components = None
    elif pca_components is None:  # when loading: real values come from the state_dict
        pca_components = np.zeros((cfg["k"], cfg["out_dim"]), np.float32)
        pca_mean = np.zeros(cfg["out_dim"], np.float32)
    return ClothNet(cfg["in_dim"], cfg["out_dim"], tuple(cfg["hidden"]), pca_mean, pca_components)


def save_checkpoint(path, model, cfg, **extra):
    torch.save({"config": cfg, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def fit_closed_form(model, x, target):
    """Least-squares fit of a zero-hidden-layer model (the converged linear regression).

    target = offsets (direct mode) or PCA coefficients (pca mode).
    """
    assert len(model.net) == 1, "closed form only works with no hidden layers"
    x, target = x.double().cpu(), target.double().cpu()
    A = torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], dim=1)
    sol = torch.linalg.lstsq(A, target).solution  # (in+1, out)
    model.net[0].weight.copy_(sol[:-1].T.float())
    model.net[0].bias.copy_(sol[-1].float())
