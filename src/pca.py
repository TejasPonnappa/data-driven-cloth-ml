"""PCA of the vertex offsets (fit on TRAIN offsets only).

We do one full SVD and keep every component; `get(k)` then returns the top-k.
That makes the k = 256 / 128 / 64 ablation cheap.
"""
import numpy as np


class PCA:
    def __init__(self):
        self.mean_ = None
        self.components_ = None  # (n_comp, D), orthonormal rows
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt.astype(np.float32)
        self.mean_ = self.mean_.astype(np.float32)
        var = (S ** 2) / (len(X) - 1)
        self.explained_variance_ = var
        self.explained_variance_ratio_ = var / var.sum()
        return self

    def get(self, k):
        """(mean, top-k components) as float32 arrays."""
        k = min(k, len(self.components_))
        return self.mean_, self.components_[:k]

    def transform(self, X, k):
        mean, comps = self.get(k)
        return (np.asarray(X, dtype=np.float32) - mean) @ comps.T

    def inverse_transform(self, Z):
        k = Z.shape[1]
        mean, comps = self.get(k)
        return Z @ comps + mean

    def reconstruct(self, X, k):
        return self.inverse_transform(self.transform(X, k))

    def reconstruction_mse(self, X, k):
        X = np.asarray(X, dtype=np.float32)
        return float(np.mean((self.reconstruct(X, k) - X) ** 2))

    def cumulative_variance(self, k):
        return float(self.explained_variance_ratio_[:k].sum())
