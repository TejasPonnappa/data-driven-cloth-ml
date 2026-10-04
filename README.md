# Data-Driven Cloth for Interactive Applications (ML Mini Project)

**UE24CS352A Machine Learning – Mini-Project 2026**

| Team member | SRN |
|---|---|
| Tejas Ponnappa | PES1UG24CS498 |
| Mukesh Sheregar| PES1UG24CS828 |

Reproduction of the pipeline from Xue & Wu, *"Data-Driven Clothing for Interactive Applications"*
(Stanford CS229 project report, Spring 2021,
<https://cs229.stanford.edu/proj2021spr/report2/81970649.pdf>): predict how a piece of cloth
deforms from the character's pose, fast enough for interactive use.

**Pipeline:** 14 joint quaternions (56 numbers) → MLP (ReLU, hidden 128 → 256) → *k* PCA
coefficients → fixed linear layer (PCA basis) → per-vertex offsets.
**Baseline:** linear regression (an MLP with zero hidden layers).

## Dataset
The paper's data (Blender skinning + PhysBAM simulation) is not public, so we use a
**synthetic stand-in** with the same structure (`src/data.py`): random joint rotations →
a fixed nonlinear function → offsets of a 20×25-vertex cloth panel (1500 outputs),
plus noise and a few "exploded simulation" outliers that are removed with a
distance threshold. Split: 6000 train / 2000 val / rest test. PCA is fit on train only.

> To use a real dataset, save it in the same `.npz` format (see `build_dataset` in `src/data.py`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python -m src.data --n 10000 --seed 0                       # generate dataset -> data/cloth.npz
python -m src.train --name pca128 --mode pca --k 128        # train one model
python -m src.evaluate                                      # test metrics + plots in results/
python demo.py --model pca128 --compare linreg              # live demo (arrow keys change pose)
```
Or everything at once (Windows): `run_all.bat`

Models trained for the ablation: `linreg`, `direct` (no PCA), `pca256`, `pca128`, `pca64`.
The reported results use 1000 epochs for the four MLPs (as in the paper), which is what `run_all.bat` uses
(about 8-9 minutes per model on our laptop).
On Linux/macOS, run the python commands listed in `run_all.bat` one by one.

## Results
Test set, single run (seed 0), 1000 epochs, Adam lr 1e-4, batch 32. Exact numbers are in `results/summary.csv`.

| model | trainable params | test MSE (cm²) | mean vertex error (cm) | train time |
|-------|------------------|----------------|------------------------|-----------|
| predict train mean | 0 | 1.831 | – | – |
| linear regression (closed form) | 85,500 | 0.392 | 0.991 | – |
| MLP, direct offsets | 425,820 | 0.251 | 0.789 | 537 s |
| MLP + PCA, k = 256 | 106,112 | 0.253 | 0.794 | 520 s |
| MLP + PCA, k = 128 | 73,216 | 0.254 | 0.795 | 459 s |
| MLP + PCA, k = 64 | 56,768 | 0.250 | 0.790 | 532 s |

Takeaways: the MLPs beat linear regression by 35-36% in MSE. All four MLPs are within 1.5% of each other
(within single-run noise), so PCA costs no measurable accuracy while making the network up to 7.5x smaller
(k = 64: 56.8k vs 425.8k parameters). Unlike the paper (where k = 128 was best and PCA beat direct prediction),
we see no preferred k; the data is synthetic and smooth, which is a likely reason. See the report in `docs/`.

Plots (`results/`): `explained_variance.png`, `training_curves.png`, `ablation.png`, `per_vertex_error_*.png`.

## Repo layout
```
src/data.py      dataset generation, outlier filter, split
src/pca.py       PCA on vertex offsets
src/model.py     MLP + fixed PCA decoder, closed-form linear baseline
src/train.py     training (Adam, MSE, lr 1e-4, batch 32)
src/evaluate.py  test metrics, timing, plots
demo.py          interactive predicted-vs-ground-truth viewer
docs/            slides and report
```

## Team contributions
- **Tejas Ponnappa (PES1UG24CS498):** dataset generation, outlier filtering and train/val/test split (`src/data.py`, `src/utils.py`); training pipeline and the model runs for the ablation (`src/train.py`, `run_all.bat`); evaluation, metrics and plots (`src/evaluate.py`); live demo (`demo.py`); repository setup, results and report.
- **Mukesh Sheregar (PES1UG24CS828):** PCA of the vertex offsets (`src/pca.py`); network architecture with the fixed PCA layer and the linear-regression baseline (`src/model.py`); added the PCA, model and evaluation modules to the repository and re-ran the evaluation (`python -m src.evaluate`) to check the results.
- **Both:** presenting the project at the review.
