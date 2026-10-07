# Data-Driven Cloth for Interactive Applications (ML Mini Project)

**UE24CS352A Machine Learning – Mini-Project 2026**

| Team member | SRN |
|---|---|
| Tejas Ponnappa | PES1UG24CS498 |
| Mukesh Sheregar | PES1UG25CS828 |

Reproduction of the pipeline from Xue & Wu, *"Data-Driven Clothing for Interactive Applications"*
(Stanford CS229 project report, Spring 2021,
<https://cs229.stanford.edu/proj2021spr/report2/81970649.pdf>): predict how a garment
deforms from the character's pose, fast enough for interactive use.

**Pipeline:** input (SMPL pose 72 + body shape 10 + garment style 4 = 86 numbers) → MLP (ReLU, hidden 128 → 256)
→ *k* PCA coefficients → fixed linear layer (PCA basis) → per-vertex offsets of the garment.
**Baseline:** linear regression (an MLP with zero hidden layers), solved in closed form.
**Ablation:** direct offsets (no PCA) and *k* = 256, 128, 64.

## Dataset
The paper's coat data (Blender skinning + PhysBAM simulation) is not public, so we use the public
**TailorNet** dataset of simulated garments (Patel et al., CVPR 2020,
<https://github.com/zycliao/TailorNet_dataset>). We use the **male short-pants** garment
(`short-pant_male.zip`, about 2 GB): 2,710 vertices, so 8,130 outputs.

* From the 40 body-shape / garment-style combinations in the download we sample up to 100 poses each: 3,833 poses.
* Target = simulated garment minus the rest garment of that shape and style, in cm.
* Outlier filter (largest vertex offset above median + 5 robust standard deviations = 19.8 cm) removes 91 poses.
* Random split of the remaining 3,742: 2,245 train / 748 val / 749 test. PCA is fit on train only.
* Because poses from different bodies and styles are mixed, the input also contains the body shape (10) and garment style (4).

**Download:** get `short-pant_male.zip` from the dataset page linked above and unzip it **outside** this repo
(it is not committed). The unzipped folder must contain `pose/`, `shape/`, `style/` and `style_shape/`.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m src.data_tailornet --root "C:\path\to\short-pant_male"     # build data/tailornet.npz
python -m src.train --name tn_pca128 --mode pca --k 128 --hidden 128 256 --epochs 1000 --data data/tailornet.npz
python -m src.evaluate --names tn_linreg tn_direct tn_pca256 tn_pca128 tn_pca64 --data data/tailornet.npz --out_dir results/tailornet
python demo.py --model tn_pca128 --compare tn_linreg --data data/tailornet.npz   # live demo (arrow keys change pose)
```
Or everything at once (Windows): `.\run_all_tn.bat "C:\path\to\short-pant_male"`
(trains `tn_linreg`, `tn_direct`, `tn_pca256`, `tn_pca128`, `tn_pca64`; about 45 minutes on a laptop CPU).
For a quick run use `--epochs 200` (about 10 minutes). On Linux/macOS run the python commands in `run_all_tn.bat` one by one.

## Results
Test set (749 poses), single run (seed 0), Adam lr 1e-4, batch 32, best-validation checkpoint.
Exact numbers are in `results/tailornet/summary.csv`. MSE is in cm²; predicting the training mean gives 1.868.

| model | trainable params | test MSE, 200 epochs | test MSE, 1000 epochs | mean vertex error, 1000 ep (cm) |
|-------|------------------|----------------------|-----------------------|---------------------------------|
| linear regression (closed form) | 707,310 | 0.907 | 0.907 | 1.207 |
| MLP, direct offsets | 2,133,570 | 0.475 | **0.379** | 0.753 |
| MLP + PCA, k = 256 | 109,952 | 0.662 | 0.403 | 0.788 |
| MLP + PCA, k = 128 | 77,056 | 0.662 | 0.406 | 0.793 |
| MLP + PCA, k = 64 | 60,608 | 0.660 | 0.410 | 0.799 |

Takeaways: the MLPs beat linear regression by 55-58% in MSE (the best network is 2.4x better; the paper reports 2.6x).
PCA gives a 19-35x smaller network for a 6-8% higher MSE than direct prediction. The paper found PCA slightly *better*
than direct prediction and k = 128 best; we do not reproduce that. The PCA networks were still improving at epoch 1000
(the gap to direct shrank from about 39% at 200 epochs to 6-8% at 1000), so longer training may change this (not tested).
The three values of k differ by less than 2%, which is within single-run noise. Limitations: one garment, one seed,
a random split (test poses share body shapes and styles with training). See the report in `docs/`.

Plots (`results/tailornet/`): `explained_variance.png`, `training_curves.png`, `ablation.png`.

## Repo layout
```
src/data_tailornet.py  loads TailorNet, builds offsets, outlier filter, split
src/data.py            outlier filter helper (also holds a synthetic-data generator used only to test the pipeline early on)
src/pca.py             PCA on vertex offsets
src/model.py           MLP + fixed PCA decoder, closed-form linear baseline
src/train.py           training (Adam, MSE, lr 1e-4, batch 32)
src/evaluate.py        test metrics, timing, plots
demo.py                interactive predicted-vs-ground-truth viewer (point cloud)
run_all_tn.bat         full pipeline
docs/                  slides and report
```

## Team contributions
- **Tejas Ponnappa (PES1UG24CS498):** TailorNet data loading, outlier filtering and train/val/test split (`src/data_tailornet.py`, `src/data.py`, `src/utils.py`); training pipeline and the model runs for the ablation (`src/train.py`, `run_all_tn.bat`); evaluation, metrics and plots (`src/evaluate.py`); live demo (`demo.py`); repository setup, results and report.
- **Mukesh Sheregar (PES1UG25CS828):** PCA of the vertex offsets (`src/pca.py`); network architecture with the fixed PCA layer and the linear-regression baseline (`src/model.py`); added modules to the repository.
- **Both:** presenting the project at the review.