@echo off
REM Retrain the 4 neural models for 600 epochs on the already-built data\tailornet.npz, then evaluate.
REM (linear regression is closed-form, so it does not need retraining.)  Takes ~25-30 min.
set D=--data data/tailornet.npz
set T=--hidden 128 256 --epochs 1000 --batch_size 32 %D%
python -m src.train --name tn_pca64 --mode pca --k 64 %T% || exit /b 1
python -m src.train --name tn_pca128 --mode pca --k 128 %T% || exit /b 1
python -m src.train --name tn_pca256 --mode pca --k 256 %T% || exit /b 1
python -m src.train --name tn_direct --mode direct %T% || exit /b 1
python -m src.evaluate --names tn_linreg tn_direct tn_pca256 tn_pca128 tn_pca64 --data data/tailornet.npz --out_dir results/tailornet