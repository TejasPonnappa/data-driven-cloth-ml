@echo off
REM Windows version of run_all.sh. Run from the repo root:  .\run_all.bat

python -m src.data --n 10000 --seed 0 || exit /b 1

python -m src.train --name linreg --mode direct --hidden --closed_form || exit /b 1
python -m src.train --name direct --mode direct --hidden 128 256 --epochs 1000 || exit /b 1
python -m src.train --name pca256 --mode pca --k 256 --hidden 128 256 --epochs 1000 || exit /b 1
python -m src.train --name pca128 --mode pca --k 128 --hidden 128 256 --epochs 1000 || exit /b 1
python -m src.train --name pca64 --mode pca --k 64 --hidden 128 256 --epochs 1000 || exit /b 1

python -m src.evaluate