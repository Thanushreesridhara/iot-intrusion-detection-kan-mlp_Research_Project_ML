# Session notes (2026-10-02)

Project dir: /home/thanu/Research Project  (venv: .venv — xgboost, torch+CUDA, pandas, sklearn, scipy, tabulate)
Nothing committed/pushed to git. GPU: RTX 5070 Ti used for XGBoost/torch.

## Run A — HuggingFace mirror (bencorn/CIC-IoT-2023), top-level scripts + out/
- download_data.py: fetches 309 per-class CSVs, adds `label` from folder name -> data/CICIoT2023/
- 01_prepare_data.py modified: single-class files => file-split for classes with >=3 files, contiguous row-block split otherwise; label aliases; inf->NaN
- 02 XGBoost: test macro-F1 0.547 (acc 0.714). Web-based/BruteForce ~0.05/0.13 (too few train rows).

## Run B — Kaggle data (C:\Users\thanu\Downloads\archive\wataiData\csv\CICIoT2023, 169 files) in run_kaggle_original/
- Original 01 (split by file) + inf->NaN fix, original 02: XGBoost test macro-F1 0.829 (acc 0.994)
- 03/04 KAN vs MLP (results_kan_mlp.jsonl, tables, learning_curve.png): full-data macro-F1
  XGB 0.837 | KAN-h16 0.651 | KAN-h8 0.639 | MLP-h16 0.621 | MLP-h8 0.617
  KAN-MLP paired diff positive in 5/5 seeds at every size (+0.016..+0.060); 500-row CI includes 0.
- 03/04 *_v2.py (env TAG/SIZES/PATIENCE/KAN_HIDDEN) re-run with TAG=_v2 -> *_v2 outputs; same numbers. Earlier results untouched.

## Ideas for next time
- Neural models far behind XGB on rare classes (BruteForce/Web-based); try more epochs/capacity, focal loss, tuning.
- XGB in run B used all 600 trees w/o early stopping -> try more n_estimators.
