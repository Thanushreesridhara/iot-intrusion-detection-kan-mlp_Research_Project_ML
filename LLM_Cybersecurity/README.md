# LLM_Cybersecurity

IoT intrusion detection on CICIoT2023: leakage-controlled data preparation, XGBoost baseline,
and a controlled KAN vs MLP comparison (accuracy vs cost, data efficiency).

Pipeline: `01_prepare_data.py` -> `02_baseline_xgboost.py` -> `03_kan_vs_mlp.py` -> `04_analyze_kan_mlp.py`.
Data, trained models and parquet splits are not tracked (see `.gitignore`).
See `SESSION_NOTES.md` for the results summary.
