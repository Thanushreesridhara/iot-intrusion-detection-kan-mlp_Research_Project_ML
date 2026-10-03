"""
02_baseline_xgboost.py
Train the main detector (XGBoost) on the prepared splits and report honest metrics.

Outputs (in OUT_DIR):
  xgb_model.json          trained model (used later by the triage service)
  metrics_xgb.json        macro-F1, per-class P/R/F1, latency, top features
  confusion_xgb_test.png  row-normalised confusion matrix on the test split
Headline metric = macro-F1 (accuracy is dominated by DDoS in natural traffic).
"""

# %% Config
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

OUT_DIR = os.environ.get("OUT_DIR", "out")
SEED = int(os.environ.get("SEED", 42))

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
print("XGBoost device:", DEVICE)

# %% Load data
with open(os.path.join(OUT_DIR, "meta.json")) as f:
    meta = json.load(f)
CLASSES, FEATS = meta["classes"], meta["features"]

train = pd.read_parquet(os.path.join(OUT_DIR, "train.parquet"))
val = pd.read_parquet(os.path.join(OUT_DIR, "val.parquet"))
test = pd.read_parquet(os.path.join(OUT_DIR, "test.parquet"))

X_tr, y_tr = train[FEATS].values, train["y"].values
X_va, y_va = val[FEATS].values, val["y"].values
X_te, y_te = test[FEATS].values, test["y"].values
print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

# %% Train (class-balanced sample weights, early stopping on val)
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    n_estimators=600,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device=DEVICE,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=SEED,
)
t0 = time.time()
clf.fit(
    X_tr,
    y_tr,
    sample_weight=compute_sample_weight("balanced", y_tr),
    eval_set=[(X_va, y_va)],
    verbose=50,
)
train_time = time.time() - t0
print(f"training time: {train_time:.1f}s, best iteration: {clf.best_iteration}")


# %% Evaluate
def evaluate(name, X, y):
    pred = clf.predict(X)
    labels = list(range(len(CLASSES)))
    rep = classification_report(
        y, pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0
    )
    print(f"\n=== {name} ===")
    print(
        classification_report(
            y, pred, labels=labels, target_names=CLASSES, digits=4, zero_division=0
        )
    )
    return pred, {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "per_class": {c: rep[c] for c in CLASSES},
    }


_, m_val = evaluate("validation", X_va, y_va)
pred_te, m_te = evaluate("test", X_te, y_te)

cm = confusion_matrix(y_te, pred_te, labels=list(range(len(CLASSES))), normalize="true")
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
    ax=ax, xticks_rotation=45, values_format=".2f", colorbar=False
)
ax.set_title("XGBoost - test split (row-normalised)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_xgb_test.png"), dpi=150)

# %% Inference cost (CPU, single-row = realistic API call)
clf.set_params(device="cpu")
lat = []
for row in X_te[:300]:
    t = time.perf_counter()
    clf.predict(row.reshape(1, -1))
    lat.append((time.perf_counter() - t) * 1000)
t = time.perf_counter()
clf.predict(X_te[:10000])
batch_rows_per_s = min(10000, len(X_te)) / (time.perf_counter() - t)
latency = {
    "single_row_ms_p50": float(np.percentile(lat, 50)),
    "single_row_ms_p95": float(np.percentile(lat, 95)),
    "batch_rows_per_second": float(batch_rows_per_s),
}
print("\nLatency (CPU):", latency)

# %% Save model + metrics + top features (the LLM stage will use top features per alert)
clf.save_model(os.path.join(OUT_DIR, "xgb_model.json"))
imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nTop 15 features:\n", imp.head(15))

with open(os.path.join(OUT_DIR, "metrics_xgb.json"), "w") as f:
    json.dump(
        {
            "val": m_val,
            "test": m_te,
            "train_time_s": train_time,
            "best_iteration": int(clf.best_iteration),
            "latency_cpu": latency,
            "top_features": imp.head(15).round(4).to_dict(),
            "seed": SEED,
        },
        f,
        indent=2,
    )
print("\nSaved model, metrics and confusion matrix to", OUT_DIR)
