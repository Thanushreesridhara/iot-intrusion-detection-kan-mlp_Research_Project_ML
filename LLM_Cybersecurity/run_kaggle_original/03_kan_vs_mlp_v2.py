"""
03_kan_vs_mlp.py
Controlled KAN vs MLP comparison on CICIoT2023 (uses train/val/test.parquet from 01_prepare_data.py).

Research questions
  RQ1 (accuracy vs cost): At matched parameter budgets, do KANs reach higher macro-F1 than MLPs on
      leakage-controlled CICIoT2023, especially on the rare classes (BruteForce, Web-based)?
      What do they cost in training time and single-row CPU inference latency? XGBoost = reference.
  RQ2 (data efficiency): Does a KAN need fewer labelled training samples than an MLP to reach a given
      macro-F1? (learning curve over training-set size, multiple seeds)

Fairness controls (state these in the README):
  * identical preprocessing (signed log1p -> min-max to [-1,1] fitted on train only -> clip)
  * identical loss weighting (sqrt-balanced class weights), optimiser, batch size, epoch budget,
    early stopping on validation macro-F1
  * MLP width is searched so its parameter count matches the KAN's (within ~3%)
  * same fixed test split for every run; seeds change init, batch order and the train subsample
  * XGBoost included as a strong tabular reference

Usage:
  OUT_DIR=./out python 03_kan_vs_mlp.py            # full run, resumable (results.jsonl)
  QUICK=1 OUT_DIR=./out python 03_kan_vs_mlp.py    # tiny smoke test
"""

# %% Imports and config
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import xgboost as xgb

OUT_DIR = os.environ.get("OUT_DIR", "./out")
QUICK = os.environ.get("QUICK") == "1"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2" if QUICK else "0,1,2,3,4").split(",")]
_sizes = os.environ.get("SIZES")  # e.g. SIZES=full  or  SIZES=500,2000,full
if _sizes:
    SIZES = [None if t.strip() == "full" else int(t) for t in _sizes.split(",")]
else:
    SIZES = [500, 2000] if QUICK else [500, 2000, 10000, 50000, None]  # None = full training set
TAG = os.environ.get("TAG", "")  # suffix for the results file, keeps experiments separate
MAX_EPOCHS = 3 if QUICK else int(os.environ.get("MAX_EPOCHS", 25))
PATIENCE = int(os.environ.get("PATIENCE", 5))
BATCH = 1024
LR = 2e-3
VAL_MAX = int(os.environ.get("VAL_MAX", 100_000))  # val subsample used for early stopping
TEST_MAX = int(os.environ["TEST_MAX"]) if os.environ.get("TEST_MAX") else None
GRID, ORDER = 5, 3
KAN_HIDDEN = [int(h) for h in os.environ.get("KAN_HIDDEN", "8,16").split(",")]  # parameter budgets (full-data runs); learning curve uses the largest
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(OUT_DIR, f"results_kan_mlp{TAG}.jsonl")
print("device:", DEVICE, "| seeds:", SEEDS, "| sizes:", SIZES)

# %% Data + preprocessing (identical for all models)
meta = json.load(open(os.path.join(OUT_DIR, "meta.json")))
CLASSES, FEATS = meta["classes"], meta["features"]
K, D = len(CLASSES), len(FEATS)

train = pd.read_parquet(os.path.join(OUT_DIR, "train.parquet"))
val = pd.read_parquet(os.path.join(OUT_DIR, "val.parquet"))
test = pd.read_parquet(os.path.join(OUT_DIR, "test.parquet"))


def stratified_cap(df, n, seed):
    if n is None or len(df) <= n:
        return df
    parts = []
    for _, g in df.groupby("y"):
        parts.append(g.sample(n=min(len(g), max(20, round(n * len(g) / len(df)))), random_state=seed))
    return pd.concat(parts)


val = stratified_cap(val, VAL_MAX, 0)
if TEST_MAX:
    test = stratified_cap(test, TEST_MAX, 0)

med = train[FEATS].median()  # NaN (from inf fix) -> train median


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


tr_log = slog(train[FEATS].fillna(med).values.astype("float64"))
lo, hi = tr_log.min(0), tr_log.max(0)
span = np.where(hi - lo < 1e-9, 1.0, hi - lo)


def prep(df):
    a = slog(df[FEATS].fillna(med).values.astype("float64"))
    a = 2 * (a - lo) / span - 1
    return np.clip(a, -1, 1).astype("float32")


X_va, y_va = prep(val), val["y"].values
X_te, y_te = prep(test), test["y"].values
TRAIN_ALL = train.reset_index(drop=True)
X_train_all = prep(TRAIN_ALL)
y_train_all = TRAIN_ALL["y"].values
print(f"train {X_train_all.shape}, val {X_va.shape}, test {X_te.shape}")


# %% Models
class KANLinear(nn.Module):
    """B-spline KAN layer (efficient formulation: spline bases computed once, then one matmul).
    phi(x) = w_base * silu(x) + sum_j c_j * B_j(x), grid on [-1, 1]. Simplified: no spline-scaler, no
    sparsity regularisation (document this)."""

    def __init__(self, n_in, n_out, grid=GRID, order=ORDER, lo=-1.0, hi=1.0):
        super().__init__()
        self.n_in, self.n_out, self.grid_size, self.order = n_in, n_out, grid, order
        h = (hi - lo) / grid
        g = torch.arange(-order, grid + order + 1) * h + lo
        self.register_buffer("grid", g.expand(n_in, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(n_out, n_in))
        self.spline_weight = nn.Parameter(torch.empty(n_out, n_in, grid + order))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(n_in))

    def bases(self, x):
        g = self.grid
        x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = (x - g[:, : -(k + 1)]) / (g[:, k:-1] - g[:, : -(k + 1)]) * b[:, :, :-1] + (
                g[:, k + 1 :] - x
            ) / (g[:, k + 1 :] - g[:, 1:-k]) * b[:, :, 1:]
        return b

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(self.bases(x).reshape(x.size(0), -1), self.spline_weight.reshape(self.n_out, -1))
        return base + spline


class KAN(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(KANLinear(a, b) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_in, width, d_out, depth=2):
        super().__init__()
        dims = [d_in] + [width] * depth + [d_out]
        mods = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            mods.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_mlp(target):
    best = min(range(4, 512), key=lambda w: abs(n_params(MLP(D, w, K)) - target))
    return MLP(D, best, K), best


# %% Training / evaluation helpers
def class_weights(y):
    cnt = np.bincount(y, minlength=K).astype("float64")
    w = (len(y) / (K * np.maximum(cnt, 1))) ** 0.5  # sqrt-balanced
    return w / w.mean()


@torch.no_grad()
def predict(model, X, bs=16384):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i : i + bs]).to(DEVICE)).argmax(1).cpu())
    return torch.cat(out).numpy()


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0)


def train_torch(model, X, y, seed):
    torch.manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y), dtype=torch.float32, device=DEVICE))
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y.astype("int64")).to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(DEVICE)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        f1 = mf1(y_va, predict(model, X_va))
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return time.time() - t0, ep + 1, best


def cpu_latency_ms(model, X, n=200):
    m = copy.deepcopy(model).cpu().eval()
    torch.set_num_threads(1)
    lat = []
    with torch.no_grad():
        for row in X[:n]:
            t = time.perf_counter()
            m(torch.from_numpy(row[None]))
            lat.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


def run_xgb(X, y, seed):
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", device=DEVICE, early_stopping_rounds=20,
        random_state=seed, eval_metric="mlogloss",
    )
    w = class_weights(y)[y]
    t0 = time.time()
    clf.fit(X, y, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)
    train_time = time.time() - t0
    pred = clf.predict(X_te)
    clf.set_params(device="cpu")
    lat = []
    for row in X_te[:200]:
        t = time.perf_counter()
        clf.predict(row[None])
        lat.append((time.perf_counter() - t) * 1000)
    n_nodes = int(clf.get_booster().trees_to_dataframe().shape[0])
    return pred, train_time, int(clf.best_iteration) + 1, n_nodes, float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


# %% Experiment loop (resumable)
done = set()
if os.path.exists(RESULTS):
    for line in open(RESULTS):
        r = json.loads(line)
        done.add(f'{r["model"]}|{r["n_key"]}|{r["seed"]}')


def save(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def configs_for(n):
    cfgs = [("XGB", None)]
    hidden = KAN_HIDDEN if n is None else [KAN_HIDDEN[-1]]
    for h in hidden:
        cfgs += [(f"KAN-h{h}", h), (f"MLP-h{h}", h)]  # MLP-hX is parameter-matched to KAN-hX
    return cfgs


for n in SIZES:
    for seed in SEEDS:
        sub = stratified_cap(TRAIN_ALL, n, seed)
        idx = sub.index.values
        X, y = X_train_all[idx], y_train_all[idx]
        for name, h in configs_for(n):
            key = f"{name}|{n}|{seed}"
            if key in done:
                continue
            if name == "XGB":
                pred, tt, ep, size_stat, p50, p95 = run_xgb(X, y, seed)
                rec_extra = {"params": size_stat, "epochs": ep, "train_s": tt}
            else:
                torch.manual_seed(seed)
                kan = KAN([D, h, K])
                model = kan if name.startswith("KAN") else matched_mlp(n_params(kan))[0]
                tt, ep, best_val = train_torch(model, X, y, seed)
                pred = predict(model, X_te)
                p50, p95 = cpu_latency_ms(model, X_te)
                rec_extra = {"params": n_params(model), "epochs": ep, "train_s": tt, "best_val_macro_f1": best_val}
            per_class = f1_score(y_te, pred, average=None, labels=list(range(K)), zero_division=0)
            rec = {
                "model": name, "n_key": n, "n_train": n if n is not None else len(TRAIN_ALL), "seed": seed,
                "macro_f1": float(mf1(y_te, pred)), "accuracy": float((pred == y_te).mean()),
                "per_class_f1": {c: float(v) for c, v in zip(CLASSES, per_class)},
                "lat_ms_p50": p50, "lat_ms_p95": p95, **rec_extra,
            }
            save(rec)
            print(f'{key:28s} macroF1={rec["macro_f1"]:.4f} params={rec["params"]:>8} '
                  f'train={rec["train_s"]:.0f}s lat={p50:.2f}ms')
print("done ->", RESULTS)
