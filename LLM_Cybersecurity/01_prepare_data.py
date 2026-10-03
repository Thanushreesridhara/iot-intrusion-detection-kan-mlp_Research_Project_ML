"""
01_prepare_data.py
CICIoT2023 (per-class CSV parts, see download_data.py) -> leakage-controlled train / val / test parquet files.

Design decisions (write these in your README, they are part of the method):
  1. Split BY FILE, not by row. Each CSV part is a separate chunk of the dataset,
     so train/val/test never share a file. Files are split per class (stratified by
     file). Classes with fewer than 3 files (most rare attacks) are split into
     CONTIGUOUS row blocks (70/15/15) of their file(s) instead, which keeps
     temporally adjacent flows together.
  2. Train set is class-capped (max N rows per class per unit) so rare attacks
     (BruteForce, Web-based) are not drowned by DDoS.
  3. Val/test keep the NATURAL class distribution (random fraction of each unit),
     so the numbers reflect a realistic traffic mix. Report macro-F1 + per-class.
  4. Exact-duplicate feature rows are removed inside each split, and any val/test
     row that also appears in train is dropped (cross-split leakage control).
  5. The original 34-class label is kept as `subtype` for the LLM stage later.

Smoke test: set env MAX_FILES=12 to run in a couple of minutes.
"""

# %% Config
import glob
import json
import os
import time

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = os.environ.get("DATA_DIR", "data/CICIoT2023")
OUT_DIR = os.environ.get("OUT_DIR", "out")
MAX_FILES = int(os.environ["MAX_FILES"]) if os.environ.get("MAX_FILES") else None

TRAIN_CAP_PER_CLASS_PER_FILE = int(os.environ.get("TRAIN_CAP", 600))
EVAL_FRAC = float(os.environ.get("EVAL_FRAC", 0.30))  # fraction of rows kept from val/test units
FILE_SPLIT = (0.70, 0.15, 0.15)  # train / val / test

# %% Label mapping: 34 original labels -> 8 coarse classes (as in the CICIoT2023 paper)
GROUPS = {
    "Benign": ["BenignTraffic"],
    "DDoS": [
        "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-PSHACK_Flood",
        "DDoS-SYN_Flood", "DDoS-RSTFINFlood", "DDoS-SynonymousIP_Flood",
        "DDoS-ICMP_Fragmentation", "DDoS-UDP_Fragmentation", "DDoS-ACK_Fragmentation",
        "DDoS-HTTP_Flood", "DDoS-SlowLoris",
    ],
    "DoS": ["DoS-UDP_Flood", "DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-HTTP_Flood"],
    "Mirai": ["Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"],
    "Recon": [
        "Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan", "Recon-PingSweep",
        "VulnerabilityScan",
    ],
    "Spoofing": ["MITM-ArpSpoofing", "DNS_Spoofing"],
    "Web-based": [
        "BrowserHijacking", "XSS", "Uploading_Attack", "SqlInjection",
        "CommandInjection", "Backdoor_Malware",
    ],
    "BruteForce": ["DictionaryBruteForce"],
}
CLASSES = list(GROUPS)  # index = class id
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
FINE2COARSE = {fine: coarse for coarse, fines in GROUPS.items() for fine in fines}
# folder names in the HF mirror that differ from the official label names
ALIASES = {
    "Benign_Final": "BenignTraffic",
    "DDoS-PSHACK_FLOOD": "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFLOOD": "DDoS-RSTFINFlood",
}


# %% Loading helpers
def load_unit(path: str, rows: tuple, mode: str, seed: int) -> pd.DataFrame:
    """Load one split unit: rows[0]:rows[1] as fractions of the file."""
    df = pd.read_csv(path)
    lo, hi = int(len(df) * rows[0]), int(len(df) * rows[1])
    df = df.iloc[lo:hi].reset_index(drop=True)
    df["label"] = df["label"].replace(ALIASES)

    unknown = set(df["label"].unique()) - set(FINE2COARSE)
    if unknown:
        raise ValueError(f"Unmapped labels in {os.path.basename(path)}: {unknown}")

    feats = [c for c in df.columns if c != "label"]
    # float32 overflow / raw inf -> NaN (XGBoost treats NaN as missing)
    out = df[feats].astype("float32").replace([np.inf, -np.inf], np.nan)
    out["subtype"] = df["label"].values  # original fine-grained label (string)
    out["y"] = df["label"].map(FINE2COARSE).map(CLASS_ID).astype("int8").values

    if mode == "train":
        # keep at most TRAIN_CAP_PER_CLASS_PER_FILE random rows of each class
        rnd = np.random.default_rng(seed).random(len(out))
        rank = pd.Series(rnd).groupby(out["y"].values).rank(method="first").values
        out = out[rank <= TRAIN_CAP_PER_CLASS_PER_FILE]
    else:
        out = out.sample(frac=EVAL_FRAC, random_state=seed)
    return out.reset_index(drop=True)


# %% Split files into units and assign to splits
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
assert files, f"No CSV files found in {DATA_DIR}"
rng = np.random.default_rng(SEED)
files = [files[i] for i in rng.permutation(len(files))]
if MAX_FILES:
    files = files[:MAX_FILES]

by_class = {}
for p in files:
    by_class.setdefault(os.path.basename(p).split("__")[0], []).append(p)

SPLITS = ("train", "val", "test")
cuts = np.cumsum((0.0,) + FILE_SPLIT)  # row-block boundaries for small classes
split_units = {s: [] for s in SPLITS}  # (path, (lo, hi))
for cls, paths in by_class.items():
    if len(paths) >= 3:  # file-level split
        n = len(paths)
        n_tr, n_va = max(1, int(n * FILE_SPLIT[0])), max(1, int(n * FILE_SPLIT[1]))
        parts = {"train": paths[:n_tr], "val": paths[n_tr : n_tr + n_va], "test": paths[n_tr + n_va :]}
        for s in SPLITS:
            split_units[s] += [(p, (0.0, 1.0)) for p in parts[s]]
    else:  # contiguous row blocks of the file(s)
        for p in paths:
            for i, s in enumerate(SPLITS):
                split_units[s].append((p, (cuts[i], cuts[i + 1])))
print({k: len(v) for k, v in split_units.items()}, "units per split")

data = {}
for name, units in split_units.items():
    parts, t0 = [], time.time()
    for i, (p, rows) in enumerate(units):
        mode = "train" if name == "train" else "eval"
        parts.append(load_unit(p, rows, mode, seed=SEED + i))
        if (i + 1) % 20 == 0:
            print(f"  {name}: {i + 1}/{len(units)} units, {time.time() - t0:.0f}s")
    data[name] = pd.concat(parts, ignore_index=True)
    print(f"{name}: {len(data[name]):,} rows loaded in {time.time() - t0:.0f}s")

FEATS = [c for c in data["train"].columns if c not in ("y", "subtype")]


# %% Deduplicate and remove cross-split leakage
def row_hash(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[FEATS], index=False).values


report = {}
hashes = {k: row_hash(v) for k, v in data.items()}

# identical feature vectors that carry DIFFERENT labels = label noise in the dataset itself
tmp = pd.DataFrame({"h": hashes["train"], "y": data["train"]["y"].values})
conflicts = int((tmp.groupby("h")["y"].nunique() > 1).sum())
report["train_feature_rows_with_conflicting_labels"] = conflicts

for name in data:
    df = data[name].assign(_h=hashes[name])
    before = len(df)
    df = df.drop_duplicates("_h")
    report[f"{name}_removed_duplicates"] = before - len(df)
    data[name] = df

train_hashes = set(data["train"]["_h"].values)
for name in ("val", "test"):
    df = data[name]
    before = len(df)
    df = df[~df["_h"].isin(train_hashes)]
    report[f"{name}_removed_overlap_with_train"] = before - len(df)
    data[name] = df

val_hashes = set(data["val"]["_h"].values)  # also keep test disjoint from val
before = len(data["test"])
data["test"] = data["test"][~data["test"]["_h"].isin(val_hashes)]
report["test_removed_overlap_with_val"] = before - len(data["test"])

# %% Save + summary
os.makedirs(OUT_DIR, exist_ok=True)
counts = {}
for name, df in data.items():
    df = df.drop(columns="_h").reset_index(drop=True)
    df["subtype"] = df["subtype"].astype("category")
    df.to_parquet(os.path.join(OUT_DIR, f"{name}.parquet"), index=False)
    counts[name] = {CLASSES[k]: int(v) for k, v in df["y"].value_counts().sort_index().items()}

meta = {
    "seed": SEED,
    "classes": CLASSES,
    "features": FEATS,
    "n_units": {k: len(v) for k, v in split_units.items()},
    "train_cap_per_class_per_file": TRAIN_CAP_PER_CLASS_PER_FILE,
    "eval_frac": EVAL_FRAC,
    "cleaning_report": report,
    "class_counts": counts,
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nCleaning report:", json.dumps(report, indent=2))
print("\nClass counts per split:")
print(pd.DataFrame(counts).fillna(0).astype(int))
