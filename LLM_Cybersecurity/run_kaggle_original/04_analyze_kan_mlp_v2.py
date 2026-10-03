"""
04_analyze_kan_mlp.py
Turns results_kan_mlp.jsonl into the tables and figure for the write-up.

Outputs (in OUT_DIR):
  table_full_data.md      RQ1: mean +- std over seeds at full training data (+ cost columns)
  table_paired_diff.md    RQ2: KAN minus MLP macro-F1 per training size, mean and 95% CI over seeds
  learning_curve.png      RQ2: macro-F1 vs number of training samples
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.environ.get("OUT_DIR", "./out")
TAG = os.environ.get("TAG", "")  # same TAG as used in 03_kan_vs_mlp.py
rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, f"results_kan_mlp{TAG}.jsonl"))]
df = pd.DataFrame(rows)
for c in ("BruteForce", "Web-based"):
    df[f"f1_{c}"] = df["per_class_f1"].apply(lambda d, c=c: d.get(c, np.nan))
full_n = df["n_train"].max()


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


# --- RQ1: full-data table -------------------------------------------------------------
full = df[df["n_train"] == full_n]
tab = (
    full.groupby("model")
    .agg(
        seeds=("seed", "nunique"),
        params=("params", "first"),
        macro_f1=("macro_f1", ms),
        f1_bruteforce=("f1_BruteForce", ms),
        f1_webbased=("f1_Web-based", ms),
        train_s=("train_s", lambda x: f"{x.mean():.0f}"),
        lat_ms_p50=("lat_ms_p50", lambda x: f"{x.mean():.2f}"),
    )
    .reset_index()
)
md = tab.to_markdown(index=False)
open(os.path.join(OUT_DIR, f"table_full_data{TAG}.md"), "w").write(
    md + "\n\n(XGB 'params' = total tree nodes, not comparable to neural parameters.)\n"
)
print("RQ1 - full training data (n =", full_n, ")\n", md, "\n")

# --- RQ2: paired KAN-MLP difference per size ---------------------------------------
big = max(int(m.split("-h")[1]) for m in df["model"] if m.startswith("KAN"))
kan, mlp = f"KAN-h{big}", f"MLP-h{big}"
lines = ["| n_train | mean ΔmacroF1 (KAN−MLP) | 95% CI | seeds | KAN better in |", "|---|---|---|---|---|"]
for n, g in df[df["model"].isin([kan, mlp])].groupby("n_train"):
    p = g.pivot(index="seed", columns="model", values="macro_f1").dropna()
    if len(p) < 2:
        continue
    d = (p[kan] - p[mlp]).values
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    lines.append(f"| {n} | {d.mean():+.3f} | [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}] | {len(d)} | {(d > 0).sum()}/{len(d)} |")
md2 = "\n".join(lines)
open(os.path.join(OUT_DIR, f"table_paired_diff{TAG}.md"), "w").write(md2 + "\n")
print("RQ2 - paired differences (", kan, "vs", mlp, ")\n", md2, "\n")

# --- Learning curve ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for model in ("XGB", kan, mlp):
    g = df[df["model"] == model].groupby("n_train")["macro_f1"].agg(["mean", "std"]).reset_index()
    ax.errorbar(g["n_train"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=model)
ax.set_xscale("log")
ax.set_xlabel("training samples (log scale)")
ax.set_ylabel("test macro-F1")
ax.set_title("Data efficiency on CICIoT2023 (mean ± std over seeds)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"learning_curve{TAG}.png"), dpi=150)
print("saved tables and learning_curve.png to", OUT_DIR)
