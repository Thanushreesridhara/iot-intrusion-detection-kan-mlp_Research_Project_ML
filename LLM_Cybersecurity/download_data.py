"""Download CICIoT2023 per-class CSV parts from the HF mirror bencorn/CIC-IoT-2023 and
add the `label` column (taken from the folder name) so 01_prepare_data.py can consume them."""
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

REPO = "bencorn/CIC-IoT-2023"
RAW = "raw"
OUT = "data/CICIoT2023"
os.makedirs(OUT, exist_ok=True)
files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("CSV/CSV/") and f.endswith(".csv")]
print(len(files), "files")

def fetch(f):
    folder, name = f.split("/")[2], os.path.basename(f)
    dst = os.path.join(OUT, f"{folder}__{name}")
    if os.path.exists(dst):
        return
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW)
    df = pd.read_csv(p)
    df["label"] = folder
    df.to_csv(dst + ".tmp", index=False)
    os.replace(dst + ".tmp", dst)
    os.remove(p)
    print("done", dst, flush=True)

with ThreadPoolExecutor(6) as ex:
    list(ex.map(fetch, files))
