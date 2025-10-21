import os, pandas as pd
from datasets import load_dataset

def run(cfg):
    ds = load_dataset("glue", "sst2")
    train = ds["train"]
    df = pd.DataFrame({"text": train["sentence"], "label": train["label"]})
    out = os.path.join(cfg.paths.data_dir, "processed", "sst2.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    return out
