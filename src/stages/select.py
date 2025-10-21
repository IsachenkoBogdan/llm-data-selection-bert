import os
import pandas as pd
import wandb
from ..selectors import REGISTRY
from ..utils import time_block


def run(cfg, run_wandb=None):
    data_path = os.path.join(cfg.paths.data_dir, "processed", "sst2.parquet")
    df = pd.read_parquet(data_path)
    
    k = max(1, int(len(df) * cfg.subset.frac))
    
    selector_cls = REGISTRY[cfg.selection.name]
    selector = selector_cls(**cfg.selection.params)
    
    with time_block() as elapsed:
        selector.fit(df, cfg=cfg)
        subset = selector.select(df, k)
    selection_time = elapsed()
    
    subset = subset.reset_index(drop=False).rename(columns={"index": "id"})
    
    out_path = os.path.join(
        "data",
        "manifests",
        f"{cfg.selection.name}_p{int(round(cfg.subset.frac * 100)):02d}_seed{cfg.seed}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subset.to_csv(out_path, index=False)
    
    if wandb.run is not None:
        metrics = {
            "selection/time_sec": selection_time,
            "selection/size": len(subset),
        }
        
        if run_wandb:
            metrics["subset_stats"] = {
                "size": len(subset),
                "class_balance": subset["label"].value_counts(normalize=True).to_dict(),
                "avg_len": float(subset["text"].str.split().apply(len).mean()),
            }
        
        wandb.log(metrics)
    
    return out_path
