# src/features/data_diet.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..utils import set_seed

IDX_COL = "idx"


def _get_text_col(df: pd.DataFrame) -> str:
    if "text" in df.columns:
        return "text"
    if "sentence" in df.columns:
        return "sentence"
    raise KeyError("Neither 'text' nor 'sentence' column found in dataframe.")


def _get_cache_path(cfg) -> Path:
    base_dir = getattr(getattr(cfg, "features", None), "cache_dir", "features")
    return Path(base_dir) / "el2n_scores.parquet"


def ensure_el2n_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    cache_path = _get_cache_path(cfg)
    force = bool(getattr(getattr(cfg.features, "data_diet", None), "force_recompute", False))

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(int(getattr(cfg, "seed", 42)))

    text_col = _get_text_col(df)
    df_local = df.reset_index(drop=False).rename(columns={"index": IDX_COL})

    if "label" not in df_local.columns:
        raise KeyError("DataFrame must contain 'label' column for DataDiet.")

    hf_ds = Dataset.from_pandas(
        df_local[[IDX_COL, text_col, "label"]],
        preserve_index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)

    def encode(batch):
        enc = tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=cfg.data.max_length,
        )
        enc["labels"] = batch["label"]
        enc[IDX_COL] = batch[IDX_COL]
        return enc

    hf_ds = hf_ds.map(encode, batched=True)
    hf_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", IDX_COL],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dd_cfg = getattr(cfg.features, "data_diet", None)
    epochs = int(getattr(dd_cfg, "epochs", 1))
    batch_size = int(getattr(dd_cfg, "batch_size", 32))
    lr = float(getattr(dd_cfg, "lr", 2e-5))
    max_steps = int(getattr(dd_cfg, "max_steps", 0))  # 0 = без лимита

    loader = DataLoader(hf_ds, batch_size=batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(loader)
    if max_steps and max_steps < total_steps:
        total_steps = max_steps
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    n = len(df_local)
    el2n_sum = np.zeros(n, dtype=np.float32)
    el2n_cnt = np.zeros(n, dtype=np.int32)

    step = 0
    num_labels = 2

    for _ in range(epochs):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            logits = out.logits.detach()
            probs = torch.softmax(logits, dim=-1)
            labels_oh = torch.nn.functional.one_hot(
                batch["labels"],
                num_classes=num_labels,
            ).float()

            el2n_batch = torch.linalg.norm(labels_oh - probs, dim=-1)

            idx_np = batch[IDX_COL].cpu().numpy()
            val_np = el2n_batch.cpu().numpy()
            el2n_sum[idx_np] += val_np
            el2n_cnt[idx_np] += 1

            loss = out.loss
            loss.backward()
            optim.step()
            scheduler.step()

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    mask = el2n_cnt > 0
    el2n = np.zeros(n, dtype=np.float32)
    el2n[mask] = el2n_sum[mask] / el2n_cnt[mask]

    scores_df = pd.DataFrame(
        {
            IDX_COL: np.arange(n, dtype=np.int64),
            "el2n": el2n,
        }
    )

    scores_df.to_parquet(cache_path, index=False)
    return scores_df
