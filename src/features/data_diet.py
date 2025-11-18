from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _get_dd_cfg(cfg) -> Dict[str, Any]:
    """Аккуратно вытащить параметры из cfg с дефолтами."""
    dd = getattr(cfg, "datadiet", None)
    train = getattr(cfg, "train", None)

    def g(section, name, default):
        if section is None:
            return default
        try:
            return getattr(section, name)
        except Exception:
            return section.get(name, default) if isinstance(section, dict) else default

    return {
        "epochs": int(g(dd, "epochs", 1)),
        "batch_size": int(g(dd, "batch_size", 32)),
        "lr": float(g(dd, "lr", 2e-5 if train is None else g(train, "lr", 2e-5))),
    }


def ensure_el2n_scores(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Считает (или грузит из кэша) EL2N score для каждого примера train SST-2.

    EL2N(x) = || p_theta(y | x) - one_hot(y) ||_2, усреднённый по нескольким проходам обучения.

    Возвращает DataFrame с колонками:
      - idx  : индекс примера в df
      - el2n : float score (чем выше, тем "важнее"/труднее пример)
    """
    if cfg is not None and hasattr(cfg, "paths"):
        artifacts_dir = Path(cfg.paths.artifacts_dir)
    else:
        artifacts_dir = Path("artifacts")

    feat_dir = artifacts_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    scores_path = feat_dir / "sst2_el2n_scores.parquet"

    # если уже посчитано — просто читаем и возвращаем
    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        # sanity-check: если размер совпадает, просто возвращаем
        if len(scores_df) == len(df):
            return scores_df

    # иначе считаем с нуля
    dd_cfg = _get_dd_cfg(cfg) if cfg is not None else {"epochs": 1, "batch_size": 32, "lr": 2e-5}
    epochs = dd_cfg["epochs"]
    batch_size = dd_cfg["batch_size"]
    lr = dd_cfg["lr"]

    model_name = getattr(cfg.model, "name", "answerdotai/ModernBERT-base") if cfg is not None else "answerdotai/ModernBERT-base"
    max_length = getattr(cfg.data, "max_length", 128) if cfg is not None and hasattr(cfg, "data") else 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # HuggingFace Dataset с индексом
    df_with_idx = df.reset_index(drop=False).rename(columns={"index": "idx"})
    hf_ds = Dataset.from_pandas(df_with_idx[["idx", "sentence", "label"]], preserve_index=False)

    def encode(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        enc["idx"] = batch["idx"]
        return enc

    hf_ds = hf_ds.map(encode, batched=True)

    # PyTorch DataLoader
    columns = ["input_ids", "attention_mask", "labels", "idx"]

    def collate_fn(batch_list):
        collated = {col: [] for col in columns}
        for b in batch_list:
            for col in columns:
                collated[col].append(b[col])
        for col in ["input_ids", "attention_mask"]:
            collated[col] = torch.tensor(collated[col], dtype=torch.long)
        collated["labels"] = torch.tensor(collated["labels"], dtype=torch.long)
        collated["idx"] = torch.tensor(collated["idx"], dtype=torch.long)
        return collated

    loader = torch.utils.data.DataLoader(
        hf_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n = len(df)
    # суммарные EL2N и количество появлений каждого примера
    el2n_sum = torch.zeros(n, dtype=torch.float32)
    counts = torch.zeros(n, dtype=torch.int32)

    for epoch in range(epochs):
        for batch in loader:
            idx = batch["idx"].to(device)
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)  # (B, num_labels)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(-1)).float()
                diff = probs - one_hot
                el2n_batch = torch.linalg.vector_norm(diff, ord=2, dim=-1)  # (B,)

            idx_cpu = idx.detach().cpu()
            el2n_cpu = el2n_batch.detach().cpu()

            el2n_sum[idx_cpu] += el2n_cpu
            counts[idx_cpu] += 1

    # усредняем
    counts = counts.clamp(min=1)
    el2n_mean = el2n_sum / counts.to(dtype=torch.float32)

    scores_df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int64),
            "el2n": el2n_mean.numpy().astype("float32"),
        }
    )
    scores_df.to_parquet(scores_path, index=False)

    return scores_df
