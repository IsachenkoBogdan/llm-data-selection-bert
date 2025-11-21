import time
from dataclasses import dataclass
from typing import Any

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

EL2N_COL = "el2n"
IDX_COL = "idx"


@dataclass
class DataDietConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 2
    max_steps: int = 1000  # ограничение на число шагов (ранняя фаза)


def _get_dd_cfg(cfg: Any | None) -> DataDietConfig:
    if cfg is None:
        return DataDietConfig()

    model_name = getattr(getattr(cfg, "model", None), "name", DataDietConfig.model_name)
    max_length = getattr(getattr(cfg, "data", None), "max_length", DataDietConfig.max_length)

    dd = getattr(getattr(cfg, "features", None), "data_diet", None)

    return DataDietConfig(
        model_name=getattr(dd, "model_name", model_name) if dd is not None else model_name,
        max_length=getattr(dd, "max_length", max_length) if dd is not None else max_length,
        batch_size=getattr(dd, "batch_size", DataDietConfig.batch_size) if dd is not None else DataDietConfig.batch_size,
        lr=getattr(dd, "lr", DataDietConfig.lr) if dd is not None else DataDietConfig.lr,
        epochs=getattr(dd, "epochs", DataDietConfig.epochs) if dd is not None else DataDietConfig.epochs,
        max_steps=getattr(dd, "max_steps", DataDietConfig.max_steps) if dd is not None else DataDietConfig.max_steps,
    )


def ensure_el2n_scores(df: pd.DataFrame, cfg: Any | None = None) -> pd.DataFrame:
    """Рассчитать EL2N-score для каждого примера в df (train-часть)."""

    dd_cfg = _get_dd_cfg(cfg)
    seed = int(getattr(cfg, "seed", 42)) if cfg is not None else 42

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df_local = df.reset_index(drop=True).copy()
    n = len(df_local)
    df_local[IDX_COL] = np.arange(n, dtype=int)

    hf_ds = Dataset.from_pandas(
        df_local[[IDX_COL, "sentence", "label"]],
        preserve_index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(dd_cfg.model_name, use_fast=True)

    def _encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=dd_cfg.max_length,
        )
        enc["labels"] = batch["label"]
        enc[IDX_COL] = batch[IDX_COL]
        return enc

    hf_ds = hf_ds.map(_encode, batched=True, remove_columns=["sentence", "label"])
    hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", IDX_COL])

    dataloader = DataLoader(
        hf_ds,
        batch_size=dd_cfg.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        dd_cfg.model_name,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=dd_cfg.lr)

    total_steps = dd_cfg.epochs * len(dataloader)
    max_steps = min(dd_cfg.max_steps, total_steps)
    num_warmup = max(1, max_steps // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=max_steps,
    )

    scores = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)

    step = 0
    t0 = time.time()

    for epoch in range(dd_cfg.epochs):
        model.train()
        for batch in dataloader:
            step += 1
            if step > max_steps:
                break

            idx_batch = batch[IDX_COL].cpu().numpy()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            with torch.no_grad():
                probs = torch.softmax(outputs.logits, dim=-1)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(-1)).float()
                el2n = torch.sum((probs - one_hot) ** 2, dim=-1)  # [B]
                el2n_np = el2n.cpu().numpy()

            scores[idx_batch] += el2n_np
            counts[idx_batch] += 1

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step > max_steps:
            break

    elapsed = time.time() - t0
    print(f"[DataDiet] computed EL2N for {n} examples in {elapsed:.1f}s "
          f"({step} steps, epochs_used≈{step / max(len(dataloader), 1):.2f})")

    counts[counts == 0] = 1
    scores = scores / counts

    return pd.DataFrame(
        {
            IDX_COL: df_local[IDX_COL].to_numpy(),
            EL2N_COL: scores.astype(np.float32),
        }
    )
