# src/features/data_diet.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _infer_text_column(df: pd.DataFrame) -> str:
    """
    Определяем, где лежит текст: в 'text' или в 'sentence'.
    Бросаем внятную ошибку, если ничего не нашли.
    """
    if "text" in df.columns:
        return "text"
    if "sentence" in df.columns:
        return "sentence"
    raise ValueError(
        f"ensure_el2n_scores: expected df to have 'text' or 'sentence' column, got: {list(df.columns)}"
    )


class _EL2NDataset(Dataset):
    """
    Простой Dataset, который хранит (idx, sentence, label) и отдаёт их по одному.
    """

    def __init__(self, df_with_idx: pd.DataFrame):
        # ожидаем колонки: idx, sentence, label
        self._sentences = df_with_idx["sentence"].astype(str).tolist()
        self._labels = df_with_idx["label"].astype(int).tolist()
        self._idx = df_with_idx["idx"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {
            "idx": self._idx[i],
            "sentence": self._sentences[i],
            "label": self._labels[i],
        }


def ensure_el2n_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Упрощённая реализация Data Diet / EL2N:

    - Берём модель cfg.model.name (ModernBERT),
    - Делаем ОДИН проход по датасету без обучения,
    - Для каждого примера считаем:
        * p = softmax(logits)
        * y_onehot = one_hot(label)
        * el2n = ||p - y_onehot||_2
        * ce_loss = cross-entropy(logits, label)
    - Возвращаем DataFrame с колонками: idx, el2n, ce_loss

    Важно:
    - idx == исходный индекс df (после reset_index), чтобы селектор мог
      выровнять по df.index.
    """

    if "label" not in df.columns:
        raise ValueError("ensure_el2n_scores: expected 'label' column in df")

    text_col = _infer_text_column(df)

    # добавляем явный idx и переименовываем текст → 'sentence'
    df_with_idx = (
        df.reset_index(drop=False)
        .rename(columns={"index": "idx", text_col: "sentence"})
        [["idx", "sentence", "label"]]
        .copy()
    )

    # ---- конфиги / гиперпараметры из cfg ----
    model_name = getattr(getattr(cfg, "model", None), "name", None) or "answerdotai/ModernBERT-base"
    max_length = getattr(getattr(cfg, "data", None), "max_length", 128)
    batch_size = getattr(getattr(cfg, "train", None), "batch_size", 32)
    seed = int(getattr(cfg, "seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- токенизатор и модель ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ---- Dataset / DataLoader ----
    dataset = _EL2NDataset(df_with_idx)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        sentences = [b["sentence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

        enc = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        enc["idx"] = idx
        return enc

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---- проходим по датасету и считаем EL2N ----
    all_idx: list[int] = []
    all_el2n: list[float] = []
    all_ce: list[float] = []

    with torch.no_grad():
        for batch in loader:
            idx = batch["idx"].cpu().numpy()
            labels = batch["labels"].to(device)

            # убираем служебные поля из batch перед подачей в модель
            model_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("labels", "idx")
            }

            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            # one-hot метки
            num_classes = probs.shape[-1]
            one_hot = F.one_hot(labels, num_classes=num_classes).float()

            # EL2N = ||p - y||_2
            diff = probs - one_hot
            el2n_batch = torch.sqrt(torch.sum(diff * diff, dim=-1))

            # CE loss (per-example)
            ce_batch = F.cross_entropy(logits, labels, reduction="none")

            all_idx.extend(idx.tolist())
            all_el2n.extend(el2n_batch.cpu().numpy().tolist())
            all_ce.extend(ce_batch.cpu().numpy().tolist())

    # ---- собираем DataFrame ----
    scores_df = pd.DataFrame(
        {
            "idx": all_idx,
            "el2n": all_el2n,
            "ce_loss": all_ce,
        }
    )

    # на всякий случай сортируем по idx
    scores_df = scores_df.sort_values("idx").reset_index(drop=True)

    return scores_df
