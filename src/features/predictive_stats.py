# src/features/predictive_stats.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Модель для predictive entropy (готовая SST-2)
DEFAULT_ENTROPY_MODEL = "textattack/bert-base-uncased-SST-2"
# Модель для WordPiece ratio — используем ModernBERT из конфига, но есть дефолт
DEFAULT_WP_MODEL = "answerdotai/ModernBERT-base"


# --- Вспомогательная обвязка для модели энтропии --- #

_entropy_tokenizer = None
_entropy_model = None
_entropy_device = None


def _get_entropy_model(
    model_name: str = DEFAULT_ENTROPY_MODEL,
    device: Optional[str] = None,
):
    """
    Лениво загружаем маленький SST-2-классификатор для оценки энтропии.
    Никаких градиентов, только inference.
    """
    global _entropy_tokenizer, _entropy_model, _entropy_device

    if _entropy_tokenizer is not None and _entropy_model is not None:
        return _entropy_tokenizer, _entropy_model, _entropy_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    _entropy_tokenizer = tokenizer
    _entropy_model = model
    _entropy_device = device

    return tokenizer, model, device


def compute_predictive_entropy(
    texts: pd.Series,
    model_name: str = DEFAULT_ENTROPY_MODEL,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Считает predictive entropy для каждого текста:
      H(p) = -sum_j p_j log p_j, где p = softmax(logits).

    Возвращает np.ndarray shape (N,).
    """
    tokenizer, model, device = _get_entropy_model(model_name=model_name, device=device)

    entropies = []

    # гарантируем последовательность строк
    series = texts.astype(str).reset_index(drop=True)

    for start in range(0, len(series), batch_size):
        batch_texts = list(series.iloc[start : start + batch_size])

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=-1)

        entropies.append(entropy.cpu().numpy())

    if not entropies:
        return np.zeros(shape=(0,), dtype=np.float32)

    return np.concatenate(entropies, axis=0).astype(np.float32)


# --- WordPiece ratio --- #

def compute_wordpiece_ratio(
    texts: pd.Series,
    model_name: str = DEFAULT_WP_MODEL,
    max_length: int = 128,
) -> np.ndarray:
    """
    WordPiece ratio = (# субтокенов) / (# слов) для каждого текста.

    Высокое значение → больше редких / сложных слов и субтокенов.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    ratios = []
    series = texts.astype(str).reset_index(drop=True)

    for t in series:
        words = t.split()
        n_words = len(words)
        if n_words == 0:
            ratios.append(0.0)
            continue

        encoded = tokenizer(
            t,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=True,
        )
        n_subtokens = len(encoded["input_ids"])

        ratios.append(float(n_subtokens) / float(n_words))

    return np.asarray(ratios, dtype=np.float32)
