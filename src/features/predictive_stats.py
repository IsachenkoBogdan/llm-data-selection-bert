from typing import Sequence, Literal, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .cache import memory  # joblib.Memory как в других фичах


def _load_entropy_model(
    model_name: str,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@memory.cache
def ensure_predictive_entropy(
    texts: Sequence[str],
    model_name: str = "textattack/bert-base-uncased-SST-2",
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """
    Predictive entropy для каждого текста.

    H(p) = -∑ p_k log p_k
    Чем выше энтропия → тем менее уверена модель.
    """
    tokenizer, model, device = _load_entropy_model(model_name)

    all_ent = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits  # (B, C)
            probs = torch.softmax(logits, dim=-1)  # (B, C)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (B,)
            all_ent.append(ent.cpu().numpy())

    return np.concatenate(all_ent, axis=0)


@memory.cache
def ensure_wordpiece_ratio(
    texts: Sequence[str],
    tokenizer_name: str = "answerdotai/modernbert-base",
    max_length: int = 128,
) -> np.ndarray:
    """
    WordPiece ratio = (# сабтокенов) / (# слов).
    Высокое значение → редкая / морфологически сложная лексика.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ratios = []
    for t in texts:
        # сабтокены без [CLS]/[SEP]
        enc = tokenizer(
            t,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        n_wp = len(enc["input_ids"])
        n_words = len(t.split())
        if n_words == 0:
            ratios.append(0.0)
        else:
            ratios.append(n_wp / float(n_words))

    return np.asarray(ratios, dtype=np.float32)


def combine_entropy_wordpiece(
    entropy: np.ndarray,
    wp_ratio: np.ndarray,
    mode: Literal["product", "sum", "entropy_div_wp"] = "product",
) -> np.ndarray:
    """
    Комбинируем predictive entropy и WordPiece ratio в один скор.

    - "product": нормируем обе фичи в [0,1] и берём произведение
    - "sum": сумма нормированных
    - "entropy_div_wp": энтропия / (wp_ratio), если хочешь штрафовать за "ломаность"
    """
    ent = entropy.astype(np.float32)
    wpr = wp_ratio.astype(np.float32)

    def _norm(x: np.ndarray) -> np.ndarray:
        xmin = float(x.min())
        xmax = float(x.max())
        if xmax <= xmin + 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    ent_n = _norm(ent)
    wpr_n = _norm(wpr)

    if mode == "product":
        score = ent_n * wpr_n
    elif mode == "sum":
        score = ent_n + wpr_n
    elif mode == "entropy_div_wp":
        score = ent_n / (wpr_n + 1e-3)
    else:
        raise ValueError(f"Unknown combine mode: {mode}")

    return score
