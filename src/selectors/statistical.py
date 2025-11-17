# src/selectors/statistical.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseSelector
from ..features.predictive_stats import (
    compute_predictive_entropy,
    compute_wordpiece_ratio,
)


# ---------- Predictive entropy selector ----------

@dataclass
class _EntropyCfg:
    model_name: str = "textattack/bert-base-uncased-SST-2"
    batch_size: int = 32
    keep: str = "middle"      # "high" | "low" | "middle"
    middle_low_q: float = 0.2
    middle_high_q: float = 0.8


class PredictiveEntropySel(BaseSelector):
    """
    Селектор по predictive entropy:

      - keep="high"   → берём самые неопределённые примеры
      - keep="low"    → берём самые уверенные
      - keep="middle" → отбрасываем хвосты, берём среднюю сложность

    Энтропия считается поверх маленького SST-2 классификатора (по умолчанию
    textattack/bert-base-uncased-SST-2), отдельно от ModernBERT.
    """

    def __init__(
        self,
        model_name: str = "textattack/bert-base-uncased-SST-2",
        batch_size: int = 32,
        keep: str = "middle",
        middle_low_q: float = 0.2,
        middle_high_q: float = 0.8,
    ):
        self.cfg = _EntropyCfg(
            model_name=model_name,
            batch_size=batch_size,
            keep=keep,
            middle_low_q=middle_low_q,
            middle_high_q=middle_high_q,
        )
        self._entropy: Optional[np.ndarray] = None
        self._seed: int = 0

    def fit(self, df: pd.DataFrame, cfg=None):
        # df должен содержать колонку "text"
        self._entropy = compute_predictive_entropy(
            df["text"],
            model_name=self.cfg.model_name,
            batch_size=self.cfg.batch_size,
        )
        self._seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if self._entropy is None:
            raise RuntimeError("PredictiveEntropySel.fit must be called before select()")

        ent = self._entropy
        n = len(df)
        k = min(k, n)

        if k <= 0:
            return df.iloc[0:0].copy()

        if self.cfg.keep == "high":
            # самые неопределённые
            idx = np.argsort(-ent)[:k]

        elif self.cfg.keep == "low":
            # самые уверенные
            idx = np.argsort(ent)[:k]

        else:  # "middle"
            lo = np.quantile(ent, self.cfg.middle_low_q)
            hi = np.quantile(ent, self.cfg.middle_high_q)
            middle_mask = (ent >= lo) & (ent <= hi)
            middle_idx = np.where(middle_mask)[0]

            if len(middle_idx) <= k:
                idx = middle_idx
            else:
                rng = np.random.default_rng(self._seed)
                idx = rng.choice(middle_idx, size=k, replace=False)

        return df.iloc[idx].copy()


# ---------- WordPiece ratio selector ----------

class WordPieceRatioSel(BaseSelector):
    """
    Селектор по WordPiece ratio:

      WordPiece ratio = (# субтокенов) / (# слов) для текста.

      - keep="high"   → тексты с большим количеством субтокенов / более "сложные"
      - keep="low"    → более простые / короткие
      - keep="middle" → отбрасываем экстремально короткие/длинные, берём середину
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 128,
        keep: str = "middle",
        middle_low_q: float = 0.2,
        middle_high_q: float = 0.8,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.keep = keep
        self.middle_low_q = middle_low_q
        self.middle_high_q = middle_high_q

        self._ratio: Optional[np.ndarray] = None
        self._seed: int = 0

    def fit(self, df: pd.DataFrame, cfg=None):
        self._ratio = compute_wordpiece_ratio(
            df["text"],
            model_name=self.model_name,
            max_length=self.max_length,
        )
        self._seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if self._ratio is None:
            raise RuntimeError("WordPieceRatioSel.fit must be called before select()")

        ratio = self._ratio
        n = len(df)
        k = min(k, n)

        if k <= 0:
            return df.iloc[0:0].copy()

        if self.keep == "high":
            idx = np.argsort(-ratio)[:k]

        elif self.keep == "low":
            idx = np.argsort(ratio)[:k]

        else:  # "middle"
            lo = np.quantile(ratio, self.middle_low_q)
            hi = np.quantile(ratio, self.middle_high_q)
            middle_mask = (ratio >= lo) & (ratio <= hi)
            middle_idx = np.where(middle_mask)[0]

            if len(middle_idx) <= k:
                idx = middle_idx
            else:
                rng = np.random.default_rng(self._seed)
                idx = rng.choice(middle_idx, size=k, replace=False)

        return df.iloc[idx].copy()
