from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseSelector
from ..features.predictive_stats import (
    ensure_predictive_entropy,
    ensure_wordpiece_ratio,
)


class PredictiveEntropySel(BaseSelector):
    """
    Селектор по predictive entropy внешнего SST-2 классификатора.

    Идея: берем примеры, где модель максимально не уверена.
    """

    def __init__(
        self,
        entropy_model_name: str = "textattack/bert-base-uncased-SST-2",
        entropy_batch_size: int = 64,
        entropy_max_length: int = 128,
        per_class_balance: bool = True,
    ):
        self.entropy_model_name = entropy_model_name
        self.entropy_batch_size = int(entropy_batch_size)
        self.entropy_max_length = int(entropy_max_length)
        self.per_class_balance = bool(per_class_balance)

        self._scores: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, cfg=None):
        if "text" not in df.columns:
            raise ValueError("PredictiveEntropySel expects column 'text' in df")

        texts = df["text"].tolist()

        entropy = ensure_predictive_entropy(
            texts,
            model_name=self.entropy_model_name,
            batch_size=self.entropy_batch_size,
            max_length=self.entropy_max_length,
        )

        self._scores = entropy.astype(np.float32)
        self._labels = df["label"].to_numpy() if "label" in df.columns else None
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("Selector must be fit(df) before select(df, k).")

        n = len(df)
        k = min(k, n)
        scores = self._scores
        idx_all = np.arange(n)

        # high entropy = more uncertain = более "интересный"
        if self.per_class_balance and (self._labels is not None):
            labels = self._labels
            uniq = np.unique(labels)
            k_per = k // len(uniq)
            selected_idx = []

            for y in uniq:
                mask = labels == y
                idx_y = idx_all[mask]
                scores_y = scores[mask]
                order_y = np.argsort(-scores_y)  # по убыванию энтропии
                take = min(k_per, len(idx_y))
                selected_idx.extend(idx_y[order_y[:take]].tolist())

            if len(selected_idx) < k:
                selected_idx_arr = np.asarray(selected_idx, dtype=int)
                rest = np.setdiff1d(idx_all, selected_idx_arr, assume_unique=True)
                scores_rest = scores[rest]
                order_rest = np.argsort(-scores_rest)
                need = k - len(selected_idx)
                selected_idx.extend(rest[order_rest[:need]].tolist())

            selected_idx = np.asarray(selected_idx, dtype=int)
        else:
            order = np.argsort(-scores)
            selected_idx = order[:k]

        return df.iloc[selected_idx]


class WordPieceRatioSel(BaseSelector):
    """
    Селектор по WordPiece ratio для ModernBERT (или другого токенизатора).

    Идея: берем примеры с наиболее "ломаной" лексикой (много сабтокенов на слово)
    или наоборот — с более простой лексикой, в зависимости от order.
    """

    def __init__(
        self,
        wp_tokenizer_name: str = "answerdotai/modernbert-base",
        wp_max_length: int = 128,
        order: str = "high",          # "high" → самые большие ratio, "low" → самые маленькие
        per_class_balance: bool = True,
    ):
        self.wp_tokenizer_name = wp_tokenizer_name
        self.wp_max_length = int(wp_max_length)
        self.order = order
        self.per_class_balance = bool(per_class_balance)

        self._scores: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, cfg=None):
        if "text" not in df.columns:
            raise ValueError("WordPieceRatioSel expects column 'text' in df")

        texts = df["text"].tolist()

        wp_ratio = ensure_wordpiece_ratio(
            texts,
            tokenizer_name=self.wp_tokenizer_name,
            max_length=self.wp_max_length,
        )

        self._scores = wp_ratio.astype(np.float32)
        self._labels = df["label"].to_numpy() if "label" in df.columns else None
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("Selector must be fit(df) before select(df, k).")

        n = len(df)
        k = min(k, n)
        scores = self._scores
        idx_all = np.arange(n)

        # выбираем направление сортировки
        if self.order == "high":
            sort_scores = -scores  # большие ratio → интереснее
        elif self.order == "low":
            sort_scores = scores   # маленькие ratio → интереснее
        else:
            raise ValueError(f"Unknown order='{self.order}', use 'high' or 'low'.")

        if self.per_class_balance and (self._labels is not None):
            labels = self._labels
            uniq = np.unique(labels)
            k_per = k // len(uniq)
            selected_idx = []

            for y in uniq:
                mask = labels == y
                idx_y = idx_all[mask]
                sort_y = sort_scores[mask]
                order_y = np.argsort(sort_y)  # учитываем выбранное направление
                take = min(k_per, len(idx_y))
                selected_idx.extend(idx_y[order_y[:take]].tolist())

            if len(selected_idx) < k:
                selected_idx_arr = np.asarray(selected_idx, dtype=int)
                rest = np.setdiff1d(idx_all, selected_idx_arr, assume_unique=True)
                sort_rest = sort_scores[rest]
                order_rest = np.argsort(sort_rest)
                need = k - len(selected_idx)
                selected_idx.extend(rest[order_rest[:need]].tolist())

            selected_idx = np.asarray(selected_idx, dtype=int)
        else:
            order = np.argsort(sort_scores)
            selected_idx = order[:k]

        return df.iloc[selected_idx]
