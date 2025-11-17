# src/selectors/llm_quality.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseSelector
from ..features.llm_clarity import ensure_llm_clarity_scores


class LLMQualitySel(BaseSelector):
    """
    Селектор, который использует LLM-based clarity-классификатор.

    Логика:
      - есть 3 класса однозначности (0,1,2) от студента (по LLM-разметке),
      - хотим набрать k примеров:
          * примерно k/3 из каждого класса,
          * внутри каждого класса — самые уверенные (по P(class=c)).

    Это как QuRating-style quality score, но с явной стратификацией по классам.
    """

    def fit(self, df: pd.DataFrame, cfg=None):
        # получаем (или считаем) clarity score для ВСЕГО датасета
        scores_df = ensure_llm_clarity_scores(df, cfg)

        # выравниваем по индексу df
        scores_df = scores_df.set_index("idx")
        aligned = scores_df.loc[df.index]

        self._p0 = aligned["llm_clarity_p0"].to_numpy()
        self._p1 = aligned["llm_clarity_p1"].to_numpy()
        self._p2 = aligned["llm_clarity_p2"].to_numpy()
        self._pred = aligned["llm_clarity_pred"].to_numpy(dtype=int)

        self._seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if not hasattr(self, "_pred"):
            raise RuntimeError("LLMQualitySel.fit must be called before select()")

        n = len(df)
        k = min(k, n)
        if k <= 0:
            return df.iloc[0:0].copy()

        preds = self._pred
        proba = np.stack([self._p0, self._p1, self._p2], axis=1)  # shape (N, 3)

        # целевые количества на класс: равномерно по трём
        base = k // 3
        rem = k % 3
        target_per_class = [base, base, base]
        for i in range(rem):
            target_per_class[i] += 1

        chosen_indices = []
        rng = np.random.default_rng(self._seed)

        for c in range(3):
            idx_c = np.where(preds == c)[0]
            if len(idx_c) == 0:
                continue

            # сортировка по уверенности P(class=c) по убыванию
            scores_c = proba[idx_c, c]
            order = np.argsort(-scores_c)
            idx_sorted_c = idx_c[order]

            take_c = min(target_per_class[c], len(idx_sorted_c))
            chosen_indices.extend(idx_sorted_c[:take_c].tolist())

        # если не набрали k (например, мало примеров класса 2) — добираем лучшими по max proba
        if len(chosen_indices) < k:
            chosen_set = set(chosen_indices)
            remaining_idx = np.array([i for i in range(n) if i not in chosen_set], dtype=int)
            if len(remaining_idx) > 0:
                max_proba = proba[remaining_idx].max(axis=1)
                order_rem = np.argsort(-max_proba)
                extra = remaining_idx[order_rem[: k - len(chosen_indices)]]
                chosen_indices.extend(extra.tolist())

        chosen_indices = np.array(chosen_indices[:k], dtype=int)
        return df.iloc[chosen_indices].copy()
