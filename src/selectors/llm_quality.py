import numpy as np
import pandas as pd

from .base import BaseSelector
from ..features.llm_clarity import ensure_llm_clarity_scores


class LLMQualitySel(BaseSelector):
    """
    Селектор, который использует LLM-based clarity-классификатор,
    но балансирует выборку по основным меткам SST-2 (label 0/1).

    Логика:
      - для всего датасета считаем p_clear = P(clarity class == 0) от студента;
      - хотим набрать k примеров:
          * примерно одинаковое количество для каждого значения df['label'];
          * внутри каждого sentiment-класса выбираем топ по p_clear.
      - если какого-то sentiment-класса не хватает, добираем лучшие по p_clear
        из оставшихся примеров.
    """

    def fit(self, df: pd.DataFrame, cfg=None):
        # получаем (или считаем) clarity scores для ВСЕГО датасета
        scores_df = ensure_llm_clarity_scores(df, cfg)

        # выравниваем по индексу df
        scores_df = scores_df.set_index("idx")
        aligned = scores_df.loc[df.index]

        # сохраняем только "clear sentiment" вероятность
        self._p_clear = aligned["llm_clarity_p0"].to_numpy(dtype=float)

        # основные метки SST-2 (0/1)
        if "label" not in df.columns:
            raise ValueError("LLMQualitySel expects column 'label' in the dataframe.")
        self._labels = df["label"].to_numpy()

        self._seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if not hasattr(self, "_labels"):
            raise RuntimeError("LLMQualitySel.fit must be called before select().")

        n = len(df)
        k = min(k, n)
        if k <= 0:
            return df.iloc[0:0].copy()

        labels = self._labels
        p_clear = self._p_clear

        unique_labels = np.sort(np.unique(labels))
        n_classes = len(unique_labels)

        # целевые количества на sentiment-класс: равномерно
        base = k // n_classes
        rem = k % n_classes
        target_per_label = {lbl: base for lbl in unique_labels}
        for lbl in unique_labels[:rem]:
            target_per_label[lbl] += 1

        chosen_indices = []

        # внутри каждого sentiment-класса выбираем самые "чистые" примеры по p_clear
        for lbl in unique_labels:
            idx_lbl = np.where(labels == lbl)[0]
            if len(idx_lbl) == 0:
                continue

            scores_lbl = p_clear[idx_lbl]
            order = np.argsort(-scores_lbl)  # по убыванию p_clear
            take = min(target_per_label[lbl], len(idx_lbl))
            chosen_lbl = idx_lbl[order[:take]]
            chosen_indices.extend(chosen_lbl.tolist())

        # если не набрали k (например, какой-то класс редкий) — добираем лучших по p_clear
        if len(chosen_indices) < k:
            chosen_set = set(chosen_indices)
            remaining_idx = np.array(
                [i for i in range(n) if i not in chosen_set],
                dtype=int,
            )
            if len(remaining_idx) > 0:
                scores_rem = p_clear[remaining_idx]
                order_rem = np.argsort(-scores_rem)
                extra = remaining_idx[order_rem[: k - len(chosen_indices)]]
                chosen_indices.extend(extra.tolist())

        chosen_indices = np.array(chosen_indices[:k], dtype=int)
        return df.iloc[chosen_indices].copy()
