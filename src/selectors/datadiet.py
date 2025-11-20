import numpy as np
import pandas as pd

from .base import BaseSelector
from ..features.data_diet import ensure_el2n_scores


class DataDietSel(BaseSelector):
    """
    'BERT on a Data Diet' style селектор.

    Шаги:
      1) считаем (или грузим из кэша) EL2N score для каждого train-примера;
      2) сортируем примеры по убыванию EL2N (чем выше, тем "важнее");
      3) берём top-k по score.

    k задаётся снаружи через cfg.subset.frac (5%, 10%, 20% и т.д.).
    """

    def fit(self, df: pd.DataFrame, cfg=None):
        # считаем / подгружаем EL2N для ВСЕГО df
        scores_df = ensure_el2n_scores(df, cfg)

        # выравниваем по индексу df (в df prepare stage индекс = позиция в parquet)
        scores_df = scores_df.set_index("idx")
        aligned = scores_df.loc[df.index]

        self._scores = aligned["el2n"].to_numpy()
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if not hasattr(self, "_scores"):
            raise RuntimeError("DataDietSel.fit must be вызван перед select().")

        n = len(df)
        k = min(k, n)
        if k <= 0:
            return df.iloc[0:0].copy()

        # top-k по EL2N (большие — "важные")
        order_desc = np.argsort(-self._scores)
        chosen = order_desc[:k]

        return df.iloc[chosen].copy()
