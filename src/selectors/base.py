import pandas as pd


class BaseSelector:
    def fit(self, df: pd.DataFrame, cfg=None):
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        raise NotImplementedError


class RandomSel(BaseSelector):
    def fit(self, df: pd.DataFrame, cfg=None):
        self._seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        return df.sample(n=min(k, len(df)), random_state=self._seed)


class FullSel(BaseSelector):
    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        return df.iloc[:k] if k < len(df) else df
