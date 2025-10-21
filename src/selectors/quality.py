import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .base import BaseSelector
from ..features import ensure_pppl, ensure_quality_indicators


class PerplexitySel(BaseSelector):
    def __init__(
        self,
        p_low: float = 0.3,
        p_high: float = 0.7,
        use_quality: bool = True,
        length_weight: float = 0.2,
        ttr_weight: float = 0.2,
        readability_weight: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.use_quality = bool(use_quality)
        self.length_weight = float(length_weight)
        self.ttr_weight = float(ttr_weight)
        self.readability_weight = float(readability_weight)
        self.normalize = bool(normalize)
        self._scores: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, cfg=None):
        pppl_scores, _ = ensure_pppl(cfg, df)
        
        if self.use_quality:
            quality, _ = ensure_quality_indicators(cfg, df)
            self._scores = pppl_scores.join(quality, how="left")
        else:
            self._scores = pppl_scores
        
        return self

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        if not self.normalize:
            return series
        
        scaler = MinMaxScaler()
        values = series.values.reshape(-1, 1)
        normalized = scaler.fit_transform(values).flatten()
        
        return pd.Series(normalized, index=series.index)

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        scores = self._scores.copy()
        
        q_low, q_high = sorted([np.clip(self.p_low, 0.0, 1.0), np.clip(self.p_high, 0.0, 1.0)])
        lower = np.nanpercentile(scores["pppl"], q_low * 100)
        upper = np.nanpercentile(scores["pppl"], q_high * 100)
        
        mask = scores["pppl"].between(lower, upper, inclusive="both")
        candidates = scores[mask]

        if self.use_quality and len(candidates) > 0:
            quality_components = []
            
            if "token_count" in candidates.columns:
                quality_components.append(
                    self.length_weight * self._normalize_series(-candidates["token_count"])
                )
            
            if "type_token_ratio" in candidates.columns:
                quality_components.append(
                    self.ttr_weight * self._normalize_series(candidates["type_token_ratio"])
                )
            
            if "readability" in candidates.columns:
                quality_components.append(
                    self.readability_weight * self._normalize_series(candidates["readability"])
                )
            
            if quality_components:
                quality_score = sum(quality_components)
                candidates = candidates.assign(quality_score=quality_score)
                candidates = candidates.sort_values("quality_score", ascending=False)

        k = min(k, len(candidates))
        if k <= 0:
            return df.iloc[[]]

        top_ids = candidates.index[:k]
        return df.loc[top_ids]
