import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

from .base import BaseSelector
from ..features import ensure_modernbert_cls


class KCenterSel(BaseSelector):
    def __init__(self, cache_dir: str = "features", batch_size: int = 1024, **kwargs):
        self.cache_dir = cache_dir
        self.batch_size = int(batch_size)
        self._embeddings: np.ndarray | None = None
        self._seed: int = 0

    def fit(self, df: pd.DataFrame, cfg=None):
        embeddings, _ = ensure_modernbert_cls(
            cfg,
            df,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )
        self._embeddings = embeddings
        self._seed = int(getattr(cfg, "seed", 0))
        return self

    def _update_min_distances(self, center_idx: int, min_dists: np.ndarray) -> None:
        center = self._embeddings[center_idx].reshape(1, -1)
        for start in range(0, len(self._embeddings), self.batch_size):
            end = min(start + self.batch_size, len(self._embeddings))
            batch = self._embeddings[start:end]
            distances = euclidean_distances(batch, center).flatten()
            min_dists[start:end] = np.minimum(min_dists[start:end], distances)
        min_dists[center_idx] = 0.0

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        n = len(df)
        if n == 0 or k <= 0:
            return df.iloc[[]]
        k = min(k, n)

        rng = np.random.default_rng(self._seed)
        chosen = [int(rng.integers(0, n))]
        min_dists = np.full(n, np.inf, dtype=np.float64)
        self._update_min_distances(chosen[0], min_dists)

        while len(chosen) < k:
            idx = int(np.argmax(min_dists))
            chosen.append(idx)
            self._update_min_distances(idx, min_dists)

        return df.iloc[chosen]


class KMeansSel(BaseSelector):
    def __init__(
        self,
        cache_dir: str = "features",
        n_clusters: int = 128,
        batch_size: int = 1024,
        random_state: int | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self._embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, cfg=None):
        embeddings, _ = ensure_modernbert_cls(
            cfg,
            df,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )
        self._embeddings = embeddings
        
        random_state = (
            self.random_state
            if self.random_state is not None
            else getattr(cfg, "seed", 0)
        )
        kmeans = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(df)),
            batch_size=self.batch_size,
            random_state=random_state,
            n_init="auto",
        )
        self._labels = kmeans.fit_predict(self._embeddings)
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        n = len(df)
        if n == 0 or k <= 0:
            return df.iloc[[]]
        k = min(k, n)

        cluster_ids, counts = np.unique(self._labels, return_counts=True)
        proportions = counts / counts.sum()
        quotas = np.maximum(1, np.round(proportions * k)).astype(int)

        selected_indices = []
        rng = np.random.default_rng(self.random_state or 0)
        
        for cluster_id, quota in zip(cluster_ids, quotas):
            cluster_indices = np.where(self._labels == cluster_id)[0]
            
            if len(cluster_indices) <= quota:
                selected_indices.extend(cluster_indices.tolist())
            else:
                chosen = rng.choice(cluster_indices, size=quota, replace=False)
                selected_indices.extend(chosen.tolist())
            
            if len(selected_indices) >= k:
                break

        return df.iloc[selected_indices[:k]]


class HerdingSel(BaseSelector):
    def __init__(
        self,
        cache_dir: str = "features",
        batch_size: int = 1024,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.batch_size = int(batch_size)
        self._embeddings: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, cfg=None):
        embeddings, _ = ensure_modernbert_cls(
            cfg,
            df,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )
        self._embeddings = embeddings
        self._mean = embeddings.mean(axis=0)
        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        n = len(df)
        if n == 0 or k <= 0:
            return df.iloc[[]]
        k = min(k, n)

        selected = []
        residual = self._mean.copy()
        available = np.arange(n)

        for _ in range(k):
            scores = self._embeddings[available] @ residual
            best_idx = int(np.argmax(scores))
            chosen_idx = int(available[best_idx])
            
            selected.append(chosen_idx)
            residual -= self._embeddings[chosen_idx]
            available = np.delete(available, best_idx)

        return df.iloc[selected]
