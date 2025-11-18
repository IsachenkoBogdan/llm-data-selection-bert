import numpy as np
import pandas as pd
from typing import Dict

from .base import BaseSelector
from ..features import (
    ensure_modernbert_cls,
    ensure_pppl,
    ensure_quality_indicators,
)

try:
    # опционально: если есть модуль с предиктивной энтропией – используем
    from ..features.predictive_stats import ensure_predictive_entropy

    HAS_PRED_ENTROPY = True
except Exception:
    HAS_PRED_ENTROPY = False

from sklearn.cluster import KMeans


def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or (M - m) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m)


def _compute_target_counts(labels: np.ndarray, k: int) -> Dict[int, int]:
    """Сколько примеров каждого класса нужно взять, чтобы суммарно было k."""
    uniq, counts = np.unique(labels, return_counts=True)
    fracs = counts / counts.sum()

    raw = fracs * k
    base = np.floor(raw).astype(int)
    missing = k - base.sum()

    fractional = raw - base
    order = np.argsort(-fractional)

    for idx in order[:missing]:
        base[idx] += 1

    return {int(c): int(n) for c, n in zip(uniq, base)}


class HybridQDISel(BaseSelector):
    """
    Гибридный селектор Quality–Diversity–Importance.

    Идея:
      - Quality: перплексия Qwen + readability (Flesch).
      - Diversity: размер кластера ModernBERT-эмбеддингов (KMeans) –
                   редкие кластеры считаются более разнообразными.
      - Importance: предиктивная энтропия ModernBERT (если доступна),
                    иначе "средняя" перплексия как proxy сложности.

    Алгоритм:
      1) считаем Q, D, I для каждого примера;
      2) склеиваем в qdi_score = α_q*Q + α_i*I + α_d*D;
      3) выбрасываем нижний quantile по качеству (quality_quantile_cut);
      4) из оставшихся выбираем top-k по qdi_score так, чтобы
         сохранить распределение классов как в исходном train.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        alpha_q: float = 0.4,
        alpha_i: float = 0.4,
        alpha_d: float = 0.2,
        quality_quantile_cut: float = 0.1,
        max_kmeans_points: int = 50000,
    ):
        self.n_clusters = int(n_clusters)
        self.alpha_q = float(alpha_q)
        self.alpha_i = float(alpha_i)
        self.alpha_d = float(alpha_d)
        self.quality_quantile_cut = float(quality_quantile_cut)
        self.max_kmeans_points = int(max_kmeans_points)

    def fit(self, df: pd.DataFrame, cfg=None):
        if "label" not in df.columns:
            raise ValueError("HybridQDISel expects a 'label' column in df.")

        # ===== 1) Эмбеддинги ModernBERT =====
        emb_df = ensure_modernbert_cls(df, cfg)
        # ожидаем формат с колонкой "idx" и колонкой "emb" (np.ndarray)
        emb_df = emb_df.set_index("idx").loc[df.index]
        X = np.stack(emb_df["emb"].to_numpy()).astype("float32")
        self._X = X
        self._labels = df["label"].to_numpy()

        # ===== 2) Перплексия (Qwen) =====
        pppl_df = ensure_pppl(df, cfg)
        pppl_df = pppl_df.set_index("idx").loc[df.index]
        pppl = pppl_df["pppl"].to_numpy().astype("float32")
        # quality: низкая перплексия → лучше (но без особой хитрости)
        log_pppl = np.log1p(pppl)
        q_pppl = 1.0 - _minmax(log_pppl)

        # ===== 3) Текстовые метрики (readability) =====
        qual_df = ensure_quality_indicators(df, cfg)
        qual_df = qual_df.set_index("idx").loc[df.index]
        if "flesch_reading_ease" in qual_df.columns:
            fre = qual_df["flesch_reading_ease"].to_numpy(dtype=np.float32)
            q_read = _minmax(fre)
        else:
            q_read = np.zeros_like(q_pppl, dtype=np.float32)

        # общий quality-score
        q_score = 0.7 * q_pppl + 0.3 * q_read
        q_score = _minmax(q_score)

        # ===== 4) Importance =====
        if HAS_PRED_ENTROPY:
            ent_df = ensure_predictive_entropy(df, cfg)
            ent_df = ent_df.set_index("idx").loc[df.index]
            # пытаемся угадать правильную колонку энтропии
            entropy_cols = [c for c in ent_df.columns if c not in ("idx",)]
            if not entropy_cols:
                raise ValueError(
                    "ensure_predictive_entropy returned no entropy-like columns."
                )
            ent = ent_df[entropy_cols[0]].to_numpy(dtype=np.float32)
            i_score = _minmax(ent)
        else:
            # fallback: "средняя" сложность — примеры не слишком простые и не экстремально сложные
            logp = log_pppl
            med = float(np.median(logp))
            dev = np.abs(logp - med)
            # чем ближе к медиане → тем выше "важность"
            i_score = 1.0 - _minmax(dev)

        # ===== 5) Diversity через KMeans-кластеры =====
        rng_seed = int(getattr(cfg, "seed", 0)) if cfg is not None else 0
        rng = np.random.default_rng(rng_seed)

        n = X.shape[0]
        if n <= self.n_clusters:
            # тривиальный случай: кластеров больше, чем точек
            cluster_ids = np.zeros(n, dtype=int)
        else:
            if n > self.max_kmeans_points:
                # подвыборка для обучения kmeans, потом predict на всех
                idx_perm = rng.permutation(n)
                fit_idx = idx_perm[: self.max_kmeans_points]
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=rng_seed,
                    n_init="auto",
                )
                kmeans.fit(X[fit_idx])
                cluster_ids = kmeans.predict(X)
            else:
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=rng_seed,
                    n_init="auto",
                )
                cluster_ids = kmeans.fit_predict(X)

        cluster_ids = np.asarray(cluster_ids, dtype=int)
        max_cluster = int(cluster_ids.max()) if len(cluster_ids) > 0 else -1
        counts = np.bincount(cluster_ids, minlength=max_cluster + 1).astype(
            np.float32
        )
        counts[counts == 0.0] = 1.0
        # редкие кластеры → высокий diversity-score
        d_score = 1.0 / counts[cluster_ids]
        d_score = _minmax(d_score)

        # ===== 6) Комбинируем Q, D, I =====
        qdi_score = (
            self.alpha_q * q_score
            + self.alpha_i * i_score
            + self.alpha_d * d_score
        )
        qdi_score = _minmax(qdi_score)

        # Жёсткий фильтр по качеству: выбросим нижний quantile
        if 0.0 < self.quality_quantile_cut < 0.5:
            thr = float(np.quantile(q_score, self.quality_quantile_cut))
            keep_mask = q_score >= thr
        else:
            keep_mask = np.ones(n, dtype=bool)

        self._q_score = q_score
        self._i_score = i_score
        self._d_score = d_score
        self._qdi_score = qdi_score
        self._keep_mask = keep_mask
        self._seed = rng_seed

        return self

    def select(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if not hasattr(self, "_qdi_score"):
            raise RuntimeError("HybridQDISel.fit must be called before select().")

        n = len(df)
        if n == 0 or k <= 0:
            return df.iloc[0:0].copy()

        k = min(k, n)

        scores = self._qdi_score
        keep_mask = self._keep_mask
        labels = self._labels

        idx_all = np.arange(n, dtype=int)
        idx_candidates = idx_all[keep_mask]

        if len(idx_candidates) == 0:
            # если фильтр качества выжег всё – fallback к top-k по qdi без фильтра
            idx_candidates = idx_all

        if len(idx_candidates) <= k:
            chosen = idx_candidates
            return df.iloc[np.sort(chosen)].copy()

        labels_c = labels[idx_candidates]

        # целевое количество по классам (по распределению на кандидатов)
        target_counts = _compute_target_counts(labels_c, k)

        chosen_indices = []

        # сначала набираем по классам, сортируя по qdi_score
        for cls, target_k in target_counts.items():
            cls_mask = labels_c == cls
            idx_cls = idx_candidates[cls_mask]
            if len(idx_cls) == 0 or target_k <= 0:
                continue

            scores_cls = scores[idx_cls]
            order = np.argsort(-scores_cls)  # по убыванию
            take = min(target_k, len(idx_cls))
            chosen_indices.extend(idx_cls[order[:take]].tolist())

        chosen_indices = list(dict.fromkeys(chosen_indices))  # dedup

        # если не дотянули до k — добираем глобально по qdi_score
        if len(chosen_indices) < k:
            already = set(chosen_indices)
            remaining = np.array(
                [i for i in idx_candidates if i not in already], dtype=int
            )
            if len(remaining) > 0:
                scores_rem = scores[remaining]
                order_rem = np.argsort(-scores_rem)
                extra = remaining[order_rem[: k - len(chosen_indices)]]
                chosen_indices.extend(extra.tolist())

        # если перебрали – обрежем по лучшему score
        if len(chosen_indices) > k:
            chosen_arr = np.array(chosen_indices, dtype=int)
            order_glob = np.argsort(-scores[chosen_arr])
            chosen_arr = chosen_arr[order_glob[:k]]
        else:
            chosen_arr = np.array(chosen_indices, dtype=int)

        return df.iloc[np.sort(chosen_arr)].copy()
