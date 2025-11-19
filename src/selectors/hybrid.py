# src/selectors/hybrid.py

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from .base import BaseSelector
from ..features import (
    ensure_pppl,
    ensure_quality_indicators,
)

# опциональная важность из предиктивной энтропии
try:
    from ..features.predictive_stats import ensure_predictive_entropy

    HAS_PRED_ENTROPY = True
except Exception:
    HAS_PRED_ENTROPY = False


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
    missing = k - int(base.sum())

    fractional = raw - base
    order = np.argsort(-fractional)

    for idx in order[:missing]:
        base[idx] += 1

    return {int(c): int(n) for c, n in zip(uniq, base)}


class _CLSDataset(Dataset):
    """Простой датасет для получения CLS-эмбеддингов."""

    def __init__(self, texts: List[str]):
        self.texts = list(map(str, texts))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx]}


def _compute_cls_embeddings(df: pd.DataFrame, cfg) -> np.ndarray:
    """
    Считаем CLS-эмбеддинги ModernBERT без использования ensure_modernbert_cls.
    Возвращает массив shape (N, D), где N = len(df).
    """
    # определяем колонку с текстом
    if "text" in df.columns:
        text_col = "text"
    elif "sentence" in df.columns:
        text_col = "sentence"
    else:
        raise ValueError(
            f"_compute_cls_embeddings: expected 'text' or 'sentence' column, got {list(df.columns)}"
        )

    model_name = getattr(getattr(cfg, "model", None), "name", "answerdotai/ModernBERT-base")
    max_length = getattr(getattr(cfg, "data", None), "max_length", 128)

    # batch_size: пробуем взять из cfg.features.embeddings.batch_size, иначе дефолт
    bs = 512
    try:
        bs = int(cfg.features.embeddings.batch_size)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = _CLSDataset(df[text_col].tolist())
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)

    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            enc = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            # CLS: первый токен
            hidden = outputs.last_hidden_state  # (B, L, D)
            cls = hidden[:, 0, :]               # (B, D)
            all_embs.append(cls.cpu().numpy())

    X = np.concatenate(all_embs, axis=0)
    assert X.shape[0] == len(df), "Mismatch between embeddings and df length"
    return X.astype("float32")


class HybridQDISel(BaseSelector):
    """
    Гибридный селектор Quality–Diversity–Importance.

    • Quality: перплексия Qwen + readability (Flesch).
    • Diversity: редкость кластера ModernBERT CLS (KMeans).
    • Importance: предиктивная энтропия ModernBERT (если есть),
                  иначе "средняя сложность" по перплексии.
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

        if cfg is None:
            raise ValueError("HybridQDISel.fit requires cfg (Hydra config).")

        # ==== 1) CLS-эмбеддинги ModernBERT ====
        X = _compute_cls_embeddings(df, cfg)
        self._X = X
        self._labels = df["label"].to_numpy()

        # ==== 2) Перплексия (Qwen) ====
        # ensure_pppl ожидает (cfg, df), но возвращает tuple или DF
        pppl_out = ensure_pppl(cfg, df)
        if isinstance(pppl_out, tuple):
            pppl_df = pppl_out[0]
        else:
            pppl_df = pppl_out

        pppl_df = pppl_df.set_index("idx").loc[df.index]
        pppl = pppl_df["pppl"].to_numpy().astype("float32")

        log_pppl = np.log1p(pppl)
        # меньше лог-перплексии → лучше качество
        q_pppl = 1.0 - _minmax(log_pppl)

        # ==== 3) Текстовые метрики (readability) ====
        qual_out = ensure_quality_indicators(cfg, df)
        if isinstance(qual_out, tuple):
            qual_df = qual_out[0]
        else:
            qual_df = qual_out

        qual_df = qual_df.set_index("idx").loc[df.index]
        if "flesch_reading_ease" in qual_df.columns:
            fre = qual_df["flesch_reading_ease"].to_numpy(dtype=np.float32)
            q_read = _minmax(fre)
        else:
            q_read = np.zeros_like(q_pppl, dtype=np.float32)

        q_score = 0.7 * q_pppl + 0.3 * q_read
        q_score = _minmax(q_score)

        # ==== 4) Importance ====
        if HAS_PRED_ENTROPY:
            # нашу ensure_predictive_entropy мы писали как (df, cfg) и она возвращает DF
            ent_df = ensure_predictive_entropy(df, cfg)
            ent_df = ent_df.set_index("idx").loc[df.index]
            entropy_cols = [c for c in ent_df.columns if c not in ("idx",)]
            if not entropy_cols:
                raise ValueError(
                    "ensure_predictive_entropy returned no entropy-like columns."
                )
            ent = ent_df[entropy_cols[0]].to_numpy(dtype=np.float32)
            i_score = _minmax(ent)
        else:
            # fallback: "средняя сложность" — ближе к медиане лог-перплексии → выше важность
            logp = log_pppl
            med = float(np.median(logp))
            dev = np.abs(logp - med)
            i_score = 1.0 - _minmax(dev)

        # ==== 5) Diversity через KMeans на CLS ====
        rng_seed = int(getattr(cfg, "seed", 0))
        rng = np.random.default_rng(rng_seed)

        n = X.shape[0]
        if n <= self.n_clusters:
            cluster_ids = np.zeros(n, dtype=int)
        else:
            if n > self.max_kmeans_points:
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

        # ==== 6) Комбинация Q, I, D ====
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

        chosen_indices: List[int] = []

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

        # dedup
        chosen_indices = list(dict.fromkeys(chosen_indices))

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
