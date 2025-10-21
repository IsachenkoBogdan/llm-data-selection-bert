import os

import numpy as np
import pandas as pd

from .cache import get_cache
from .embeddings import compute_modernbert_embeddings
from .perplexity import compute_perplexity
from .text_metrics import compute_text_metrics


def _resolve_cache_dir(cfg, override: str | None = None) -> str:
    if override is not None:
        base = override if os.path.isabs(override) else os.path.join(cfg.paths.artifacts_dir, override)
    else:
        base = getattr(cfg.features, "cache_dir", None)
        if base is None:
            base = os.path.join(cfg.paths.artifacts_dir, "features")
    
    os.makedirs(base, exist_ok=True)
    return base


def ensure_modernbert_cls(
    cfg,
    df: pd.DataFrame,
    cache_dir: str | None = None,
    batch_size: int | None = None,
    force_recompute: bool = False,
) -> tuple[np.ndarray, dict]:
    cache_base = _resolve_cache_dir(cfg, cache_dir)
    cache = get_cache(cache_base, verbose=1)
    
    bs = batch_size or getattr(cfg.features.embeddings, "batch_size", 256)
    
    @cache.cache()
    def _compute(texts_tuple, model_name, max_length, batch_size):
        texts = pd.Series(list(texts_tuple))
        return compute_modernbert_embeddings(texts, model_name, max_length, batch_size)
    
    if force_recompute:
        cache.clear()
    
    embeddings = _compute(
        tuple(df["text"].tolist()),
        cfg.model.name,
        cfg.data.max_length,
        bs,
    )
    
    meta = {
        "model_name": cfg.model.name,
        "max_length": int(cfg.data.max_length),
        "batch_size": bs,
        "num_examples": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    
    return embeddings, meta


def ensure_pppl(
    cfg,
    df: pd.DataFrame,
    cache_dir: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache_base = _resolve_cache_dir(cfg, cache_dir)
    cache = get_cache(cache_base, verbose=1)
    
    pppl_cfg = getattr(cfg.features, "pppl", None)
    model_name = model_name or (pppl_cfg.model_name if pppl_cfg else "Qwen/Qwen3-1.7B-Base")
    batch_size = batch_size or (pppl_cfg.batch_size if pppl_cfg else 4)
    max_len = max_length or (pppl_cfg.max_length if pppl_cfg else cfg.data.max_length)
    add_start_token = (
        getattr(pppl_cfg, "add_start_token")
        if pppl_cfg and hasattr(pppl_cfg, "add_start_token")
        else False
    )
    
    @cache.cache()
    def _compute(texts_tuple, model_name, max_length, batch_size, add_start_token):
        texts = pd.Series(list(texts_tuple))
        return compute_perplexity(texts, model_name, max_length, batch_size, add_start_token)
    
    if force_recompute:
        cache.clear()
    
    pppl_series = _compute(
        tuple(df["text"].tolist()),
        model_name,
        max_len,
        batch_size,
        add_start_token,
    )
    
    table = pd.DataFrame({"pppl": pppl_series.values}, index=df.index)
    table.index.name = "id"
    
    meta = {
        "model_name": model_name,
        "max_length": int(max_len),
        "batch_size": int(batch_size),
        "num_examples": int(len(df)),
    }
    
    return table, meta


def ensure_quality_indicators(
    cfg,
    df: pd.DataFrame,
    cache_dir: str | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache_base = _resolve_cache_dir(cfg, cache_dir)
    cache = get_cache(cache_base, verbose=1)
    
    @cache.cache()
    def _compute(texts_tuple):
        texts = pd.Series(list(texts_tuple))
        return compute_text_metrics(texts)
    
    if force_recompute:
        cache.clear()
    
    metrics_df = _compute(tuple(df["text"].tolist()))
    
    table = metrics_df.copy()
    table.index = df.index
    table.index.name = "id"
    
    meta = {
        "num_examples": int(len(df)),
        "generated_with": "textstat",
    }
    
    return table, meta
