# src/features/llm_clarity.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----- Levenshtein (быстрая версия через rapidfuzz; если нет, будет медленный фоллбэк) -----

try:
    from rapidfuzz.distance import Levenshtein as RFLev

    def normalized_levenshtein(a: str, b: str) -> float:
        # 0 = идентично, 1 = максимально различно
        return RFLev.normalized_distance(a, b)

except ImportError:
    def _levenshtein(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            ca = a[i - 1]
            for j in range(1, lb + 1):
                temp = dp[j]
                cost = 0 if ca == b[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,     # deletion
                    dp[j - 1] + 1, # insertion
                    prev + cost,   # substitution
                )
                prev = temp
        return dp[lb]

    def normalized_levenshtein(a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        d = _levenshtein(a, b)
        return d / max(len(a), len(b), 1)


def select_diverse_subset_levenshtein(
    texts: pd.Series,
    target_frac: float = 0.10,
    max_pool_size: int = 5000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Жадный farthest point sampling по нормализованному расстоянию Левенштейна.
    Возвращает индексы (в терминах исходной Series).
    """
    rng = np.random.default_rng(random_state)
    n_total = len(texts)
    if n_total == 0:
        return np.array([], dtype=int)

    target_size = max(1, int(round(n_total * target_frac)))

    if n_total > max_pool_size:
        pool_indices = rng.choice(n_total, size=max_pool_size, replace=False)
    else:
        pool_indices = np.arange(n_total)

    pool_texts = texts.iloc[pool_indices].astype(str).tolist()
    m = len(pool_texts)

    if m <= target_size:
        return pool_indices

    # стартовая точка
    first_idx = int(rng.integers(low=0, high=m))
    selected_pool = [first_idx]

    min_dist = np.full(shape=(m,), fill_value=np.inf, dtype=float)
    min_dist[first_idx] = 0.0

    for j in range(m):
        if j == first_idx:
            continue
        d = normalized_levenshtein(pool_texts[first_idx], pool_texts[j])
        min_dist[j] = min(min_dist[j], d)

    while len(selected_pool) < target_size:
        cand_idx = int(np.argmax(min_dist))
        if cand_idx in selected_pool:
            break
        selected_pool.append(cand_idx)
        for j in range(m):
            if j in selected_pool:
                continue
            d = normalized_levenshtein(pool_texts[cand_idx], pool_texts[j])
            if d < min_dist[j]:
                min_dist[j] = d

    selected_pool = np.array(selected_pool, dtype=int)
    selected_global_idx = pool_indices[selected_pool]
    return selected_global_idx


# ----- Qwen2.5-7B для разметки однозначности тональности (0/1/2) -----

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 8
STUDENT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_INSTRUCTIONS = """You are an expert annotator for a sentiment analysis dataset of short movie reviews.

Your task is to judge how CLEAR the sentiment (positive or negative) is in a given review.

Classes:
0 - CLEAR_SENTIMENT: The review clearly expresses a positive or negative sentiment. A human annotator would almost surely agree on the label.
1 - SOMEWHAT_AMBIGUOUS: The review has mixed signals or depends on context, but sentiment is still somewhat present.
2 - VERY_AMBIGUOUS_OR_UNCLEAR: The sentiment is unclear, contradictory, almost neutral, or highly context-dependent.

Output ONLY a single integer: 0, 1, or 2.
"""


def _build_qwen_prompt(text: str) -> str:
    return (
        LLM_INSTRUCTIONS
        + "\n\nReview:\n"
        + text.strip()
        + "\n\nAnswer with a single integer (0, 1, or 2):"
    )


def _load_qwen2_5(model_name: str = QWEN_MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return tokenizer, model, device


def classify_clarity_with_qwen(texts: List[str], model_name: str = QWEN_MODEL_NAME) -> np.ndarray:
    """
    Qwen2.5-7B размечает каждый текст классом 0/1/2 по однозначности тональности.
    """
    tokenizer, model, device = _load_qwen2_5(model_name)

    labels = []
    for t in texts:
        prompt = _build_qwen_prompt(t)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full = tokenizer.decode(out[0], skip_special_tokens=True)
        gen = full[len(prompt) :].strip()
        label = 1  # default = "somewhat ambiguous"
        for ch in gen:
            if ch in "012":
                label = int(ch)
                break
        labels.append(label)

    return np.asarray(labels, dtype=np.int64)


# ----- Основная фича: ensure_llm_clarity_scores -----


def ensure_llm_clarity_scores(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Гарантирует наличие parquet с LLM-based clarity score для всех строк df.

    Возвращает DataFrame с колонками:
      - idx (индекс исходного df)
      - llm_clarity_p0, llm_clarity_p1, llm_clarity_p2 (probabilities)
      - llm_clarity_pred (argmax по студенту)
    """
    artifacts_dir = getattr(cfg.paths, "artifacts_dir", "artifacts") if cfg is not None else "artifacts"
    seed = int(getattr(cfg, "seed", 42)) if cfg is not None else 42

    feat_dir = Path(artifacts_dir) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    scores_path = feat_dir / "sst2_llm_clarity_scores.parquet"
    student_path = feat_dir / "llm_clarity_student.joblib"

    # если уже посчитано — просто читаем и выравниваем по idx
    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        if "idx" not in scores_df.columns:
            raise ValueError(f"{scores_path} must contain column 'idx'")
        # выравниваем по индексу df
        scores_df = scores_df.set_index("idx")
        # предполагаем, что df.index = [0..N-1]
        aligned = scores_df.loc[df.index].reset_index()
        return aligned

    # --- иначе считаем всё с нуля ---

    df = df.reset_index(drop=True)
    n_train = len(df)

    # 1) Левенштейн-разнообразный сабсет (~10% train)
    print("[llm_clarity] Selecting Levenshtein-diverse subset...")
    diverse_idx = select_diverse_subset_levenshtein(
        df["text"],
        target_frac=0.10,
        max_pool_size=5000,
        random_state=seed,
    )
    diverse_df = df.iloc[diverse_idx].reset_index(drop=False).rename(columns={"index": "orig_idx"})
    print(f"[llm_clarity] selected {len(diverse_df)} diverse examples out of {n_train}")

    # 2) Разметка Qwen
    print("[llm_clarity] Labeling diverse subset with Qwen2.5-7B (clarity 0/1/2)...")
    diverse_texts = diverse_df["text"].astype(str).tolist()
    llm_labels = classify_clarity_with_qwen(diverse_texts, model_name=QWEN_MODEL_NAME)
    diverse_df["llm_clarity_label"] = llm_labels

    diverse_out = feat_dir / "sst2_diverse_qwen_clarity.parquet"
    diverse_df.to_parquet(diverse_out, index=False)
    print(f"[llm_clarity] Saved labeled diverse subset to {diverse_out}")

    # 3) Студент: sentence-transformer + логрег
    print("[llm_clarity] Training student clarity classifier...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(STUDENT_EMBED_MODEL, device=device)

    embeddings = st_model.encode(
        diverse_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        llm_labels,
        test_size=0.2,
        random_state=seed,
        stratify=llm_labels if len(np.unique(llm_labels)) > 1 else None,
    )

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("[llm_clarity] === Student clarity classifier report (val) ===")
    print(classification_report(y_val, y_pred, digits=3))

    joblib.dump(
        {"clf": clf, "embed_model_name": STUDENT_EMBED_MODEL},
        student_path,
    )
    print(f"[llm_clarity] Saved student classifier to {student_path}")

    # 4) Считаем proba для ВСЕГО df
    print("[llm_clarity] Scoring full train with student (P(class=0/1/2))...")
    full_embeddings = st_model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    proba = clf.predict_proba(full_embeddings)  # shape (N, C)
    if proba.shape[1] < 3:
        raise RuntimeError(f"Student classifier has {proba.shape[1]} classes, expected 3")

    pred = np.argmax(proba, axis=1)

    scores_df = df.copy()
    scores_df["idx"] = scores_df.index
    scores_df["llm_clarity_p0"] = proba[:, 0].astype(np.float32)
    scores_df["llm_clarity_p1"] = proba[:, 1].astype(np.float32)
    scores_df["llm_clarity_p2"] = proba[:, 2].astype(np.float32)
    scores_df["llm_clarity_pred"] = pred.astype(np.int64)

    scores_df.to_parquet(scores_path, index=False)
    print(f"[llm_clarity] Saved full clarity scores to {scores_path}")

    summary = {
        "n_train": int(n_train),
        "n_diverse": int(len(diverse_df)),
        "clarity_label_counts": {
            int(k): int(v)
            for k, v in pd.Series(llm_labels).value_counts().sort_index().items()
        },
    }
    with open(feat_dir / "llm_clarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[llm_clarity] Summary:", json.dumps(summary, ensure_ascii=False, indent=2))

    return scores_df[["idx", "llm_clarity_p0", "llm_clarity_p1", "llm_clarity_p2", "llm_clarity_pred"]]
