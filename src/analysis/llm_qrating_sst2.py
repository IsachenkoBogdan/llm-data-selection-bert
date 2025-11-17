# src/analysis/llm_qrating_sst2.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Для студента
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Для Левенштейна – желательно поставить rapidfuzz
try:
    from rapidfuzz.distance import Levenshtein as RFLev

    def normalized_levenshtein(a: str, b: str) -> float:
        # 0 = идентично, 1 = максимально различно
        return RFLev.normalized_distance(a, b)

except ImportError:
    # Фоллбэк: простая O(L^2) реализация, если rapidfuzz не установлен
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


# -----------------------------
# 1) Выбор «уникального» поднабора по Левенштейну
# -----------------------------

def select_diverse_subset_levenshtein(
    texts: pd.Series,
    target_frac: float = 0.10,
    max_pool_size: int = 5000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Грубо выбирает target_frac наиболее «непохожих» примеров
    (farthest point sampling по нормализованному расстоянию Левенштейна).

    Чтобы не умереть по времени, сначала уменьшаем пул до max_pool_size,
    а уже внутри него делаем жадный алгоритм.

    Возвращает индексы (в терминах исходной Series) выбранных строк.
    """
    rng = np.random.default_rng(random_state)
    n_total = len(texts)
    if n_total == 0:
        return np.array([], dtype=int)

    target_size = max(1, int(round(n_total * target_frac)))

    # Ограничиваем пул сверху – Левенштейн по всему SST-2 будет слишком тяжёлый
    if n_total > max_pool_size:
        pool_indices = rng.choice(n_total, size=max_pool_size, replace=False)
    else:
        pool_indices = np.arange(n_total)

    pool_texts = texts.iloc[pool_indices].astype(str).tolist()
    m = len(pool_texts)

    if m <= target_size:
        return pool_indices

    # Farthest point sampling в этом пуле
    # стартуем с случайной точки
    first_idx = int(rng.integers(low=0, high=m))
    selected_pool = [first_idx]

    # минимальная дистанция до любого выбранного
    min_dist = np.full(shape=(m,), fill_value=np.inf, dtype=float)
    min_dist[first_idx] = 0.0

    # предварительно считаем расстояния от первой точки
    for j in range(m):
        if j == first_idx:
            continue
        d = normalized_levenshtein(pool_texts[first_idx], pool_texts[j])
        min_dist[j] = min(min_dist[j], d)

    while len(selected_pool) < target_size:
        # выбираем точку с максимальной минимальной дистанцией
        cand_idx = int(np.argmax(min_dist))
        if cand_idx in selected_pool:
            # на всякий случай, но не должно происходить
            break
        selected_pool.append(cand_idx)

        # обновляем min_dist с учётом новой выбранной точки
        for j in range(m):
            if j in selected_pool:
                continue
            d = normalized_levenshtein(pool_texts[cand_idx], pool_texts[j])
            if d < min_dist[j]:
                min_dist[j] = d

    selected_pool = np.array(selected_pool, dtype=int)
    # маппим обратно на индексы в исходной Series
    selected_global_idx = pool_indices[selected_pool]
    return selected_global_idx


# -----------------------------
# 2) Разметка однозначности тональности Qwen2.5-7B
# -----------------------------

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # при необходимости поменяешь
QWEN_MAX_NEW_TOKENS = 8


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


def classify_clarity_with_qwen(
    texts: List[str],
    model_name: str = QWEN_MODEL_NAME,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Прогоняет список текстов через Qwen2.5-7B-Instruct и
    возвращает массив классов (0,1,2) по однозначности тональности.
    """
    tokenizer, model, device = _load_qwen2_5(model_name)

    labels: List[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        prompts = [_build_qwen_prompt(t) for t in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, prompt in enumerate(prompts):
            full_text = tokenizer.decode(
                outputs[i],
                skip_special_tokens=True,
            )
            # вырезаем только сгенерированную часть
            gen = full_text[len(prompt) :].strip()
            # берём первую цифру 0/1/2 из ответа
            label = 1  # default = "somewhat ambiguous"
            for ch in gen:
                if ch in "012":
                    label = int(ch)
                    break
            labels.append(label)

    return np.asarray(labels, dtype=np.int64)


# -----------------------------
# 3) Студент: sentence embeddings + LogisticRegression
# -----------------------------

STUDENT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def train_student_clarity_classifier(
    texts: List[str],
    labels: np.ndarray,
    embed_model_name: str = STUDENT_EMBED_MODEL,
    outdir: Path = Path("artifacts/llm_qrating"),
    seed: int = 42,
) -> Tuple[SentenceTransformer, LogisticRegression]:
    """
    Обучает маленького студента:
      - frozen sentence-transformer для эмбеддингов,
      - логистическая регрессия для предсказания класса (0/1/2).

    Сохраняет joblib со структурой {'clf': clf, 'embed_model_name': ...}.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(embed_model_name, device=device)

    # эмбеддинги
    embeddings = st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("=== Student clarity classifier report (val) ===")
    print(classification_report(y_val, y_pred, digits=3))

    joblib.dump(
        {"clf": clf, "embed_model_name": embed_model_name},
        outdir / "student_clarity_clf.joblib",
    )

    return st_model, clf


def load_student_clarity_classifier(
    ckpt_path: Path = Path("artifacts/llm_qrating/student_clarity_clf.joblib"),
) -> Tuple[SentenceTransformer, LogisticRegression]:
    bundle = joblib.load(ckpt_path)
    embed_model_name = bundle["embed_model_name"]
    clf: LogisticRegression = bundle["clf"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(embed_model_name, device=device)
    return st_model, clf


def score_clarity_with_student(
    texts: List[str],
    ckpt_path: Path = Path("artifacts/llm_qrating/student_clarity_clf.joblib"),
    clear_class: int = 0,
) -> np.ndarray:
    """
    Прогоняет все тексты через студента и возвращает quality score:
      score = P(class == clear_class).

    По умолчанию clear_class=0 → вероятность «однозначной тональности».
    """
    st_model, clf = load_student_clarity_classifier(ckpt_path)

    embeddings = st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    proba = clf.predict_proba(embeddings)
    if clear_class >= proba.shape[1]:
        raise ValueError(f"clear_class={clear_class} but clf has only {proba.shape[1]} classes")

    scores = proba[:, clear_class]
    return scores.astype(np.float32)


# -----------------------------
# 4) End-to-end для SST-2
# -----------------------------

def run_qrating_sst2(
    target_frac: float = 0.10,
    max_pool_size: int = 5000,
    seed: int = 42,
    out_dir: Path = Path("artifacts/llm_qrating"),
):
    """
    Полный цикл:
      1) Загружаем SST-2 (train).
      2) Выбираем Левенштейн-уникальные ~10% примеров.
      3) Размечаем их Qwen2.5 по однозначности тональности (0/1/2).
      4) Обучаем студента (MiniLM + логрег).
      5) Прогоняем студента по всему train и сохраняем quality_score.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Загружаем SST-2 ---
    ds = load_dataset("glue", "sst2")
    df_train = ds["train"].to_pandas()
    df_train = df_train.rename(columns={"sentence": "text"})  # чтобы быть консистентным с твоим пайплайном
    df_train = df_train.reset_index(drop=True)

    # --- 2. Левенштейн-уникальный поднабор ---
    print("Selecting Levenshtein-diverse subset...")
    diverse_idx = select_diverse_subset_levenshtein(
        df_train["text"],
        target_frac=target_frac,
        max_pool_size=max_pool_size,
        random_state=seed,
    )
    diverse_df = df_train.iloc[diverse_idx].reset_index(drop=False).rename(columns={"index": "orig_id"})
    print(f"Selected {len(diverse_df)} diverse examples out of {len(df_train)}")

    # --- 3. Разметка Qwen2.5 ---
    print("Labeling diverse subset with Qwen2.5-7B (clarity classes 0/1/2)...")
    diverse_texts = diverse_df["text"].astype(str).tolist()
    llm_labels = classify_clarity_with_qwen(diverse_texts, model_name=QWEN_MODEL_NAME, batch_size=1)
    diverse_df["llm_clarity_label"] = llm_labels

    # сохраняем размеченный поднабор
    diverse_path = out_dir / "sst2_diverse_qwen_clarity.parquet"
    diverse_df.to_parquet(diverse_path, index=False)
    print(f"Saved diverse labeled subset to {diverse_path}")

    # --- 4. Обучаем студента ---
    print("Training student clarity classifier on LLM labels...")
    student_model, clf = train_student_clarity_classifier(
        texts=diverse_df["text"].tolist(),
        labels=llm_labels,
        outdir=out_dir,
        seed=seed,
    )

    # --- 5. Прогоняем студента по всему train ---
    print("Scoring full SST-2 train with student clarity classifier...")
    full_scores = score_clarity_with_student(
        df_train["text"].tolist(),
        ckpt_path=out_dir / "student_clarity_clf.joblib",
        clear_class=0,
    )

    df_scores = df_train.copy()
    df_scores["llm_qrating_clarity_score"] = full_scores

    scores_path = out_dir / "sst2_train_qrating_scores.parquet"
    df_scores.to_parquet(scores_path, index=False)
    print(f"Saved full train with quality scores to {scores_path}")

    # маленький summary
    summary = {
        "target_frac": target_frac,
        "max_pool_size": max_pool_size,
        "n_train": len(df_train),
        "n_diverse": len(diverse_df),
        "score_stats": {
            "min": float(full_scores.min()),
            "max": float(full_scores.max()),
            "mean": float(full_scores.mean()),
            "std": float(full_scores.std()),
        },
        "clarity_label_counts": {
            int(k): int(v)
            for k, v in pd.Series(llm_labels).value_counts().sort_index().items()
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_qrating_sst2()
