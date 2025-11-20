# src/experiments/train_llm_proxy_subsets.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# подстрой под свой пакет, если модуль metrics лежит иначе
from ..metrics import compute_metrics


# ========= НАСТРОЙКИ =========

# где лежит датасет с LLM-метками
# поменяй путь при необходимости: полный датасет или unique-10%-per-class
LLM_PROXIES_PATH = Path("artifacts/features/sst2_gemini_proxies.parquet")

# выходная директория для моделей и логов
OUT_BASE_DIR = Path("artifacts/llm_proxy_subsets")

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128
SEED = 42

SUBSET_FRAC = 0.10  # 10% от ВСЕГО числа примеров в датасете


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def _compute_target_counts(
    labels: np.ndarray,
    total_k: int,
) -> Dict[int, int]:
    """
    Вычисляет, сколько примеров каждого класса нужно взять, чтобы:
      - суммарно было total_k,
      - пропорции были максимально похожи на исходные.
    """
    uniques, counts = np.unique(labels, return_counts=True)
    fracs = counts / counts.sum()

    # базовое распределение
    raw_counts = fracs * total_k
    base_counts = np.floor(raw_counts).astype(int)
    missing = total_k - base_counts.sum()

    # добиваем остаток по убыванию дробной части
    fractional = raw_counts - base_counts
    order = np.argsort(-fractional)

    for idx in order[:missing]:
        base_counts[idx] += 1

    return {int(lbl): int(c) for lbl, c in zip(uniques, base_counts)}


def stratified_topk_by_score(
    df: pd.DataFrame,
    score_col: str,
    subset_frac: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Выбирает top-k по score_col, сохраняя распределение классов
    примерно как в исходном df['label'].

    Логика:
      1. считаем K = frac * N;
      2. считаем target_counts[class] по исходным пропорциям;
      3. для каждого класса сортируем по score_col убыванию
         (NaN улетают в конец) и берём top target_counts[c];
      4. если где-то не хватает примеров — остаток добираем
         глобально по лучшему score_col среди оставшихся.
    """
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in df.")

    n_total = len(df)
    if n_total == 0:
        return df

    k_total = max(1, int(round(subset_frac * n_total)))

    labels = df["label"].to_numpy()
    target_counts = _compute_target_counts(labels, k_total)

    selected_idx = []

    # сначала набираем внутри каждого класса
    for cls, target_k in target_counts.items():
        df_c = df[df["label"] == cls].copy()
        if score_col not in df_c.columns:
            raise ValueError(f"Column '{score_col}' not found in df.")

        df_c = df_c.sort_values(
            score_col,
            ascending=False,
            na_position="last",
        )
        take = min(target_k, len(df_c))
        selected_idx.extend(df_c.index[:take].tolist())

    selected_idx = list(dict.fromkeys(selected_idx))  # dedup, preserve order

    # если набрали меньше, чем нужно — добираем глобально
    if len(selected_idx) < k_total:
        remaining_needed = k_total - len(selected_idx)
        already = set(selected_idx)
        df_rem = df[~df.index.isin(already)].copy()
        df_rem = df_rem.sort_values(
            score_col,
            ascending=False,
            na_position="last",
        )
        extra_idx = df_rem.index[:remaining_needed].tolist()
        selected_idx.extend(extra_idx)

    # на случай, если переполнили (лучше подстраховаться)
    selected_idx = selected_idx[:k_total]

    subset = df.loc[selected_idx].sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    return subset


def print_label_stats(name: str, df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    fracs = df["label"].value_counts(normalize=True).sort_index()
    print(f"[{name}] size={len(df)}")
    print(f"  label_counts: {counts.to_dict()}")
    print(
        "  label_frac:   "
        + ", ".join(f"{int(k)}: {v:.3f}" for k, v in fracs.to_dict().items())
    )


# ========= ОБУЧЕНИЕ МОДЕЛИ НА ПОДВЫБОРКЕ =========

def train_on_subset(
    subset_name: str,
    df_subset: pd.DataFrame,
    out_base_dir: Path = OUT_BASE_DIR,
) -> Tuple[str, Dict]:
    """
    Обучает ModernBERT на df_subset["sentence"], df_subset["label"].
    Делит subset на train/val (90/10) стратифицированно.
    Возвращает (model_dir, metrics_dict).
    """
    out_dir = out_base_dir / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # выделяем фичи
    df = df_subset[["sentence", "label"]].dropna().reset_index(drop=True)

    # train/val split
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, random_state=SEED
    )
    y = df["label"].to_numpy()
    train_idx, val_idx = next(splitter.split(df, y))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def encode(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = (
        Dataset.from_pandas(train_df, preserve_index=False)
        .map(encode, batched=True)
        .rename_column("label", "labels")
        .with_format("torch")
    )
    val_ds = (
        Dataset.from_pandas(val_df, preserve_index=False)
        .map(encode, batched=True)
        .rename_column("label", "labels")
        .with_format("torch")
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_steps=100,
        logging_steps=100,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to=[],  # без W&B, при желании можно включить
        evaluation_strategy="steps",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(
            p.label_ids, p.predictions.argmax(-1)
        ),
    )

    print(f"\n=== Training on subset: {subset_name} ===")
    trainer.train()
    metrics = trainer.evaluate()

    # сохраняем
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Metrics for {subset_name}: {metrics}")
    print(f"Saved to: {out_dir}")

    return str(out_dir), metrics


# ========= ОСНОВНОЙ СКРИПТ =========

def main():
    if not LLM_PROXIES_PATH.exists():
        raise FileNotFoundError(
            f"LLM proxy dataset not found at {LLM_PROXIES_PATH}. "
            f"Change LLM_PROXIES_PATH or generate it first."
        )

    print(f"Loading LLM-annotated dataset from: {LLM_PROXIES_PATH}")
    df = pd.read_parquet(LLM_PROXIES_PATH)

    # если есть split — берем только train
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy().reset_index(drop=True)

    # базовая проверка
    needed_cols = [
        "sentence",
        "label",
        "quality_score",
        "diversity_score",
        "importance_score",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in LLM dataset: {missing}")

    print(f"Total examples in LLM dataset (train split): {len(df)}")

    # посмотрим исходное распределение классов
    print_label_stats("full_llm_dataset", df)

    # считаем три подвыборки по 10%
    subset_quality = stratified_topk_by_score(
        df, "quality_score", SUBSET_FRAC, seed=SEED
    )
    subset_diversity = stratified_topk_by_score(
        df, "diversity_score", SUBSET_FRAC, seed=SEED
    )
    subset_importance = stratified_topk_by_score(
        df, "importance_score", SUBSET_FRAC, seed=SEED
    )

    print_label_stats("subset_quality", subset_quality)
    print_label_stats("subset_diversity", subset_diversity)
    print_label_stats("subset_importance", subset_importance)

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # сохраняем сами подвыборки (на всякий случай)
    subset_quality.to_parquet(
        OUT_BASE_DIR / "subset_quality_p10.parquet", index=False
    )
    subset_diversity.to_parquet(
        OUT_BASE_DIR / "subset_diversity_p10.parquet", index=False
    )
    subset_importance.to_parquet(
        OUT_BASE_DIR / "subset_importance_p10.parquet", index=False
    )

    # обучаем три модели
    res = {}

    for name, subset in [
        ("llm_quality_p10", subset_quality),
        ("llm_diversity_p10", subset_diversity),
        ("llm_importance_p10", subset_importance),
    ]:
        model_dir, metrics = train_on_subset(name, subset, out_base_dir=OUT_BASE_DIR)
        res[name] = {
            "model_dir": model_dir,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "size": int(len(subset)),
        }

    # сводка по всем трём
    summary_path = OUT_BASE_DIR / "summary_llm_proxy_subsets.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n=== Summary over all LLM-proxy subsets ===")
    for name, info in res.items():
        print(f"{name}: size={info['size']}, metrics={info['metrics']}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
