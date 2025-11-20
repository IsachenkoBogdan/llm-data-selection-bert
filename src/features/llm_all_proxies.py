# src/features/gemini_proxies_unique.py

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Please install google-generativeai:\n"
        "  pip install google-generativeai\n"
    )

try:
    from rapidfuzz.distance import Levenshtein as RFLevenshtein
except ImportError:
    raise ImportError(
        "Please install rapidfuzz:\n"
        "  pip install rapidfuzz\n"
    )


# ================== Настройки ==================

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro")
API_KEY_ENV = "GEMINI_API_KEY"

# доля самых уникальных в каждом классе
UNIQUE_FRAC = 0.10  # 10%

# сколько случайных соседей сравнивать для оценки уникальности
N_RANDOM_NEIGHBORS = 200

OUT_PARQUET = Path("artifacts/features/sst2_gemini_proxies_unique10.parquet")
OUT_CSV = Path("artifacts/features/sst2_gemini_proxies_unique10.csv")


# ================== Промпт ==================

PROMPT_TEMPLATE = """
You are an expert annotator for sentiment analysis datasets.

You are given a short movie review from a dataset similar to SST-2.

Text:
<<<{text}>>>

You must evaluate this review along THREE separate dimensions:

------------------------------------------------------------
1) QUALITY of the text (q_quality)

Focus on:
  - grammar and spelling (is it readable?),
  - fluency and naturalness of language,
  - presence or absence of noise (HTML tags, random tokens, URLs, broken text),
  - clarity of sentiment (is it clear whether the review is positive or negative?).

Use the following 3-point scale:

  0 = low-quality / noisy:
        * many errors or broken text,
        * strong noise (random tokens, garbage, machine translation artifacts),
        * sentiment is very unclear or almost impossible to determine.

  1 = medium-quality:
        * mostly readable but with noticeable issues (awkward phrasing, some noise),
        * sentiment can be determined but is not completely clear or is weakly expressed,
        * overall acceptable but not ideal.

  2 = high-quality:
        * fluent, natural language, few or no errors,
        * no obvious noise or garbage,
        * sentiment (positive or negative) is clearly and strongly expressed.

------------------------------------------------------------
2) LOCAL SPECIFICITY / NON-GENERICNESS (q_diversity_local)

Here we measure how NON-GENERIC and SPECIFIC the review is, i.e. how much it differs
from a bland one-line opinion like "This movie was great" or "This movie was bad".

Focus on:
  - presence of concrete details (plot, acting, visuals, specific reasons),
  - originality of wording and phrasing,
  - whether the text expresses a unique point of view rather than a generic template.

Use the following 3-point scale:

  0 = very generic:
        * mostly boilerplate phrases,
        * no real details, no specific reasons or examples,
        * could easily fit any random movie.

  1 = moderately specific:
        * some concrete details or reasons are mentioned,
        * still somewhat generic but gives more information than a one-line cliché.

  2 = highly specific:
        * clear, concrete details about the movie,
        * distinctive or personal wording,
        * clearly tied to this particular movie.

Do NOT evaluate sentiment polarity here. Only specificity vs genericness.

------------------------------------------------------------
3) IMPORTANCE for training a sentiment classifier (q_importance)

We want to train a model that must reliably predict whether a review is POSITIVE or NEGATIVE.

Evaluate how USEFUL this review would be as a TRAINING EXAMPLE for such a model.

Consider:
  - clarity and strength of the sentiment signal (is the polarity obvious?),
  - how well the wording illustrates typical language of that sentiment,
  - whether the example is free of confusing noise that might mislead a model.

Use the following 3-point scale:

  0 = low importance for training:
        * sentiment is very weak, mixed, or unclear,
        * text is too short, vague, or noisy,
        * unlikely to help the model learn a robust decision boundary.

  1 = medium importance:
        * sentiment can be determined but not very strong,
        * text is acceptable but not an ideal textbook example,
        * still reasonably helpful as part of a larger training set.

  2 = high importance:
        * sentiment (positive or negative) is strong and unambiguous,
        * text is clean and representative of what we want the model to handle,
        * would be a very good example in a training curriculum.

------------------------------------------------------------

OUTPUT FORMAT (IMPORTANT):

Return a SINGLE valid JSON object with the following structure:

{
  "quality": {
    "score": 0 or 1 or 2,
    "explanation": "<short one-sentence explanation>"
  },
  "diversity": {
    "score": 0 or 1 or 2,
    "explanation": "<short one-sentence explanation>"
  },
  "importance": {
    "score": 0 or 1 or 2,
    "explanation": "<short one-sentence explanation>"
  }
}

- Do NOT add any extra keys.
- Do NOT add any text before or after the JSON.
- Make sure "score" fields are integers 0, 1, or 2.
"""


# ================== Gemini ==================

def _init_gemini_model():
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. "
            f"Export your Gemini API key, e.g.:\n  export {API_KEY_ENV}=your_key_here"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def call_gemini_for_text(
    model,
    text: str,
    max_retries: int = 3,
    sleep_base: float = 2.0,
) -> Optional[Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", None)
            if raw is None:
                if hasattr(resp, "candidates") and resp.candidates:
                    parts = resp.candidates[0].content.parts
                    raw = "".join(getattr(p, "text", "") for p in parts)
            if not raw:
                raise ValueError("Empty response from Gemini")

            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in response: {raw[:200]}")
            json_str = raw[start : end + 1]

            data = json.loads(json_str)

            for key in ("quality", "diversity", "importance"):
                if key not in data:
                    raise ValueError(f"Missing key '{key}' in JSON: {data}")
                if "score" not in data[key]:
                    raise ValueError(f"Missing score in '{key}' section: {data}")
            return data

        except Exception as e:
            print(f"[Gemini] Error on attempt {attempt + 1}: {e}")
            if attempt + 1 < max_retries:
                time.sleep(sleep_base * (attempt + 1))
            else:
                print("[Gemini] Giving up on this example.")
                return None


# ================== Оценка «уникальности» по Левенштейну ==================

def estimate_uniqueness_levenshtein(
    sentences: List[str],
    n_neighbors: int = N_RANDOM_NEIGHBORS,
    seed: int = 42,
) -> np.ndarray:
    """
    Приблизительно оценивает уникальность каждого предложения в классе.

    Для каждого i:
      - выбираем до n_neighbors случайных j (j != i) внутри класса,
      - считаем расстояние Левенштейна,
      - берём min_j distance(i, j) как "минимальную близость" к другим.
    Чем min_dist больше, тем предложение более уникально.

    Используется rapidfuzz.distance.Levenshtein.
    """
    n = len(sentences)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    rng = np.random.default_rng(seed)
    min_dist = np.full(n, np.inf, dtype=np.float32)

    for i in range(n):
        k = min(n_neighbors, n - 1)
        if k <= 0:
            min_dist[i] = 0.0
            continue

        rand_idx = rng.integers(0, n, size=k)
        rand_idx = rand_idx[rand_idx != i]
        if rand_idx.size == 0:
            min_dist[i] = 0.0
            continue

        s_i = sentences[i]
        best = np.inf
        for j in rand_idx:
            d = RFLevenshtein.distance(s_i, sentences[int(j)])
            if d < best:
                best = d

        if np.isinf(best):
            best = 0.0
        min_dist[i] = best

        if (i + 1) % 1000 == 0:
            print(f"  Levenshtein (rapidfuzz): processed {i + 1}/{n} in this class")

    return min_dist


# ================== Основной пайплайн ==================

def select_unique_sst2_train(frac: float = UNIQUE_FRAC) -> pd.DataFrame:
    """
    Загружает train SST-2, считает "уникальность" по Левенштейну отдельно для label=0 и label=1,
    выбирает top-frac по уникальности в каждом классе.

    Возвращает DataFrame с колонками:
      - global_idx: индекс в исходном train DataFrame (позиционный),
      - sentence,
      - label,
      - min_lev_dist,
      - split="train".
    """
    print("Loading SST-2 train split...")
    ds = load_dataset("glue", "sst2")
    train = ds["train"]
    df_train = train.to_pandas()

    if "label" not in df_train.columns:
        raise ValueError("Expected 'label' column in SST-2 train dataset.")

    # сохраним изначальный индекс, чтобы не потерять связь
    df_train = df_train.reset_index(drop=False).rename(columns={"index": "global_idx"})

    rows = []
    for label_val in sorted(df_train["label"].unique()):
        df_c = df_train[df_train["label"] == label_val].copy()
        sentences = df_c["sentence"].astype(str).tolist()
        print(f"\nEstimating Levenshtein-uniqueness for label={label_val}, n={len(df_c)}")

        uniq_scores = estimate_uniqueness_levenshtein(sentences)
        df_c["min_lev_dist"] = uniq_scores

        n_c = len(df_c)
        k_c = max(1, int(round(frac * n_c)))
        df_c = df_c.sort_values("min_lev_dist", ascending=False).head(k_c)

        print(f"Selected {len(df_c)}/{n_c} examples for label={label_val}")
        rows.append(df_c)

    df_sel = pd.concat(rows, axis=0).reset_index(drop=True)
    df_sel["split"] = "train"
    return df_sel


def annotate_with_gemini(df: pd.DataFrame) -> pd.DataFrame:
    """
    Прогоняет Gemini по всем строкам df["sentence"] и добавляет:
      - quality_score / _expl
      - diversity_score / _expl
      - importance_score / _expl
    """
    print("Initializing Gemini...")
    model = _init_gemini_model()

    quality_scores, quality_expl = [], []
    div_scores, div_expl = [], []
    imp_scores, imp_expl = [], []

    for i, row in df.iterrows():
        text = str(row["sentence"])
        ann = call_gemini_for_text(model, text)

        if ann is None:
            q_s = d_s = im_s = None
            q_e = d_e = im_e = ""
        else:
            q_s = ann.get("quality", {}).get("score")
            q_e = ann.get("quality", {}).get("explanation", "")

            d_s = ann.get("diversity", {}).get("score")
            d_e = ann.get("diversity", {}).get("explanation", "")

            im_s = ann.get("importance", {}).get("score")
            im_e = ann.get("importance", {}).get("explanation", "")

        quality_scores.append(q_s)
        quality_expl.append(q_e)
        div_scores.append(d_s)
        div_expl.append(d_e)
        imp_scores.append(im_s)
        imp_expl.append(im_e)

        if (i + 1) % 50 == 0:
            print(f"  Gemini annotated {i + 1}/{len(df)}")

    df = df.copy()
    df["quality_score"] = quality_scores
    df["quality_expl"] = quality_expl
    df["diversity_score"] = div_scores
    df["diversity_expl"] = div_expl
    df["importance_score"] = imp_scores
    df["importance_expl"] = imp_expl

    return df


def main():
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selecting {int(UNIQUE_FRAC * 100)}% most unique examples per class (train split)...")
    df_unique = select_unique_sst2_train(frac=UNIQUE_FRAC)
    print(f"Total selected examples: {len(df_unique)}")

    print("Annotating selected examples with Gemini...")
    df_annot = annotate_with_gemini(df_unique)

    df_annot.to_parquet(OUT_PARQUET, index=False)
    df_annot.to_csv(OUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved unique-10%-per-class Gemini annotations to:\n  {OUT_PARQUET}\n  {OUT_CSV}")


if __name__ == "__main__":
    main()
