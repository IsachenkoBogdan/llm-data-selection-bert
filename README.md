# llm-data-selection-bert

Stage-first framework for studying **data subset selection** (quality / diversity / importance) on GLUE SST-2 with ModernBERT.

Проект создан на основе исследование [Data Selection for LLM Instruction Tuning](https://arxiv.org/pdf/2408.02085), в котором авторы выделяют три ключевых аспекта:

- **Quality** — насколько примеры чистые, грамматичные, с однозначной разметкой.
- **Diversity** — насколько поднабор покрывает разные области feature-space.
- **Importance** — насколько пример влиятелен для обучения целевой модели.

Задача: **натренировать ModernBERT (заменили на bert-tiny) на 10% данных SST-2** и понять,  
какие стратегии отбора подмножеств дают лучшее качество при фиксированном бюджете данных и compute.

---

## 1. Stack

- **uv** – управление окружением и зависимостями.
- **Hydra** – конфигурация (единый файл `conf/config.yaml` + CLI overrides).
- **Snakemake** – оркестрация экспериментов (`workflow/Snakefile`: `prepare → select → train → eval`).
- **Weights & Biases (W&B)** – трекинг экспериментов (по умолчанию включён, можно отключить).
- **Transformers / ModernBERT** – модель-классификатор.
- **Qwen/Qwen3-1.7B-Base** – для псевдо-перплексии (quality proxy).
- **jiwer / sklearn / numpy / pandas / matplotlib** – метрики и аналitika.

---

## 2. Данные и модель

- **Модель:** `answerdotai/ModernBERT-base` (binary классификация, SST-2).
- **Датасет:** GLUE **SST-2** (sentiment, 2 класса).
- **Что измеряем:**
  - `accuracy` / `f1` на dev (GLUE validation split),
  - скорость обучения,
  - время селекции,
  - поведение разных прокси (perplexity, EL2N, entropy, etc.) на тех же данных.

---

## 3. Реализованные селекторы

Все селекторы зарегистрированы в `src/selectors/__init__.py` через `REGISTRY`
и вызываются по имени (`cfg.selection.name` / `selection.name=...`).

### 3.1. Базовые

- **`random`** (`RandomSel` in `base.py`)  
  Случайный выбор `k` примеров из train. Золотой baseline.

- **`full`** (`FullSel` in `base.py`)  
  Либо весь датасет, либо первые `k` примеров (для sanity-check’ов).

---

### 3.2. Diversity-based (эмбеддинги ModernBERT)

Все эти методы используют CLS-эмбеддинги ModernBERT:
`ensure_modernbert_cls(df, cfg)` → `embeddings.npy` (+ cache в `artifacts/features/`).

- **`kcenter`** (`KCenterSel`)  
  K-center greedy в embedding space:
  1. Стартуем с случайной точки.
  2. На каждом шаге берём пример, **максимально удалённый** от уже выбранных.
  3. Жёстко максимизируем coverage feature-space.

- **`kmeans`** (`KMeansSel`)  
  1. Кластеризация ModernBERT-эмбеддингов на `n_clusters`.
  2. Сэмплинг пропорционально размеру кластера (или по 1–2 из каждого).
  3. Даёт более «структурированный» coverage, чем k-center.

- **`herding`** (`HerdingSel`)  
  Классический herding из coreset-литературы:
  1. Считаем эмпирическое среднее embedding’ов `μ`.
  2. Итерируемся: выбираем пример, который максимальнее уменьшает расстояние между средним выбранных и `μ`.
  3. Восстанавливаем «центроид» датасета малым подмножеством.

---

### 3.3. Quality / rarity / uncertainty

Эти методы работают на уровне **текста / токенов** и **логитов модели**.

#### 3.3.1. Perplexity-based (`perplexity`)

- Реализация: `PerplexitySel` (`quality.py`),  
  фича: `ensure_pppl(df, cfg)` в `src/features/perplexity.py`.
- Используем **Qwen3-1.7B-Base** как MLM-подобный scorer:
  - считаем **pseudo-perplexity (PPPL)** каждого предложения.
  - низкая PPPL → текст простой и хорошо моделируемый,
  - очень высокая PPPL → шум/эрратика/неестественные тексты.
- Стратегии селекции:
  - отбор по интервалу `[p_low, p_high]` (выкидываем тривиальные и откровенный мусор),
  - либо top-k самых «адекватных» по PPPL.

#### 3.3.2. WordPiece rarity (`wordpiece_ratio`)

- Реализация: `WordPieceRatioSel` (`statistical.py`) +  
  фича: `ensure_wordpiece_stats(df, cfg)` (через ModernBERT tokenization).
- Идея:
  - rare-токены = потенциально более информативные / «edge cases»,
  - считаем отношения наподобие:
    - число subword’ов / число слов,
    - частота редких subword’ов по корпусу.
- Селекция:
  - можно брать top-k по доле rare subword’ов,
  - или избегать совсем уж «сломанного» текста (слишком большая фрагментация).

#### 3.3.3. Predictive entropy (`entropy`)

- Реализация: `PredictiveEntropySel` (`statistical.py`) +  
  фича: `ensure_predictive_entropy(df, cfg)` в `src/features/predictive_stats.py`.
- Pipeline:
  1. Берём ModernBERT (либо уже натренированный, либо аппроксимацию).
  2. Считаем logits для всех train-примеров.
  3. Из них — **предиктивную энтропию**:
     \[
     H(p) = -\sum_c p_c \log p_c
     \]
- Селекция:
  - top-k по **высокой энтропии** → «трудные / неуверенные» примеры (uncertainty sampling proxy),
  - либо наоборот filtering слишком простых.

---

### 3.4. Importance-based: BERT on a Data Diet (`datadiet`)

- Реализация: `DataDietSel` (`datadiet.py`)  
  + фича: `ensure_el2n_scores(df, cfg)` в `src/features/data_diet.py`.
- Идея из [*Deep Learning on a Data Diet: Finding Important Examples Early in Training*, 2022](https://arxiv.org/pdf/2107.07075):

  1. Запускаем модель на ранних шагах обучения и считаем для каждого примера:
     \[
     \text{EL2N}(x_i) = \mathbb{E}_t \left\| p_t(y|x_i) - y_{\text{one-hot}} \right\|_2
     \]
     — средняя L2-норма ошибки предсказания vs true label по ранним эпохам.
  2. Примеры с **большим EL2N** — систематически трудные → больше влияют на градиенты.
- В нашем контуре:
  - считаем EL2N для SST-2 (в отдельном helper),
  - селектор `DataDietSel` берёт **top-k по EL2N** (самые «важные» для обучения).

---

### 3.5. LLM-as-a-judge: clarity / quality (`llm_clarity` / `llm_quality`)

- Реализация: `LLMQualitySel` (`llm_quality.py`)  
  + фича: `ensure_llm_clarity_scores(df, cfg)` в `src/features/llm_clarity.py`.
- Используем внешний LLM (например, Qwen / Gemini) как «судью»:
  - задаём прокси-задачу: оценить **однозначность тональности**, **чистоту текста**, отсутствие артефактов и т.д.
  - LLM предсказывает класс `clarity ∈ {0,1,2}` и даёт распределение `P(c|x)`.
- Селекция:
  - **стратифицировано по классам LLM**: берём примерно `k/3` из каждого clarity-класса,
  - внутри каждого класса — **по уверенности** `P(clarity = c)` (top confident).
- Итог: балансируем между «идеальными», «нормальными» и «сомнительными» примерами вместо тупого фильтра.

---

### 3.6. LLM proxies for Q / D / I (Gemini scripts)

Не жёстко зашито в Snakemake, но есть скрипты для:

- выборки **наиболее уникальных** предложений по Levenshtein/rapidfuzz (10% per label),
- прогонки через **Gemini** с тремя промптами:
  - Quality proxy (чистота / грамматика / однозначность),
  - Diversity proxy (насколько пример отличается от типичных примеров класса),
  - Importance proxy (насколько важен для обучения эмоц. классификатора).
- получаем scores и сохраняем в отдельный parquet/JSONL для анализа и возможной селекции.

Это даёт отдельный взгляд на Q/D/I через «oracle» LLM, независимый от ModernBERT.

---

### 3.7. Hybrid QDI (`hybrid_qdi`)

- Реализация: `HybridQDISel` (`hybrid.py`).
- Использует сразу несколько фич:

  - **Quality**: pppl, text metrics, LLM clarity,
  - **Diversity**: расстояния в ModernBERT-эмбеддингах (k-center-style coverage),
  - **Importance**: EL2N / предиктивная энтропия.

- Общая логика:
  1. Нормируем каждую компоненту к [0,1].
  2. Строим общий скор:
     \[
     s_i = \alpha \cdot Q_i + \beta \cdot D_i + \gamma \cdot I_i
     \]
  3. Берём top-k по `s_i`, плюс можем добавить лёгкую стратификацию по классам.

- Этот метод по сути реализует тезис статьи 2408.02085:  
  **нужно объединять качество, разнообразие и важность**, а не крутить их по штучке.

---

## 4. Features / proxies

Все фичи живут в `src/features/` и имеют вид `ensure_*`, чтобы:

- считать один раз,
- складывать на диск (parquet / npy),
- прозрачно переиспользовать между селекторами.

Основные:

- `ensure_modernbert_cls(df, cfg)` → CLS-эмбеддинги (diversity, hybrid).
- `ensure_pppl(df, cfg)` → pseudo-perplexity через Qwen3-1.7B (quality).
- `ensure_quality_indicators(df, cfg)` → textstat-подобные метрики (длина, TTR, readability).
- `ensure_wordpiece_stats(df, cfg)` → word-piece rarity (quality/rarity).
- `ensure_predictive_entropy(df, cfg)` → энтропия предсказаний ModernBERT (uncertainty).
- `ensure_el2n_scores(df, cfg)` → EL2N-скоры (importance / data diet).
- `ensure_llm_clarity_scores(df, cfg)` → LLM-based clarity-классы (LLM-as-a-judge).
- Gemini-прокси Q/D/I → отдельный скрипт, который пишет расширенный датасет.

---

## 5. Pipeline: stages и Snakemake

Все стадии реализованы в `src/stages/` и вызываются через единый entrypoint  
`src/pipeline.py` (Hydra).

Стадии:

- `prepare` — загрузка SST-2 (HF datasets) и сохранение в `data/processed/sst2.parquet`.
- `select` — применение выбранного селектора к `df_train`, сохранение манифеста:
  - `data/manifests/{selection.name}_pXX_seedYY.csv`.
- `train` — обучение ModernBERT на выбранном манифесте:
  - чекпоинт и tokenizer сохраняются в `artifacts/{selection.name}/pXX/seedYY/model`.
  - логируются train metrics и время.
- `eval` — оценка модели на GLUE validation:
  - пишем `artifacts/{selection.name}/pXX/seedYY/metrics.json` (accuracy, f1, …),
  - логируем в W&B.

### Snakemake workflow

`workflow/Snakefile` описывает sweep:

- список стратегий `STRATS` (можно дописывать свои,
  напр. `datadiet`, `entropy`, `wordpiece_ratio`, `hybrid_qdi`, `llm_clarity`, …),
- поднаборы `SUBS = {p05, p10, p20}`,
- фиксированный seed (по умолчанию 42).

Rule chain:

```text
prepare → select → train → eval
````

Так можно одним вызовом прогнать все стратегии × доли данных.

---

## 6. Quick start

```bash
# 0) Install uv and sync deps (CPU)
pip install uv
uv sync --extra cpu

# GPU stack (requires CUDA 12.4-compatible drivers)
# uv sync --extra gpu
```

### 6.1. Подготовка данных

```bash
uv run python -m src.pipeline stage=prepare
# создаст data/processed/sst2.parquet
```

### 6.2. Один прогон (например, random 10%, seed=42)

```bash
# select
uv run python -m src.pipeline stage=select selection.name=random subset.frac=0.10 seed=42

# train
uv run python -m src.pipeline stage=train  selection.name=random subset.frac=0.10 seed=42

# eval
uv run python -m src.pipeline stage=eval   selection.name=random subset.frac=0.10 seed=42
```

Отключить W&B:

```bash
uv run python -m src.pipeline stage=train selection.name=random subset.frac=0.10 seed=42 track.wandb_mode=disabled
```

### 6.3. Примеры запусков разных селекторов

```bash
# Perplexity-based quality
uv run python -m src.pipeline stage=select \
  selection.name=perplexity subset.frac=0.10 seed=42

# WordPiece rarity
uv run python -m src.pipeline stage=select \
  selection.name=wordpiece_ratio subset.frac=0.10 seed=42

# Predictive entropy (uncertainty)
uv run python -m src.pipeline stage=select \
  selection.name=entropy subset.frac=0.10 seed=42

# Data Diet (EL2N)
uv run python -m src.pipeline stage=select \
  selection.name=datadiet subset.frac=0.10 seed=42

# LLM clarity (LLM-as-a-judge)
uv run python -m src.pipeline stage=select \
  selection.name=llm_clarity subset.frac=0.10 seed=42

# Hybrid QDI (quality + diversity + importance)
uv run python -m src.pipeline stage=select \
  selection.name=hybrid_qdi subset.frac=0.10 seed=42
```

### 6.4. Полный sweep через Snakemake

```bash
uv run snakemake -j 4 -s workflow/Snakefile
# -j = число параллельных задач
```

Перед этим отредактируйте `workflow/Snakefile`, добавив свои стратегии в `STRATS`.

---

## 7. Repo layout

```text
## Repository layout

```text
llm-data-selection-bert/
├── artifacts/
│   ├── features/           # кэш фич (эмбеддинги, pppl, EL2N, LLM-прокси и т.д.)
│   └── runs/               # чекпоинты моделей, train/eval-метрики
├── conf/
│   └── config.yaml         # единый Hydra-конфиг (модель, обучение, выбор подмножеств)
├── data/
│   ├── manifests/          # CSV-манифесты отобранных подмножеств
│   ├── processed/          # подготовленные датасеты (sst2.parquet и т.п.)
│   └── raw/                # сырые данные, если будут нужны
├── notebooks/              # jupyter/colab-ноутбуки для экспериментов
├── analysis/
│   ├── reports/
│   │   └── sst2_analysis/  # сохранённые графики и отчёты по EDA
│   ├── eda.py              # скрипт анализа SST-2 (гистограммы, боксплоты и т.п.)
│   └── train_poxy_llm.py   # код для прокси-разметки/обучения маленьких LLM-классификаторов
├── src/
│   ├── features/           # вычисление и кэширование фич для селекторов
│   │   ├── __init__.py
│   │   ├── cache.py        # простой wrapper над joblib / файловым кэшем
│   │   ├── data_diet.py    # EL2N / BERT on a Data Diet
│   │   ├── embeddings.py   # ModernBERT CLS-эмбеддинги
│   │   ├── llm_all_proxies.py  # разметка LLM по quality/diversity/importance
│   │   ├── llm_clarity.py  # LLM-скоринг по однозначности тональности
│   │   ├── perplexity.py   # псевдо-perplexity (MLM) / PPPL
│   │   ├── predictive_stats.py  # predictive entropy, word-piece ratio и др.
│   │   └── text_metrics.py # длины, TTR, readability и прочие текстовые метрики
│   ├── selectors/          # стратегии отбора подмножеств
│   │   ├── __init__.py     # REGISTRY всех селекторов
│   │   ├── base.py         # BaseSelector, RandomSel, FullSel
│   │   ├── datadiet.py     # DataDietSel (EL2N / importance)
│   │   ├── diversity.py    # k-center, k-means, herding и др. diversity-методы
│   │   ├── hybrid.py       # гибридный Q×D×I селектор (hybrid_qdi)
│   │   ├── llm_quality.py  # LLMQualitySel, LLM-based clarity / quality
│   │   ├── quality.py      # quality-методы на основе perplexity / текст-метрик
│   │   └── statistical.py  # predictive entropy, word-piece ratio и т.п.
│   ├── stages/             # отдельные стадии пайплайна
│   │   ├── prepare.py      # загрузка/подготовка SST-2 → parquet
│   │   ├── select.py       # запуск селекторов, генерация manifests/*.csv
│   │   ├── train.py        # обучение ModernBERT на выбранном подмножестве
│   │   └── eval.py         # оценка на dev/test, логирование в W&B
│   ├── metrics.py          # accuracy, F1, precision, recall и вспомогательные функции
│   ├── pipeline.py         # единая Hydra-точка входа (stage=prepare/select/train/eval)
│   └── utils.py            # таймер, установка сидов и прочие утилиты
├── workflow/
│   └── Snakefile           # Snakemake-оркестрация: гриды по методам и долям данных
├── .gitignore
├── README.md
├── example.env             # пример настроек W&B и окружения
├── pyproject.toml          # зависимости (uv + extras cpu/gpu)
└── uv.lock                 # lock-файл uv
```

---

## 8. W&B

* API-ключ — через `.env`:

```bash
echo WANDB_API_KEY=... >> .env
```

* Остальное (project, entity, mode) задаётся в `conf/config.yaml`:

  * `track.project`, `track.entity`, `track.wandb_mode` (`online | offline | disabled`).

При запуске через Snakemake всё это подтягивается автоматически, каждая комбинация
`(selector, frac, seed)` получает отдельный run-идентификатор и логирует:

* train-loss, eval-loss,
* accuracy, f1, precision, recall
* время обучения,
* время селекции,
* размер поднабора и базовую статистику (class balance, avg length).

## 9. Результаты

| Метод                 | Accuracy |   F1   | Train compute time, sec | Selection compute time, sec |
|-----------------------|:--------:|:------:|:------------------------:|:----------------------------:|
| Full                  |  0.809   | 0.812  |          1280            |             0                |
| Random                |  0.747   | 0.746  |           48             |             0                |
| *Word-piece ratio*    |  0.783   | 0.787  |           48             |            13                |
| Perplexity            |  0.749   | 0.742  |           48             |           4700               |
| LLM-based classifier  |  0.781   | 0.778  |           48             |           4787               |
| K-center              |  0.774   | 0.768  |           48             |            560               |
| Herding               |  0.763   | 0.768  |           48             |            120               |
| EL2N                  |  0.642   | 0.596  |           48             |             20               |
| **Гибрид (Q + D)**    | **0.792**| **0.789** |        48             |          **4920**            |


