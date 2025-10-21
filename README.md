# llm-data-selection-bert

Stage-first skeleton for studying **data subset selection** (quality/diversity/importance) on SST-2 with ModernBERT.

## Project Goal

Investigate how data subset selection affects small model accuracy when training on SST-2. 
Based on the taxonomy from [Data Selection for LLM Instruction Tuning](https://arxiv.org/pdf/2408.02085), which identifies three categories of selection methods:
- **Quality**: filtering noisy/incorrect examples
- **Diversity**: coverage of different areas of feature space
- **Importance**: selection of most influential examples for target task

## Stack
- **uv** for environment management
- **Hydra** for configuration (single `conf/config.yaml`)
- **Snakemake** for experiment orchestration (`workflow/Snakefile`)
- **Weights & Biases** (W&B) for experiment tracking (optional)
- **ModernBERT** as baseline model
- **SST-2** (GLUE) for sentiment classification

## Implemented Selectors

### Baseline
- `random` — random selection (baseline)
- `full` — entire dataset or first k examples

### Quality-based
- `perplexity` — select examples within perplexity range [p_low, p_high]
  - Filters too simple (low pppl) and too complex/noisy (high pppl) examples
  - Optionally ranks by text metrics (token_count, TTR, readability)

### Diversity-based
- `kcenter` — K-Center Greedy: iteratively selects maximally distant examples
- `kmeans` — embedding clustering + proportional sampling from clusters
- `herding` — greedily selects examples minimizing empirical mean deviation

## Planned Selectors

We aim to extend the selector zoo with:

- **cleanlab-based filtering** — confident learning to prune noisy labels.
- **LLM-as-a-judge flags** — lightweight ambiguity/sarcasm screens.
- **Consistency filters** — drop examples that flip labels under simple EDA perturbations.
- **Self-filtering classifier** — ModernBERT embeddings + tiny head to score quality.
- **Semantic × lexical diversity** — combine CLS embeddings with MATTR/MTLD.
- **Deterministic stratified coresets** — ensure balanced coverage across text attributes.
- **Predictive entropy / WordPiece ratio** — fast uncertainty and rarity proxies.
- **Pseudo-perplexity (MLM)** — encoder-style difficulty estimates.
- **GraNd / gradient norms** — importance scoring at initialization.
- **Mini-forgetting / streaming forgetting-lite** — retain samples that models forget.
- **Hybrid pipelines** — sequential quality→diversity→importance and weighted score fusion.

## Features

Module `src/features/` uses **modular architecture** with automatic caching via `joblib.Memory`:

### Configuration vs Orchestration
- `conf/config.yaml` — single Hydra configuration that defines paths, model/training params, and selection defaults. Every CLI call (`uv run python -m src.pipeline ...`) reads this file, so you tweak global settings there.
- `workflow/Snakefile` — Snakemake workflow that chains `prepare → select → train → eval` for multiple strategies/fractions. It does not replace the Hydra config; rather, it invokes the same pipeline entrypoint repeatedly with different overrides (`+selection.name=...`, `+subset.frac=...`).

### 1. **ModernBERT CLS Embeddings** (`ensure_modernbert_cls`)
   - Module: `src/features/embeddings.py`
   - Used for diversity/importance selectors
   - Automatic caching in `artifacts/features/`

### 2. **Prefix Perplexity (PPPL)** (`ensure_pppl`)
   - Module: `src/features/perplexity.py`
  - Uses **Qwen3-1.7B-Base** (1.7B parameters, base model) with `add_start_token=False` to accommodate missing BOS token
   - Low perplexity = high text quality
   - Automatic caching in `artifacts/features/`

### 3. **Text Metrics** (`ensure_quality_indicators`)
   - Module: `src/features/text_metrics.py`
   - Uses **textstat** library for metric computation
   - Metrics: `token_count`, `char_count`, `type_token_ratio`, `avg_word_len`, 
     `flesch_reading_ease`, `syllable_count`, `sentence_count`
   - Automatic caching in `artifacts/features/`

### Used Libraries
- **joblib.Memory** - smart disk caching for ML
- **textstat** - ready-made readability and text complexity metrics
- **transformers** - ModernBERT and Qwen3 for embeddings and perplexity

## Quick start
```bash
# 0) Install uv and sync dependencies (CPU by default)
pip install uv
uv sync --extra cpu

# (Optional) GPU stack – requires matching CUDA drivers
# uv sync --extra gpu

# 1) Prepare data (downloads SST-2 and saves to data/processed/)
uv run python -m src.pipeline prepare

# 2) Single run (10% random subset, seed=42)
uv run python -m src.pipeline select +selection.name=random +subset.frac=0.1 seed=42
uv run python -m src.pipeline train  +selection.name=random +subset.frac=0.1 seed=42
uv run python -m src.pipeline eval   +selection.name=random +subset.frac=0.1 seed=42
# add `track.wandb_mode=disabled` to any command if you want to skip W&B logging

# 3) Configure W&B (optional, defaults to online mode)
# echo WANDB_API_KEY=... >> .env
# (entity/project come from conf/config.yaml and can be overridden via CLI)

# 4) Full experiment sweep (strategies × {5%,10%,20%})
uv run snakemake -j 4 -s workflow/Snakefile   # adjust -j to available CPU cores
```

## Examples of running different selectors

```bash
# Perplexity-based (quality)
uv run python -m src.pipeline select +selection.name=perplexity +selection.params.p_low=0.3 +selection.params.p_high=0.7 +subset.frac=0.1 seed=42

# K-Center (diversity)
uv run python -m src.pipeline select +selection.name=kcenter +subset.frac=0.1 seed=42

# K-Means (diversity)
uv run python -m src.pipeline select +selection.name=kmeans +selection.params.n_clusters=128 +subset.frac=0.1 seed=42

# Herding (diversity)
uv run python -m src.pipeline select +selection.name=herding +subset.frac=0.1 seed=42
```

## Repository layout
```
llm-data-selection-bert/
├── conf/
│   └── config.yaml          # Single Hydra config (paths, model, train, eval, selection)
├── workflow/
│   └── Snakefile            # Orchestration: prepare → select → train → eval
├── src/
│   ├── pipeline.py          # Single entry point (Hydra)
│   ├── metrics.py           # Metrics: Accuracy, F1, Precision, Recall
│   ├── utils.py             # Utilities: seeding, time_block
│   ├── features/            # Modular feature system
│   │   ├── __init__.py      # Public API with caching
│   │   ├── cache.py         # Wrapper over joblib.Memory
│   │   ├── embeddings.py    # ModernBERT CLS embeddings
│   │   ├── perplexity.py    # Perplexity via Qwen3-1.7B-Base
│   │   └── text_metrics.py  # Text metrics via textstat
│   ├── selectors/           # Data selection strategies
│   │   ├── __init__.py      # REGISTRY of all selectors
│   │   ├── base.py          # BaseSelector, RandomSel, FullSel
│   │   ├── quality.py       # PerplexitySel
│   │   └── diversity.py     # KCenterSel, KMeansSel, HerdingSel
│   └── stages/              # Pipeline stages
│       ├── prepare.py       # SST-2 loading
│       ├── select.py        # Subset selection
│       ├── train.py         # ModernBERT training
│       └── eval.py          # Validation evaluation
├── data/
│   ├── processed/           # Processed datasets
│   └── manifests/           # CSV with selected examples
├── artifacts/
│   ├── features/            # Feature cache (embeddings, pppl, quality)
│   └── runs/                # Model checkpoints and metrics
└── reports/                 # Tables and result figures
```

## Env
Copy `example.env` to `.env` and fill in `WANDB_API_KEY` if you plan to log runs to W&B. Other W&B fields (`entity`, `project`, `wandb_mode`) are managed via `conf/config.yaml` or Hydra overrides.
