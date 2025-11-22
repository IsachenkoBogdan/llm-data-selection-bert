# src/analysis/visualize_selectors_and_results.py

from __future__ import annotations

import os
from pathlib import Path

# backend до любых импортов matplotlib
os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from ..features.embeddings import compute_modernbert_embeddings  # noqa: E402
from ..selectors import REGISTRY  # noqa: E402


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "axes.grid": True,
        "grid.color": "#555555",
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "legend.frameon": False,
        "savefig.transparent": True,
    }
)


# ====================== UMAP по селекторам ======================


def compute_umap_embeddings(
    cfg_path: str = "conf/config.yaml",
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = OmegaConf.load(cfg_path)

    data_path = Path("data/processed/sst2.parquet")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} не найден. Сначала запусти prepare: "
            "uv run python -m src.pipeline stage=prepare"
        )

    df = pd.read_parquet(data_path)

    # берём настройки модели из конфига, если они там есть
    if "model" in cfg and "name" in cfg.model:
        model_name = cfg.model.name
    else:
        model_name = "answerdotai/ModernBERT-base"

    if "data" in cfg and "max_length" in cfg.data:
        max_length = int(cfg.data.max_length)
    else:
        max_length = 128

    # секции analysis может не быть — берём дефолт 256
    analysis_cfg = cfg.get("analysis", {})
    if isinstance(analysis_cfg, dict):
        batch_size = int(analysis_cfg.get("umap_batch_size", 256))
    else:
        batch_size = int(getattr(analysis_cfg, "umap_batch_size", 256))

    emb = compute_modernbert_embeddings(
        df["sentence"],
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    emb_df = pd.DataFrame(
        emb,
        index=df.index,
        columns=[f"emb_{i}" for i in range(emb.shape[1])],
    )

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    X = emb_df[emb_cols].to_numpy()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=int(getattr(cfg, "seed", 42)),
    )
    X2d = reducer.fit_transform(X)

    df_umap = df.copy()
    df_umap["umap_x"] = X2d[:, 0]
    df_umap["umap_y"] = X2d[:, 1]

    return df_umap, X2d


def plot_selectors_umap(
    df_umap: pd.DataFrame,
    selector_names: list[str],
    subset_frac: float = 0.1,
    seed: int = 42,
    cfg_path: str = "conf/config.yaml",
    out_path: str = "analysis/reports/sst2_selectors_umap.png",
) -> None:
    cfg = OmegaConf.load(cfg_path)
    cfg.seed = seed
    cfg.subset = getattr(cfg, "subset", OmegaConf.create())
    cfg.subset.frac = subset_frac

    n_sel = len(selector_names)
    n_cols = min(4, n_sel)
    n_rows = int(np.ceil(n_sel / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        facecolor="none",
    )
    if n_sel == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    base_alpha = 0.08
    base_size = 6
    sel_size = 8

    x_all = df_umap["umap_x"].to_numpy()
    y_all = df_umap["umap_y"].to_numpy()
    n_total = len(df_umap)
    k = int(round(subset_frac * n_total))

    colors = [
        "#FFB000",
        "#FF6F61",
        "#6CCFF6",
        "#A3A0FB",
        "#7BD389",
        "#F4B5FF",
        "#FFED66",
        "#FF9DA7",
    ]

    for i, name in enumerate(selector_names):
        ax = axes[i]

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.scatter(
            x_all,
            y_all,
            s=base_size,
            c="#FFFFFF",
            alpha=base_alpha,
            linewidths=0,
        )

        if name not in REGISTRY:
            ax.set_title(f"{name} (NOT FOUND)", fontsize=11, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        selector_cls = REGISTRY[name]
        selector = selector_cls()
        selector.fit(df_umap, cfg=cfg)
        df_sel = selector.select(df_umap, k=k)
        mask = df_umap.index.isin(df_sel.index)

        ax.scatter(
            x_all[mask],
            y_all[mask],
            s=sel_size,
            c=colors[i % len(colors)],
            alpha=0.9,
            linewidths=0,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"SST-2, UMAP CLS-эмбеддинги\nподсветка выбранных точек (frac={subset_frac:.2f})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)


def make_results_df() -> pd.DataFrame:
    data = {
        "method": [
            "Full",
            "Random",
            "Word-piece ratio",
            "Perplexity",
            "LLM-based classifier",
            "K-center",
            "Herding",
            "EL2N",
            "Hybrid (Q + D)",
        ],
        "accuracy": [0.809, 0.747, 0.783, 0.749, 0.781, 0.774, 0.763, 0.642, 0.792],
        "f1": [0.812, 0.746, 0.787, 0.742, 0.778, 0.768, 0.768, 0.596, 0.789],
        "train_time_sec": [1280, 48, 48, 48, 48, 48, 48, 48, 48],
        "select_time_sec": [0, 0, 13, 4700, 4787, 560, 120, 20, 4920],
    }
    return pd.DataFrame(data)


def plot_accuracy_f1(
    df: pd.DataFrame,
    out_path: str = "analysis/reports/results_accuracy_f1.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", alpha=0.85)
    ax.bar(x + width / 2, df["f1"], width, label="F1", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    ax.set_ylim(0.6, 0.83)
    ax.set_ylabel("Score")
    ax.set_title("Качество моделей на 10% данных (Accuracy / F1)")
    ax.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)


def plot_compute_times(
    df: pd.DataFrame,
    out_path: str = "analysis/reports/results_compute_times.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")

    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width / 2,
        df["train_time_sec"],
        width,
        label="Train time, s",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        df["select_time_sec"],
        width,
        label="Selection time, s",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Время (сек), лог. шкала")
    ax.set_title("Затраты по времени: обучение vs отбор подмножества")
    ax.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)


def main() -> None:
    df_umap, _ = compute_umap_embeddings()

    selector_names = [
        "random",
        "wordpiece_ratio",
        "perplexity",
        "llm_quality",
        "kcenter",
        "herding",
        "datadiet",
        "hybrid_qdi",
    ]
    plot_selectors_umap(
        df_umap,
        selector_names=selector_names,
        subset_frac=0.1,
        seed=42,
    )

    df_res = make_results_df()
    plot_accuracy_f1(df_res)
    plot_compute_times(df_res)


if __name__ == "__main__":
    main()
