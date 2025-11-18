from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    import textstat

    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

# ==================== Глобальный стиль под чёрный слайд ====================

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "figure.facecolor": "none",      # прозрачный фон
        "axes.facecolor": "none",        # оси тоже прозрачные
        "savefig.facecolor": "none",
        "savefig.transparent": True,

        "font.family": "DejaVu Sans",
        "font.size": 12,

        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "axes.linewidth": 1.2,

        "xtick.color": "white",
        "ytick.color": "white",

        "axes.grid": True,
        "grid.color": "white",
        "grid.alpha": 0.18,
        "grid.linestyle": "--",

        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ==================== Вспомогательные функции ====================

def _add_basic_length_stats(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    df = df.copy()
    df["char_len"] = df[text_col].astype(str).str.len()
    df["word_len"] = df[text_col].astype(str).str.split().str.len()
    return df


def _split_stats(df: pd.DataFrame, split_name: str, label_col: str = "label") -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats["split"] = split_name
    stats["n_examples"] = int(len(df))

    if label_col in df.columns:
        vc = df[label_col].value_counts(dropna=False)
        frac = df[label_col].value_counts(normalize=True, dropna=False)
        stats["label_counts"] = {int(k): int(v) for k, v in vc.items()}
        stats["label_frac"] = {int(k): float(v) for k, v in frac.items()}

    for col in ["char_len", "word_len"]:
        if col in df.columns:
            s = df[col]
            stats[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "q05": float(s.quantile(0.05)),
                "q25": float(s.quantile(0.25)),
                "q50": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
                "q95": float(s.quantile(0.95)),
            }

    # словарь и топ-токены
    all_tokens = " ".join(df["sentence"].astype(str)).split()
    vocab = set(all_tokens)
    stats["vocab_size"] = int(len(vocab))

    top_k = 30
    counts = pd.Series(all_tokens).value_counts().head(top_k)
    stats["top_tokens"] = {str(tok): int(cnt) for tok, cnt in counts.items()}

    if HAS_TEXTSTAT:
        sample = df["sentence"].astype(str)
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        flesch, smog, fk_grade = [], [], []
        for t in sample:
            try:
                flesch.append(textstat.flesch_reading_ease(t))
                smog.append(textstat.smog_index(t))
                fk_grade.append(textstat.flesch_kincaid_grade(t))
            except Exception:
                continue
        if flesch:
            stats["readability"] = {
                "flesch_reading_ease_mean": float(np.mean(flesch)),
                "smog_index_mean": float(np.mean(smog)),
                "flesch_kincaid_grade_mean": float(np.mean(fk_grade)),
                "n_sampled": int(len(flesch)),
            }

    return stats


# ==================== Плоттинг в стиле чёрного слайда ====================

def _label_style(ax):
    """Общий стиль для осей."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("white")


def _plot_label_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="none")
    _label_style(ax)

    # нормированное распределение
    for df, name, offset in [
        (df_train, "train", -0.15),
        (df_val, "validation", 0.15),
    ]:
        vc = df["label"].value_counts(normalize=True).sort_index()
        xs = np.array(list(vc.index), dtype=float) + offset
        ax.bar(
            xs,
            vc.values,
            width=0.28,
            label=name,
            alpha=0.35,
            edgecolor="white",
            facecolor="none",
            linewidth=1.5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["negative (0)", "positive (1)"])
    ax.set_ylabel("Fraction")
    ax.set_title("SST-2 — label distribution", pad=14)
    ax.legend(frameon=False, labelcolor="white")

    fig.tight_layout()
    fig.savefig(out_dir / "label_distribution.png", transparent=True)
    plt.close(fig)


def _plot_length_histograms(df_train: pd.DataFrame, out_dir: Path):
    """Гистограммы длин: полупрозрачные белые step-кривые."""
    word_len = pd.to_numeric(df_train["word_len"], errors="coerce").dropna().to_numpy()
    char_len = pd.to_numeric(df_train["char_len"], errors="coerce").dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="none")

    # ----- words -----
    ax = axes[0]
    _label_style(ax)
    if len(word_len) > 0:
        bins_w = np.linspace(word_len.min(), word_len.max(), 41)
        counts_w, edges_w = np.histogram(word_len, bins=bins_w)
        # step-график вместо баров
        ax.step(
            edges_w[:-1],
            counts_w,
            where="post",
            linewidth=1.8,
            color="white",
            alpha=0.8,
        )

    ax.set_title("Sentence length (words)", pad=12)
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Number of examples")

    # ----- chars -----
    ax = axes[1]
    _label_style(ax)
    if len(char_len) > 0:
        bins_c = np.linspace(char_len.min(), char_len.max(), 41)
        counts_c, edges_c = np.histogram(char_len, bins=bins_c)
        ax.step(
            edges_c[:-1],
            counts_c,
            where="post",
            linewidth=1.8,
            color="white",
            alpha=0.8,
        )

    ax.set_title("Sentence length (characters)", pad=12)
    ax.set_xlabel("Characters per sentence")
    ax.set_ylabel("Number of examples")

    fig.suptitle("SST-2 — length distributions", y=1.05, fontsize=14, fontweight="bold", color="white")
    fig.tight_layout()
    fig.savefig(out_dir / "length_histograms.png", transparent=True)
    plt.close(fig)


def _plot_length_by_label(df_train: pd.DataFrame, out_dir: Path):
    """Boxplot длины в разрезе классов — минималистичные белые боксплоты."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="none")

    # word_len by label
    ax = axes[0]
    _label_style(ax)
    word_neg = pd.to_numeric(df_train.loc[df_train["label"] == 0, "word_len"], errors="coerce").dropna()
    word_pos = pd.to_numeric(df_train.loc[df_train["label"] == 1, "word_len"], errors="coerce").dropna()

    bp = ax.boxplot(
        [word_neg.to_numpy(), word_pos.to_numpy()],
        labels=["negative (0)", "positive (1)"],
        showfliers=False,
        patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor="white", linewidth=1.4)
    for median in bp["medians"]:
        median.set(color="white", linewidth=1.6)
    for whisker in bp["whiskers"]:
        whisker.set(color="white", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="white", linewidth=1.2)

    ax.set_title("Words per sentence by label", pad=12)
    ax.set_ylabel("Words per sentence")

    # char_len by label
    ax = axes[1]
    _label_style(ax)
    char_neg = pd.to_numeric(df_train.loc[df_train["label"] == 0, "char_len"], errors="coerce").dropna()
    char_pos = pd.to_numeric(df_train.loc[df_train["label"] == 1, "char_len"], errors="coerce").dropna()

    bp = ax.boxplot(
        [char_neg.to_numpy(), char_pos.to_numpy()],
        labels=["negative (0)", "positive (1)"],
        showfliers=False,
        patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor="white", linewidth=1.4)
    for median in bp["medians"]:
        median.set(color="white", linewidth=1.6)
    for whisker in bp["whiskers"]:
        whisker.set(color="white", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="white", linewidth=1.2)

    ax.set_title("Characters per sentence by label", pad=12)
    ax.set_ylabel("Characters per sentence")

    fig.suptitle("SST-2 — length vs label", y=1.05, fontsize=14, fontweight="bold", color="white")
    fig.tight_layout()
    fig.savefig(out_dir / "length_by_label.png", transparent=True)
    plt.close(fig)


def _print_example_texts(df_train: pd.DataFrame, out_dir: Path, n_per_class: int = 5):
    samples = {}
    for label in [0, 1]:
        subset = df_train[df_train["label"] == label]
        if len(subset) == 0:
            continue
        samples[str(label)] = subset.sample(
            min(n_per_class, len(subset)), random_state=42
        )["sentence"].tolist()

    with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def _build_summary_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Собирает компактную сводную таблицу по всем сплитам:
    n_examples, label_frac_*, word_len_* и char_len_*.
    """
    rows = []
    for split_name, s in stats.items():
        row: Dict[str, Any] = {"split": split_name, "n_examples": s["n_examples"]}

        if "label_frac" in s:
            for lbl, frac in s["label_frac"].items():
                row[f"label_frac_{lbl}"] = frac

        for col in ["word_len", "char_len"]:
            if col in s:
                for key, value in s[col].items():
                    row[f"{col}_{key}"] = value

        if "vocab_size" in s:
            row["vocab_size"] = s["vocab_size"]

        rows.append(row)

    return pd.DataFrame(rows)


# ==================== Основная функция ====================

def analyze_sst2(output_dir: str | Path = "reports/sst2_analysis") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("glue", "sst2")
    df_train = ds["train"].to_pandas()
    df_val = ds["validation"].to_pandas()
    df_test = ds["test"].to_pandas()

    df_train = _add_basic_length_stats(df_train, text_col="sentence")
    df_val = _add_basic_length_stats(df_val, text_col="sentence")
    df_test = _add_basic_length_stats(df_test, text_col="sentence")

    # per-split stats
    stats = {
        "train": _split_stats(df_train, "train"),
        "validation": _split_stats(df_val, "validation"),
        "test": _split_stats(df_test, "test"),
    }

    # общий датасет (train + val + test) как сводка "overall"
    df_all = pd.concat(
        [
            df_train.assign(split="train"),
            df_val.assign(split="validation"),
            df_test.assign(split="test"),
        ],
        ignore_index=True,
    )
    stats["overall"] = _split_stats(df_all, "overall")

    # детальный json
    with open(out_dir / "sst2_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # компактная сводная табличка
    summary_df = _build_summary_table(stats)
    summary_df.to_csv(out_dir / "sst2_stats_summary.csv", index=False)

    # графики
    _plot_label_distribution(df_train, df_val, out_dir)
    _plot_length_histograms(df_train, out_dir)
    _plot_length_by_label(df_train, out_dir)
    _print_example_texts(df_train, out_dir, n_per_class=10)

    # печать сводки
    print("=== SST-2 summary ===")
    for split_name in ["train", "validation", "test", "overall"]:
        s = stats[split_name]
        print(f"[{split_name}] n_examples={s['n_examples']}")
        if "label_counts" in s:
            print("  label_counts:", s["label_counts"])
            print("  label_frac:  ", {k: round(v, 3) for k, v in s["label_frac"].items()})
        if "word_len" in s:
            wl = s["word_len"]
            print(
                f"  word_len: mean={wl['mean']:.1f}, q05={wl['q05']:.1f}, "
                f"q50={wl['q50']:.1f}, q95={wl['q95']:.1f}"
            )
        if "char_len" in s:
            cl = s["char_len"]
            print(
                f"  char_len: mean={cl['mean']:.1f}, q05={cl['q05']:.1f}, "
                f"q50={cl['q50']:.1f}, q95={cl['q95']:.1f}"
            )
        print()

    print(f"Reports written to: {out_dir.resolve()}")
    return stats


if __name__ == "__main__":
    analyze_sst2()
