import json
import os

import pandas as pd
import wandb
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..metrics import compute_metrics
from ..utils import time_block


def run(cfg):
    manifest_path = os.path.join(
        "data",
        "manifests",
        f"{cfg.selection.name}_p{int(round(cfg.subset.frac * 100)):02d}_seed{cfg.seed}.csv",
    )
    df = pd.read_csv(manifest_path)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.seed)
    train_idx, val_idx = next(splitter.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.data.max_length,
        )

    train_ds = (
        Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
        .map(encode, batched=True)
        .rename_column("label", "labels")
        .with_format("torch")
    )
    val_ds = (
        Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
        .map(encode, batched=True)
        .rename_column("label", "labels")
        .with_format("torch")
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=2,
    )

    out_dir = os.path.join(
        "artifacts",
        cfg.selection.name,
        f"p{int(round(cfg.subset.frac * 100)):02d}",
        f"seed{cfg.seed}",
        "model",
    )
    os.makedirs(out_dir, exist_ok=True)

    report_to = ["wandb"] if cfg.track.wandb_mode != "disabled" else []
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.train.lr,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        eval_steps=cfg.train.eval_steps,
        logging_steps=cfg.train.eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to=report_to,
        eval_strategy="steps",
    )

    def _compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return compute_metrics(pred.label_ids, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    with time_block() as elapsed:
        trainer.train()
    train_time = elapsed()

    # Явно считаем метрики по валидации, как в eval.py
    eval_result = trainer.evaluate()
    pred_output = trainer.predict(val_ds)
    y_true = pred_output.label_ids
    y_pred = pred_output.predictions.argmax(-1)

    core_metrics = compute_metrics(y_true, y_pred)
    core_metrics = {k: float(v) for k, v in core_metrics.items()}  # accuracy, f1, precision, recall

    extra_metrics = {}
    if "eval_loss" in eval_result:
        extra_metrics["loss"] = float(eval_result["eval_loss"])

    metrics = {**core_metrics, **extra_metrics}

    metrics_path = os.path.join(os.path.dirname(out_dir), "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if cfg.track.wandb_mode != "disabled" and wandb.run:
        wandb.log(
            {
                **{f"train/{key}" for key in []},  # заглушка, чтобы не забыть что тут есть логика ниже
            }
        )
        # нормальный лог — одним словарём
        wandb.log(
            {
                **{f"train/{k}": v for k, v in metrics.items()},
                "train/examples": len(train_df),
                "train/validation_examples": len(val_df),
                "train/time_sec": train_time,
            }
        )

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    return out_dir
