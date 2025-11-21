import json
import os

import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..metrics import compute_metrics
from ..utils import time_block


def _p_tag(cfg) -> str:
    # Совпадает с train.py и Snakefile: p05, p10, p20
    return f"p{int(round(cfg.subset.frac * 100)):02d}"


def run(cfg):
    dataset = load_dataset("glue", "sst2", split="validation")

    p_tag = _p_tag(cfg)

    model_dir = os.path.join(
        "artifacts",
        cfg.selection.name,
        p_tag,
        f"seed{cfg.seed}",
        "model",
    )

    tokenizer_source = (
        model_dir
        if os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
        else cfg.model.name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    def encode(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=cfg.data.max_length,
        )

    eval_dataset = (
        dataset.map(encode, batched=True)
        .rename_column("label", "labels")
        .with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "eval"),
        per_device_eval_batch_size=cfg.train.batch_size,
        report_to=[],
    )

    def _compute_metrics(pred):
        predictions = pred.predictions.argmax(-1)
        return compute_metrics(pred.label_ids, predictions)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    with time_block() as elapsed:
        result = trainer.evaluate()
    eval_time = elapsed()

    metrics = {
        key.replace("eval_", ""): float(value)
        for key, value in result.items()
        if key.startswith("eval_") and key != "eval_loss"
    }

    metrics_path = os.path.join(
        "artifacts",
        cfg.selection.name,
        p_tag,
        f"seed{cfg.seed}",
        "metrics.json",
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if cfg.track.wandb_mode != "disabled" and wandb.run is not None:
        wandb.log(
            {
                "eval/time_sec": eval_time,
                **{f"eval/{k}": v for k, v in metrics.items()},
            }
        )

    return metrics_path
