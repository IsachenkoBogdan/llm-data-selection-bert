import os
import sys
import wandb
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
import hydra

from . import utils
from .stages import (
    eval as eval_stage,
    prepare as prepare_stage,
    select as select_stage,
    train as train_stage,
)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    utils.set_seed(cfg.seed)

    # аккуратно определяем stage
    stage = None
    try:
        stage = cfg.get("stage")
    except Exception:
        stage = None
    if stage is None:
        stage = next((tok for tok in reversed(sys.argv) if tok in ["prepare", "select", "train", "eval"]), "prepare")

    # ---- W&B init с осмысленным именем ----
    run = None
    if cfg.track.wandb_mode != "disabled":
        env_vars = dotenv_values()
        os.environ.setdefault("WANDB_API_KEY", env_vars.get("WANDB_API_KEY"))

        # селектор / frac могут отсутствовать на stage=prepare
        sel_name = getattr(cfg.selection, "name", "none") if hasattr(cfg, "selection") else "none"
        frac = float(getattr(cfg.subset, "frac", 1.0)) if hasattr(cfg, "subset") else 1.0
        pct = int(round(frac * 100))

        # человекочитаемое имя ранна
        run_name = f"{stage}-{sel_name}-p{pct:02d}-seed{cfg.seed}"

        # group: все стадии для одного (selector, frac, seed) в одной группе
        group_name = f"{sel_name}_p{pct:02d}_seed{cfg.seed}"

        # тэги для быстрого фильтра в UI
        tags = [
            f"stage:{stage}",
            f"selector:{sel_name}",
            f"frac:{frac:.2f}",
            f"dataset:{getattr(cfg.data, 'dataset', 'unknown')}",
            f"model:{getattr(cfg.model, 'name', 'unknown')}",
        ]

        wandb.init(
            project=cfg.track.project,
            entity=cfg.track.entity,
            mode=cfg.track.wandb_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            group=group_name,
            tags=tags,
            # reinit=True  # можешь оставить, но он уже deprecated как bool — warning можно игнорить
        )
        run = wandb.run

    dispatch = {
        "prepare": lambda: prepare_stage.run(cfg),
        "select":  lambda: select_stage.run(cfg, run_wandb=run),
        "train":   lambda: train_stage.run(cfg),
        "eval":    lambda: eval_stage.run(cfg),
    }

    dispatch[stage]()


if __name__ == "__main__":
    main()
