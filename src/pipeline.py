import os
import sys
import wandb
from dotenv import dotenv_values
from omegaconf import DictConfig
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

    run = None
    if cfg.track.wandb_mode != "disabled":
        env_vars = dotenv_values()
        os.environ.setdefault("WANDB_API_KEY", env_vars.get("WANDB_API_KEY"))
        wandb.init(
            project=cfg.track.project,
            entity=cfg.track.entity,
            mode=cfg.track.wandb_mode,
            config=dict(cfg),
            reinit=True,
        )
        run = wandb.run

    dispatch = {
        "prepare": lambda: prepare_stage.run(cfg),
        "select": lambda: select_stage.run(cfg, run_wandb=run),
        "train": lambda: train_stage.run(cfg),
        "eval": lambda: eval_stage.run(cfg),
    }

    stage = None
    if isinstance(cfg, dict) or hasattr(cfg, "get"):
        try:
            stage = cfg.get("stage")
        except Exception:
            stage = None
    if stage is None:
        stage = next((tok for tok in reversed(sys.argv) if tok in dispatch), "prepare")

    dispatch[stage]()


if __name__ == "__main__":
    main()
