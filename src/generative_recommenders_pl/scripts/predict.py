from typing import Any

import hydra
import lightning as L
import torch.multiprocessing
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.utils.instantiators import instantiate_loggers
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval.yaml")
def predict(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    if cfg.ckpt_path is None or cfg.ckpt_path == "":
        raise ValueError("Please provide a checkpoint path for prediction!")

    if cfg.output_file is None or cfg.output_file == "":
        raise ValueError("Please provide an output file for predictions!")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(
        cfg.model, datamodule=datamodule, _recursive_=False
    )

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("Starting prediction!")
    predictions = trainer.predict(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )

    log.info(f"Saving predictions to {cfg.output_file}")
    datamodule.save_predictions(cfg.output_file, predictions)


if __name__ == "__main__":
    predict()
