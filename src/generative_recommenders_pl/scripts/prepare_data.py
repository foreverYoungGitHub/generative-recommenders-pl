from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.data.preprocessor import DataProcessor
from generative_recommenders_pl.utils.logger import configure_logger

log = configure_logger(__name__)


OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    version_base="1.3", config_path="../../../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> int:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"{OmegaConf.to_yaml(cfg.data.data_preprocessor)}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    preprocessor: DataProcessor = hydra.utils.instantiate(
        cfg.data.data_preprocessor
    )
    
    Path(preprocessor.output_format_csv()).parent.mkdir(parents=True, exist_ok=True)
    num_unique_items = preprocessor.preprocess_rating()
    return num_unique_items

if __name__ == "__main__":
    main()
