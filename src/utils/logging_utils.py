import logging
import time
from typing import List

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Log select config fields + model parameter counts to Lightning loggers."""

    hparams = {
        "trainer": config["trainer"],
        "model": config["model"],
        "datamodule": config["datamodule"],
    }

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    trainer.logger.log_hyperparams(hparams)
    trainer.logger.log_hyperparams = empty  # block further logging


def eval_time(method):
    """Decorator to log the duration of the decorated method."""

    def timed(*args, **kwargs):
        log = get_logger()
        time_start = time.time()
        result = method(*args, **kwargs)
        time_elapsed = round(time.time() - time_start, 2)
        log.info(f"Runtime of {method.__name__}: {time_elapsed}s")
        return result

    return timed
