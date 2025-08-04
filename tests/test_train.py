# import os
# from pathlib import Path

# import pytest
# from hydra.core.hydra_config import HydraConfig
# from omegaconf import DictConfig, open_dict

# from src.train import train
# from tests.helpers.run_if import RunIf


# def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
#     """Run for 1 train, val and test step.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.fast_dev_run = True
#         cfg_train.trainer.accelerator = "cpu"
#     train(cfg_train)


# @RunIf(min_gpus=1)
# def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
#     """Run for 1 train, val and test step on GPU.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.fast_dev_run = True
#         cfg_train.trainer.accelerator = "gpu"
#     train(cfg_train)


# @RunIf(min_gpus=1)
# @pytest.mark.slow
# def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
#     """Train 1 epoch on GPU with mixed-precision.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1
#         cfg_train.trainer.accelerator = "gpu"
#         cfg_train.trainer.precision = 16
#     train(cfg_train)


# @pytest.mark.slow
# def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
#     """Train 1 epoch with validation loop twice per epoch.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1
#         cfg_train.trainer.val_check_interval = 0.5
#     train(cfg_train)


# @pytest.mark.slow
# def test_train_ddp_sim(cfg_train: DictConfig) -> None:
#     """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 2
#         cfg_train.trainer.accelerator = "cpu"
#         cfg_train.trainer.devices = 2
#         cfg_train.trainer.strategy = "ddp_spawn"
#     train(cfg_train)


# @pytest.mark.slow
# def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
#     """Run 1 epoch, finish, and resume for another epoch.

#     :param tmp_path: The temporary logging path.
#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     with open_dict(cfg_train):
#         cfg_train.trainer.max_epochs = 1

#     HydraConfig().set_config(cfg_train)
#     metric_dict_1, _ = train(cfg_train)

#     files = os.listdir(tmp_path / "checkpoints")
#     assert "last.ckpt" in files
#     assert "epoch_000.ckpt" in files

#     with open_dict(cfg_train):
#         cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
#         cfg_train.trainer.max_epochs = 2

#     metric_dict_2, _ = train(cfg_train)

#     files = os.listdir(tmp_path / "checkpoints")
#     assert "epoch_001.ckpt" in files
#     assert "epoch_002.ckpt" not in files

#     assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
#     assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]

from typing import List

import numpy as np
import pytest
from lightning.pytorch.accelerators import find_usable_cuda_devices

from src.pctl.dataset.utils import pdal_read_las_array
from src.train import train
from tests.conftest import (
    make_default_hydra_cfg,
)
from tests.helpers.run_if import RunIf

"""
Sanity checks to make sure the model train/val/predict/test logics do not crash.
"""


@pytest.fixture(scope="session")
def one_epoch_trained_RandLaNet_checkpoint(toy_dataset_hdf5_path, tmpdir_factory):
    """Train a RandLaNet model for one epoch, in order to run it in different other tests.

    Args:
        toy_dataset_hdf5_path (str): path to toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session level tempdir.

    Returns:
        str: path to trained model checkpoint, which persists for the whole pytest session.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=["experiment=RandLaNetDebug"] + tmp_paths_overrides
    )
    trainer = train(cfg_one_epoch)
    return trainer.checkpoint_callback.best_model_path


@RunIf(min_gpus=1)
def test_FrenchLidar_RandLaNetDebug_with_gpu(toy_dataset_hdf5_path, tmpdir_factory):
    """Train a RandLaNet model for one epoch using GPU. XFail is no GPU available.

    Args:
        toy_dataset_hdf5_path (str): path to isolated toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session-level tempdir.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    gpu_id = find_usable_cuda_devices(1)
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=[
            "experiment=RandLaNetDebug",
            "trainer.accelerator=gpu",
            f"trainer.devices={gpu_id}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_one_epoch)


def test_run_test_with_trained_model_on_toy_dataset_on_cpu(
    one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir
):
    _run_test_right_after_training(
        one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir, "cpu"
    )


@RunIf(min_gpus=1)
def test_run_test_with_trained_model_on_toy_dataset_on_gpu(
    one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir
):
    _run_test_right_after_training(
        one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir, "gpu"
    )


def _run_test_right_after_training(
    one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir, accelerator
):
    """Run test using the model that was just trained for one epoch.

    Args:
        toy_dataset_hdf5_path (fixture -> str): path to toy dataset
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Run testing on toy testset with trainer.test(...)
    # function's name is train, but under the hood and thanks to configuration,
    # trainer.test(...) is called.
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    devices = find_usable_cuda_devices(1) if accelerator == "gpu" else 1
    cfg_test_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=test",  # sets task.task_name to "test"
            f"model.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
            f"trainer.devices={devices}",
            f"trainer.accelerator={accelerator}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_test_using_trained_model)


def check_las_contains_dims(las_path: str, dims_to_check: List[str] = []):
    """Utility: check that LAS contains some dimensions.


    Args:
        las_path (str): path to LAS file.
        dims_to_check (List[str], optional): list of dimensions expected to be there. Defaults to [].

    """
    a1 = pdal_read_las_array(las_path, "2154")
    for dim in dims_to_check:
        assert dim in a1.dtype.fields.keys()


def check_las_does_not_contains_dims(las_path, dims_to_check=[]):
    """Utility: check that LAS does NOT contain some dimensions.


    Args:
        las_path (str): path to LAS file.
        dims_to_check (List[str], optional): list of dimensions expected not to be there. Defaults to [].

    """
    a1 = pdal_read_las_array(las_path, "2154")
    for dim in dims_to_check:
        assert dim not in a1.dtype.fields.keys()


def check_las_invariance(las_path_1: str, las_path_2: str):
    """Check that key dimensions are equal between two LAS files

    Args:
        las_path_1 (str): path to first LAS file.
        las_path_2 (str): path to second LAS file.

    """
    a1 = pdal_read_las_array(las_path_1, "2154")
    a2 = pdal_read_las_array(las_path_2, "2154")
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert a1.shape == a2.shape  # no loss of points
    assert all(d in a2.dtype.fields.keys() for d in key_dims)  # key dims are here

    # order of points is allowed to change, so we assess a relaxed equality.
    rel_tolerance = 0.0001
    for d in key_dims:
        assert pytest.approx(np.min(a2[d]), rel_tolerance) == np.min(a1[d])
        assert pytest.approx(np.max(a2[d]), rel_tolerance) == np.max(a1[d])
        assert pytest.approx(np.mean(a2[d]), rel_tolerance) == np.mean(a1[d])
        assert pytest.approx(np.sum(a2[d]), rel_tolerance) == np.sum(a1[d])


def _make_list_of_necesary_hydra_overrides_with_tmp_paths(
    toy_dataset_hdf5_path: str, tmpdir: str
):
    """Get list of overrides for hydra, the ones that are always needed when calling train/test.

    Args:
        toy_dataset_hdf5_path (str): path to directory to dataset.
        tmpdir (str): path to temporary directory.

    """

    return [
        f"datamodule.hdf5_file_path={toy_dataset_hdf5_path}",
        "logger=csv",  # disables comet logging
        f"logger.csv.save_dir={tmpdir}",
        f"callbacks.model_checkpoint.dirpath={tmpdir}",
    ]
