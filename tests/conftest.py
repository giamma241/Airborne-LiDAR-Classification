"""This file prepares config fixtures for other tests."""
import os
import subprocess
from typing import List

import pytest
from hydra import compose, initialize
from pytorch_lightning import seed_everything

from src.pctl.dataset.toy_dataset import make_toy_dataset_from_test_file

# @pytest.fixture(scope="package")
# def cfg_train_global() -> DictConfig:
#     """A pytest fixture for setting up a default Hydra DictConfig for training.

#     :return: A DictConfig object containing a default Hydra configuration for training.
#     """
#     with initialize(version_base="1.3", config_path="../configs"):
#         cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

#         # set defaults for all tests
#         with open_dict(cfg):
#             cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
#             cfg.trainer.max_epochs = 1
#             cfg.trainer.limit_train_batches = 0.01
#             cfg.trainer.limit_val_batches = 0.1
#             cfg.trainer.limit_test_batches = 0.1
#             cfg.trainer.accelerator = "cpu"
#             cfg.trainer.devices = 1
#             cfg.data.num_workers = 0
#             cfg.data.pin_memory = False
#             cfg.extras.print_config = False
#             cfg.extras.enforce_tags = False
#             cfg.logger = None

#     return cfg


# @pytest.fixture(scope="package")
# def cfg_eval_global() -> DictConfig:
#     """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

#     :return: A DictConfig containing a default Hydra configuration for evaluation.
#     """
#     with initialize(version_base="1.3", config_path="../configs"):
#         cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

#         # set defaults for all tests
#         with open_dict(cfg):
#             cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
#             cfg.trainer.max_epochs = 1
#             cfg.trainer.limit_test_batches = 0.1
#             cfg.trainer.accelerator = "cpu"
#             cfg.trainer.devices = 1
#             cfg.data.num_workers = 0
#             cfg.data.pin_memory = False
#             cfg.extras.print_config = False
#             cfg.extras.enforce_tags = False
#             cfg.logger = None

#     return cfg


# @pytest.fixture(scope="function")
# def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
#     """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
#     logging path `tmp_path` for generating a temporary logging path.

#     This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

#     :param cfg_train_global: The input DictConfig object to be modified.
#     :param tmp_path: The temporary logging path.

#     :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
#     """
#     cfg = cfg_train_global.copy()

#     with open_dict(cfg):
#         cfg.paths.output_dir = str(tmp_path)
#         cfg.paths.log_dir = str(tmp_path)

#     yield cfg

#     GlobalHydra.instance().clear()


# @pytest.fixture(scope="function")
# def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
#     """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
#     logging path `tmp_path` for generating a temporary logging path.

#     This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

#     :param cfg_train_global: The input DictConfig object to be modified.
#     :param tmp_path: The temporary logging path.

#     :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
#     """
#     cfg = cfg_eval_global.copy()

#     with open_dict(cfg):
#         cfg.paths.output_dir = str(tmp_path)
#         cfg.paths.log_dir = str(tmp_path)

#     yield cfg

#     GlobalHydra.instance().clear()

SINGLE_POINT_CLOUD = "tests/data/single-point-cloud.laz"
DEFAULT_EPSG = "2154"


@pytest.fixture(scope="session")
def toy_dataset_hdf5_path(tmpdir_factory):
    """Creates a toy dataset accessible from within the pytest session."""
    return make_toy_dataset_from_test_file()


def make_default_hydra_cfg(overrides=[]):
    """Compose the repository hydra config, with specified overrides."""
    with initialize(config_path="./../configs/", job_name="config"):
        # there is no hydra:runtime.cwd when using compose, and therefore we have
        # to specify where our working directory is.
        workdir_override = ["work_dir=./../"]
        return compose(config_name="config", overrides=workdir_override + overrides)


@pytest.fixture(autouse=True)  # Auto-used for every test function
def set_logs_dir_env_variable(monkeypatch):
    """Sets where hydra saves its logs, as we cannot rely on a .env file for tests.

    See: https://docs.pytest.org/en/stable/how-to/monkeypatch.html#monkeypatching-environment-variables

    """
    monkeypatch.setenv("LOGS_DIR", "tests/logs/")
    # to ignore it when making prediction
    # However, this seems not found when running via CLI.
    monkeypatch.setenv("PREPARED_DATA_DIR", "placeholder")


@pytest.fixture(autouse=True)  # Auto-used for every test function
def seed_everything_in_tests():
    seed_everything(12345, workers=True)


def run_command(command: List[str]):
    """Run shell command and fail the test if it returns an error."""
    try:
        subprocess.run(["python"] + command, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(reason=e.stderr if e.stderr else str(e))


def run_command_with_return_error(command: List[str]) -> str:
    """Run shell command and return stderr output if it fails."""
    try:
        result = subprocess.run(
            ["python"] + command, check=True, capture_output=True, text=True
        )
        return ""  # no error
    except subprocess.CalledProcessError as e:
        return e.stderr or str(e)


def run_hydra_decorated_command_with_return_error(command: List[str]):
    """Default method for executing hydra decorated shell commands with pytest."""
    hydra_specific_paths = [
        "hydra.run.dir=" + os.getcwd(),
    ]
    return run_command_with_return_error(command + hydra_specific_paths)


def run_hydra_decorated_command(command: List[str]):
    """Default method for executing hydra decorated shell commands with pytest."""
    hydra_specific_paths = [
        "hydra.run.dir=" + os.getcwd(),
    ]

    run_command(command + hydra_specific_paths)


def test_sanity():
    assert True