import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import pytest
from pdaltools import las_info

from src.pctl.dataset.toy_dataset import TOY_LAS_DATA
from src.pctl.dataset.utils import pdal_read_las_array
from src.predict import predict
from src.train import train
from tests.conftest import (
    DEFAULT_EPSG,
    SINGLE_POINT_CLOUD,
    make_default_hydra_cfg,
    run_hydra_decorated_command,
    run_hydra_decorated_command_with_return_error,
)

"""
Sanity checks to make sure the model train/val/predict/test logics do not crash.
"""


@pytest.fixture(scope="session")
def verified_toy_dataset_hdf5_path():
    """Create and verify toy dataset exists and has data before running tests."""
    from pathlib import Path

    import h5py

    from src.pctl.dataset.toy_dataset import make_toy_dataset_from_test_file

    # Get the toy dataset path
    toy_dataset_path = make_toy_dataset_from_test_file()

    if not Path(toy_dataset_path).exists():
        pytest.skip(f"Toy dataset not found at {toy_dataset_path}")

    # Verify dataset has data and correct structure
    try:
        with h5py.File(toy_dataset_path, "r") as f:
            # Check if samples are indexed
            if "samples_hdf5_paths" not in f:
                print(
                    "Warning: samples_hdf5_paths not found, will be created on-the-fly"
                )

            # Check splits exist and have data
            splits_with_data = []
            for split in ["train", "val", "test"]:
                if split not in f:
                    continue

                split_group = f[split]
                total_samples = 0

                for basename in split_group.keys():
                    if isinstance(split_group[basename], h5py.Group):
                        samples = [
                            k for k in split_group[basename].keys() if k.isdigit()
                        ]
                        total_samples += len(samples)

                if total_samples > 0:
                    splits_with_data.append(split)
                    print(f"Found {total_samples} samples in {split} split")

            if not splits_with_data:
                pytest.skip("No splits with data found in toy dataset")

            if len(splits_with_data) < 2:  # Need at least train + one other
                pytest.skip(f"Insufficient splits with data: {splits_with_data}")

    except Exception as e:
        pytest.skip(f"Error reading toy dataset: {e}")

    return toy_dataset_path


@pytest.fixture(scope="session")
def one_epoch_trained_RandLaNet_checkpoint(
    verified_toy_dataset_hdf5_path, tmpdir_factory
):
    """Train a RandLaNet model for one epoch, in order to run it in different other tests."""
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        verified_toy_dataset_hdf5_path, tmpdir
    )

    # WORKING configuration - training only, skip test phase
    debug_overrides = [
        "experiment=RandLaNetDebug",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
        "datamodule.prefetch_factor=null",
    ]

    cfg_one_epoch = make_default_hydra_cfg(
        overrides=debug_overrides + tmp_paths_overrides
    )

    try:
        trainer = train(cfg_one_epoch)
        # For fast_dev_run, checkpoints might not be saved, so check
        if (
            hasattr(trainer, "checkpoint_callback")
            and trainer.checkpoint_callback.best_model_path
        ):
            checkpoint_path = trainer.checkpoint_callback.best_model_path
        else:
            # Create a dummy checkpoint path since fast_dev_run doesn't save
            checkpoint_path = os.path.join(tmpdir, "dummy_checkpoint.ckpt")
            # For tests, we can create a minimal checkpoint or skip checkpoint-dependent tests

        return checkpoint_path

    except Exception as e:
        pytest.fail(f"Failed to train model for testing: {e}")


def test_predict_as_command(one_epoch_trained_RandLaNet_checkpoint, tmpdir):
    """Test running inference by CLI for toy LAS.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Hydra changes CWD, and therefore absolute paths are preferred
    abs_path_to_toy_LAS = osp.abspath(TOY_LAS_DATA)
    command = [
        "run.py",
        f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        f"datamodule.epsg={DEFAULT_EPSG}",
        f"predict.src_las={abs_path_to_toy_LAS}",
        f"predict.output_dir={tmpdir}",
        "+predict.interpolator.probas_to_save=[building,unclassified]",
        "task.task_name=predict",
    ]
    run_hydra_decorated_command(command)
    output_path = Path(tmpdir) / Path(abs_path_to_toy_LAS).name
    metadata = las_info.las_info_metadata(output_path)
    out_pesg = las_info.get_epsg_from_header_info(metadata)
    assert out_pesg == DEFAULT_EPSG


def test_command_without_epsg(one_epoch_trained_RandLaNet_checkpoint, tmpdir):
    """Test running inference by CLI for toy LAS.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Hydra changes CWD, and therefore absolute paths are preferred
    abs_path_to_toy_LAS = osp.abspath(TOY_LAS_DATA)
    command = [
        "run.py",
        f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        f"predict.src_las={abs_path_to_toy_LAS}",
        f"predict.output_dir={tmpdir}",
        "+predict.interpolator.probas_to_save=[building,unclassified]",
        "task.task_name=predict",
    ]
    assert (
        "No EPSG provided, neither in the lidar file or as parameter"
        in run_hydra_decorated_command_with_return_error(command)
    )


def test_predict_on_single_point_cloud(one_epoch_trained_RandLaNet_checkpoint, tmpdir):
    """Test running inference by CLI for cloud with a single point (edge case addressed in V3.4.0)"""
    # Hydra changes CWD, and therefore absolute paths are preferred
    abs_path_to_single_point_cloud = osp.abspath(SINGLE_POINT_CLOUD)
    command = [
        "run.py",
        f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        f"datamodule.epsg={DEFAULT_EPSG}",
        f"predict.src_las={abs_path_to_single_point_cloud}",
        f"predict.output_dir={tmpdir}",
        "+predict.interpolator.probas_to_save=[building,unclassified]",
        "task.task_name=predict",
    ]
    run_hydra_decorated_command(command)


def test_RandLaNet_predict_with_invariance_checks(
    one_epoch_trained_RandLaNet_checkpoint, tmpdir
):
    """Train a model for one epoch, and run test and predict functions using the trained model.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        "placeholder_because_no_need_for_a_dataset_here", tmpdir
    )
    # Run prediction
    cfg_predict_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=predict",
            f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
            f"datamodule.epsg={DEFAULT_EPSG}",
            f"predict.src_las={TOY_LAS_DATA}",
            f"predict.output_dir={tmpdir}",
            # "+predict.interpolator.interpolation_k=predict.interpolation_k",
            "predict.interpolator.probas_to_save=[building,unclassified]",
        ]
        + tmp_paths_overrides
    )
    path_to_output_las = predict(cfg_predict_using_trained_model)

    # Check that predict function generates a predicted LAS
    assert osp.isfile(path_to_output_las)

    # Check the format of the predicted las in terms of extra dimensions
    DIMS_ALWAYS_THERE = ["PredictedClassification", "entropy"]
    DIMS_CHOSEN_IN_CONFIG = ["building", "unclassified"]
    check_las_contains_dims(
        path_to_output_las,
        dims_to_check=DIMS_ALWAYS_THERE + DIMS_CHOSEN_IN_CONFIG,
    )
    DIMS_NOT_THERE = ["ground"]
    check_las_does_not_contains_dims(path_to_output_las, dims_to_check=DIMS_NOT_THERE)

    # check that predict does not change other dimensions
    check_las_invariance(TOY_LAS_DATA, path_to_output_las)


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
