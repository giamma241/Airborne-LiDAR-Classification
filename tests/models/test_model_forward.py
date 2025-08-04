import hydra
from pytorch_lightning import LightningDataModule, LightningModule

from src.models.model import Model
from src.utils import utils  # noqa
from tests.conftest import make_default_hydra_cfg


def test_model_get_batch_tensor_by_enumeration():
    config = make_default_hydra_cfg(
        overrides=[
            "predict.src_las=tests/data/toy_dataset_src/862000_6652000.classified_toy_dataset.100mx100m.las",
            "datamodule.epsg=2154",
            "work_dir=./../../..",
            "datamodule.subtile_width=1",  # Extreme case with very few points per subtile
            "datamodule.hdf5_file_path=null",
        ]
    )

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_las)

    model = Model(
        neural_net_class_name="PyGRandLANet",
        neural_net_hparams=dict(num_features=2, num_classes=7),
    )
    for batch in datamodule.predict_dataloader():
        # Check that no error is raised ("TypeError: object of type 'numpy.int64' has no len()")
        _ = model._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)


def test_model_forward():
    config = make_default_hydra_cfg(
        overrides=[
            "predict.src_las=tests/data/toy_dataset_src/862000_6652000.classified_toy_dataset.100mx100m.las",
            "datamodule.epsg=2154",
            "work_dir=./../../..",
            "datamodule.subtile_width=1",  # Extreme case with very few points per subtile
            "datamodule.hdf5_file_path=null",
        ]
    )

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_las)

    model: LightningModule = hydra.utils.instantiate(config.model)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()
    assert hasattr(model, "model")
    for batch in datamodule.predict_dataloader():
        # Check that no error is raised
        targets, logits = model.forward(batch)
