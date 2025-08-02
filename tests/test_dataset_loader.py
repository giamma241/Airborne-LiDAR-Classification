import numpy as np
import laspy
from pathlib import Path

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.datasets.dataset_loader import (
    PointCloudDataset,
    grid_patch,
    load_point_cloud,
    sample_random,
    save_block,
)


def _create_sample_npy(path: Path) -> None:
    arr = np.zeros(
        4,
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("intensity", "f8"),
            ("classification", "u1"),
        ],
    )
    arr["x"] = [0, 1, 0, 1]
    arr["y"] = [0, 0, 1, 1]
    arr["z"] = [0, 1, 2, 3]
    arr["intensity"] = [10, 20, 30, 40]
    arr["classification"] = [1, 2, 3, 4]
    np.save(path, arr)


def _create_sample_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = [0, 1, 0, 1]
    las.y = [0, 0, 1, 1]
    las.z = [0, 1, 2, 3]
    las.intensity = [10, 20, 30, 40]
    las.classification = [1, 2, 3, 4]
    las.write(path)


def test_load_point_cloud(tmp_path: Path) -> None:
    npy_path = tmp_path / "sample.npy"
    las_path = tmp_path / "sample.las"
    _create_sample_npy(npy_path)
    _create_sample_las(las_path)
    f_np, l_np = load_point_cloud(npy_path)
    f_las, l_las = load_point_cloud(las_path)
    assert f_np.shape == (4, 4)
    assert f_las.shape == (4, 4)
    assert l_np.shape == (4,)
    assert l_las.shape == (4,)


def test_sample_and_patch(tmp_path: Path) -> None:
    npy_path = tmp_path / "sample.npy"
    _create_sample_npy(npy_path)
    features, labels = load_point_cloud(npy_path)
    s_features, s_labels = sample_random(features, labels, 2)
    assert s_features.shape[0] == 2
    patches = list(grid_patch(features, labels, size=1.5, stride=1.0))
    assert len(patches) > 0


def test_dataset_class(tmp_path: Path) -> None:
    features = np.random.rand(10, 4).astype(np.float32)
    labels = np.random.randint(0, 5, 10).astype(np.uint8)
    out_dir = tmp_path / "cache" / "train"
    save_block(features, labels, out_dir / "block1.npz")
    save_block(features, labels, out_dir / "block2.npz")
    ds = PointCloudDataset(
        tmp_path / "cache", split="train", return_numpy=torch is None
    )
    assert len(ds) == 2
    x, y = ds[0]
    if torch is None:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
    else:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    assert x.shape[1] == 4
    assert y.ndim == 1
