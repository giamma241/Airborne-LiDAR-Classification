import argparse
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Iterator, List, Optional, Tuple, Union, cast

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover
    torch = cast(ModuleType | None, None)  # type: ignore[assignment, no-redef]

    class TorchDataset:  # type: ignore
        """Fallback dataset base class when PyTorch is unavailable."""

        pass


import yaml  # type: ignore[import-untyped]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def load_point_cloud(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a point cloud from LAS/LAZ or NumPy format.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the input file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features of shape ``(N, 4)`` (x, y, z, intensity) and labels of shape ``(N,)``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".las", ".laz"}:
        import laspy  # type: ignore[import-untyped]

        las = laspy.read(p)
        features = np.vstack((las.x, las.y, las.z, las.intensity)).T
        labels = np.asarray(las.classification, dtype=np.uint8)
    elif suffix == ".npy":
        arr = np.load(p)
        features = np.vstack((arr["x"], arr["y"], arr["z"], arr["intensity"])).T
        labels = arr["classification"].astype(np.uint8)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    return features.astype(np.float32), labels


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize Z and intensity features.

    Z values are shifted so that the minimum is zero. Intensity values
    are scaled to ``[0, 1]`` based on their maximum.

    Parameters
    ----------
    features : np.ndarray
        Array of shape ``(N, 4)``.

    Returns
    -------
    np.ndarray
        Normalized feature array of shape ``(N, 4)``.
    """
    norm = features.copy()
    norm[:, 2] -= norm[:, 2].min()
    max_intensity = norm[:, 3].max()
    if max_intensity > 0:
        norm[:, 3] /= max_intensity
    return norm


def sample_random(
    features: np.ndarray, labels: np.ndarray, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample points from a point cloud."""
    if features.shape[0] <= num_points:
        return features, labels
    idx = np.random.choice(features.shape[0], num_points, replace=False)
    return features[idx], labels[idx]


def grid_patch(
    features: np.ndarray,
    labels: np.ndarray,
    size: float,
    stride: Optional[float] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield patches using a sliding window on the XY plane."""
    if stride is None:
        stride = size
    x = features[:, 0]
    y = features[:, 1]
    min_x, min_y = x.min(), y.min()
    max_x, max_y = x.max(), y.max()
    for x0 in np.arange(min_x, max_x, stride):
        for y0 in np.arange(min_y, max_y, stride):
            mask = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
            if np.any(mask):
                yield features[mask], labels[mask]


def save_block(
    features: np.ndarray, labels: np.ndarray, path: Path, fmt: str = "npz"
) -> None:
    """Save a single block to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        np.savez_compressed(path, features=features, labels=labels)
    elif fmt == "pt":
        torch.save(
            {
                "features": torch.from_numpy(features),
                "labels": torch.from_numpy(labels),
            },
            path,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")


class PointCloudDataset(TorchDataset):
    """Dataset wrapping cached point cloud blocks."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        return_numpy: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.return_numpy = return_numpy or torch is None
        split_dir = self.root_dir / split
        self.files: List[Path] = sorted(split_dir.glob("*.npz")) + sorted(
            split_dir.glob("*.pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No cached blocks found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        if file_path.suffix == ".npz":
            data = np.load(file_path)
            features = data["features"]
            labels = data["labels"]
        else:
            data = torch.load(file_path)
            features = data["features"]
            labels = data["labels"]
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
        if self.return_numpy or torch is None:
            return features, labels
        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(labels.astype(np.uint8)),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point cloud dataset converter")
    parser.add_argument("input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "grid"],
        default="random",
        help="Block generation method",
    )
    parser.add_argument(
        "--num-points", type=int, default=4096, help="Random sample size"
    )
    parser.add_argument(
        "--patch-size", type=float, default=10.0, help="Grid patch size"
    )
    parser.add_argument("--stride", type=float, default=None, help="Grid stride")
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "pt"],
        default="npz",
        help="Cache format",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config file")
    return parser.parse_args()


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        if cfg_path.suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    for key, value in cfg.items():
        setattr(args, key.replace("-", "_"), value)
    return args


def main() -> None:
    """CLI entry point for converting point cloud files."""
    args = _apply_config(_parse_args())
    input_path = Path(args.input)
    if input_path.is_dir():
        files = [
            *input_path.glob("*.las"),
            *input_path.glob("*.laz"),
            *input_path.glob("*.npy"),
        ]
    else:
        files = [input_path]
    out_dir = Path(args.output) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        logging.info("Processing %s", file_path)
        features, labels = load_point_cloud(file_path)
        features = normalize_features(features)
        if args.method == "random":
            f_block, l_block = sample_random(features, labels, args.num_points)
            save_block(
                f_block,
                l_block,
                out_dir / f"{file_path.stem}.{args.format}",
                fmt=args.format,
            )
        else:
            for i, (f_block, l_block) in enumerate(
                grid_patch(features, labels, args.patch_size, args.stride)
            ):
                save_block(
                    f_block,
                    l_block,
                    out_dir / f"{file_path.stem}_{i:04d}.{args.format}",
                    fmt=args.format,
                )
    logging.info("Saved blocks to %s", out_dir)


if __name__ == "__main__":
    main()
