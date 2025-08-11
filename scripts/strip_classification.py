#!/usr/bin/env python3
"""
PDAL-based LAS Cleaner
======================

Purpose:
- Reset the standard LAS 'Classification' field (cannot be removed from the standard).
- Remove any extra dimensions like 'PredictedClassification', 'entropy', 'building', 'ground', etc.

Usage:
  python strip_classification.py --in input.las --out output.las
  # optional: specify which extra-dims to remove
  python strip_classification.py --in input.las --out output.las --drop PredictedClassification entropy building ground
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Set

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdal-cleaner")


def check_pdal() -> None:
    """Check if PDAL is installed and available."""
    try:
        r = subprocess.run(
            ["pdal", "--version"], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        log.info(f"PDAL: {r.stdout.strip()}")
    except FileNotFoundError:
        log.error("PDAL not found. Install with: conda install -c conda-forge pdal")
        sys.exit(1)


def pdal_list_dims(las_path: Path) -> Set[str]:
    """Return the set of dimension names present in the file (via `pdal info`)."""
    cmd = ["pdal", "info", "--metadata", str(las_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning(
            "Unable to read metadata with pdal info; continuing without auto-detection of extra-dims."
        )
        return set()
    try:
        meta = json.loads(r.stdout)
        dims = meta.get("metadata", {}).get("readers.las", {}).get("dimensions", [])
        return set(dims)
    except Exception:
        return set()


def create_cleanup_pipeline(
    input_las: Path, output_las: Path, dims_to_drop: List[str]
) -> dict:
    """
    Create PDAL pipeline:
      readers.las -> (filters.drop?) -> filters.assign(Classification=1) -> writers.las
    """
    pipeline = [{"type": "readers.las", "filename": str(input_las.absolute())}]

    # If there are extra-dims to remove, use filters.drop
    if dims_to_drop:
        pipeline.append({"type": "filters.drop", "dimensions": ",".join(dims_to_drop)})

    # Reset the standard Classification field
    # Note: cannot be deleted, only set to a value.
    pipeline.append(
        {
            "type": "filters.assign",
            "assignment": "Classification[:] = 1",  # or 1 for 'Unclassified'
        }
    )

    # Write output
    pipeline.append(
        {
            "type": "writers.las",
            "filename": str(output_las.absolute()),
            "minor_version": "4",  # LAS 1.4
            "dataformat_id": "3",  # Point format 3 (XYZ, Intensity, Time, RGB)
            "forward": "all",  # forward all other dimensions
        }
    )
    return {"pipeline": pipeline}


def run_pipeline(pipeline: dict) -> bool:
    """Execute the PDAL pipeline."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f, indent=2)
        tmp = Path(f.name)
    try:
        log.debug("Pipeline:\n" + json.dumps(pipeline, indent=2))
        r = subprocess.run(
            ["pdal", "pipeline", str(tmp)], capture_output=True, text=True
        )
        if r.returncode != 0:
            log.error("PDAL pipeline failed:\n" + r.stderr)
            return False
        return True
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Remove extra-dims and reset Classification in a LAS/LAZ file."
    )
    parser.add_argument(
        "--in", dest="in_path", required=True, type=Path, help="Input LAS/LAZ file"
    )
    parser.add_argument(
        "--out", dest="out_path", required=True, type=Path, help="Output LAS/LAZ file"
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=[],
        help="List of extra-dims to remove (e.g., PredictedClassification entropy building ground)",
    )
    parser.add_argument(
        "--auto-drop",
        action="store_true",
        help="Auto-detect and remove typical prediction extra-dims (PredictedClassification, entropy, building, ground).",
    )
    parser.add_argument(
        "--class-value",
        type=int,
        default=1,
        help="Value to set for Classification field (default: 1 for 'Unclassified')",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    check_pdal()

    if not args.in_path.exists():
        log.error(f"Input file not found: {args.in_path}")
        sys.exit(1)

    dims_to_drop: Set[str] = set(args.drop)

    if args.auto_drop:
        present = pdal_list_dims(args.in_path)
        # typical set of extra-dims generated by inference
        typical = {
            "PredictedClassification",
            "entropy",
            "building",
            "ground",
            "vegetation",
            "water",
            "bridge",
            "unclassified",
            "lasting_above",
            "confidence",
        }
        dims_to_drop.update(typical.intersection(present))
        if dims_to_drop:
            log.info(
                f"Auto-detected extra dimensions to remove: {', '.join(sorted(dims_to_drop))}"
            )

    # Update pipeline to use custom class value
    pipeline = create_cleanup_pipeline(
        args.in_path, args.out_path, sorted(dims_to_drop)
    )

    # Modify the classification assignment with the custom value
    for stage in pipeline["pipeline"]:
        if stage.get("type") == "filters.assign":
            stage["assignment"] = f"Classification[:] = {args.class_value}"

    log.info(f"Processing: {args.in_path} -> {args.out_path}")
    if dims_to_drop:
        log.info(f"Removing dimensions: {', '.join(sorted(dims_to_drop))}")
    log.info(f"Setting Classification to: {args.class_value}")

    ok = run_pipeline(pipeline)

    if ok and args.out_path.exists():
        size_mb = args.out_path.stat().st_size / (1024**2)
        log.info(f"✅ Cleanup completed: {args.out_path} ({size_mb:.1f} MB)")
        if dims_to_drop:
            log.info("Removed extra-dims: " + ", ".join(sorted(dims_to_drop)))
        log.info(f"'Classification' field set to {args.class_value}.")

        # Show size reduction
        if args.in_path.exists():
            original_size_mb = args.in_path.stat().st_size / (1024**2)
            reduction_pct = (1 - size_mb / original_size_mb) * 100
            if reduction_pct > 0:
                log.info(
                    f"File size reduced by {reduction_pct:.1f}% ({original_size_mb:.1f} MB -> {size_mb:.1f} MB)"
                )

        sys.exit(0)
    else:
        log.error("❌ Operation failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()