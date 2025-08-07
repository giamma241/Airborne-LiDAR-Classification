#!/usr/bin/env python3
"""
Fixed PDAL-based LAS Colorization (RGB Only)
===========================================

This simplified version just adds RGB colors using PDAL, without synthetic infrared.
We'll handle the infrared channel separately if needed.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_pdal_installation() -> None:
    """Check if PDAL is installed and available."""
    try:
        result = subprocess.run(
            ["pdal", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info(f"PDAL found: {result.stdout.strip()}")
        else:
            raise RuntimeError("PDAL not responding correctly")
    except FileNotFoundError:
        logger.error("PDAL not found. Install with: conda install -c conda-forge pdal")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error checking PDAL: {e}")
        sys.exit(1)


def create_rgb_colorization_pipeline(
    input_las: Path, output_las: Path, rgb_image: Path
) -> dict:
    """
    Create PDAL pipeline for RGB-only colorization.

    Args:
        input_las: Input LAS file path
        output_las: Output LAS file path
        rgb_image: RGB orthophoto path

    Returns:
        dict: PDAL pipeline configuration
    """

    pipeline = {
        "pipeline": [
            # Reader
            {"type": "readers.las", "filename": str(input_las.absolute())},
            # RGB Colorization
            {
                "type": "filters.colorization",
                "raster": str(rgb_image.absolute()),
                # Map raster bands to LAS color channels with scaling
                "dimensions": "Red:1:256.0,Green:2:256.0,Blue:3:256.0",
            },
            # Writer with RGB support
            {
                "type": "writers.las",
                "filename": str(output_las.absolute()),
                "minor_version": "4",  # LAS 1.4
                "dataformat_id": "3",  # Point format 3 (supports RGB + time)
                "forward": "all",  # Forward all dimensions from input
            },
        ]
    }

    return pipeline


def colorize_single_file_rgb(
    input_las: Path, rgb_image: Path, output_las: Path
) -> bool:
    """
    Colorize a single LAS file with RGB orthophoto only.

    Args:
        input_las: Input LAS file
        rgb_image: RGB orthophoto (TIF)
        output_las: Output colorized LAS file

    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"Colorizing {input_las.name} with {rgb_image.name}")

        # Verify input files exist
        if not input_las.exists():
            logger.error(f"Input LAS file not found: {input_las}")
            return False

        if not rgb_image.exists():
            logger.error(f"RGB image not found: {rgb_image}")
            return False

        # Create pipeline
        pipeline = create_rgb_colorization_pipeline(input_las, output_las, rgb_image)

        # Create temporary pipeline file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pipeline, f, indent=2)
            pipeline_file = Path(f.name)

        try:
            # Execute PDAL pipeline
            logger.info("Running PDAL RGB colorization pipeline...")
            cmd = ["pdal", "pipeline", str(pipeline_file)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully colorized: {output_las}")

                # Verify output file was created
                if output_las.exists():
                    file_size_mb = output_las.stat().st_size / (1024**2)
                    logger.info(f"Output file size: {file_size_mb:.1f} MB")
                    return True
                else:
                    logger.error("Output file was not created")
                    return False
            else:
                logger.error(f"PDAL pipeline failed: {result.stderr}")
                logger.info("Pipeline used:")
                logger.info(json.dumps(pipeline, indent=2))
                return False

        finally:
            # Clean up temporary pipeline file
            pipeline_file.unlink()

    except subprocess.TimeoutExpired:
        logger.error(f"PDAL pipeline timed out for {input_las.name}")
        return False
    except Exception as e:
        logger.error(f"Error colorizing {input_las.name}: {e}")
        return False


def batch_colorize_rgb(
    las_dir: Path, orthophoto_dir: Path, output_dir: Path
) -> Tuple[int, int]:
    """
    Batch colorize multiple LAS files with RGB only.

    Args:
        las_dir: Directory containing LAS files
        orthophoto_dir: Directory containing orthophotos
        output_dir: Output directory for colorized files

    Returns:
        tuple: (successful_count, failed_count)
    """

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find LAS files
    las_files = list(las_dir.glob("Fabriano_*.las"))
    logger.info(f"Found {len(las_files)} LAS files to colorize")

    if not las_files:
        logger.error(f"No LAS files found matching Fabriano_*.las in {las_dir}")
        return 0, 0

    successful = 0
    failed = 0

    for las_file in las_files:
        # Find corresponding orthophoto
        base_name = las_file.stem  # e.g., "Fabriano_000001"

        # Try different image extensions
        rgb_image = None
        for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
            candidate = orthophoto_dir / f"{base_name}{ext}"
            if candidate.exists():
                rgb_image = candidate
                break

        if not rgb_image:
            logger.error(f"No orthophoto found for {las_file.name}")
            failed += 1
            continue

        # Define output path
        output_las = output_dir / f"{base_name}_rgb.las"

        # Skip if already exists
        if output_las.exists():
            logger.info(f"⏭️  Output already exists: {output_las.name}")
            successful += 1  # Count as successful
            continue

        # Colorize
        if colorize_single_file_rgb(las_file, rgb_image, output_las):
            successful += 1
        else:
            failed += 1

    logger.info("\n=== RGB COLORIZATION COMPLETE ===")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")

    return successful, failed


def merge_rgb_files(colorized_dir: Path, output_file: Path) -> bool:
    """
    Merge RGB colorized files using PDAL.
    """
    try:
        # Find colorized files
        colorized_files = list(colorized_dir.glob("*_rgb.las"))
        logger.info(f"Found {len(colorized_files)} RGB files to merge")

        if len(colorized_files) < 2:
            logger.error("Need at least 2 files to merge")
            return False

        # Create merge pipeline
        pipeline = {"pipeline": []}

        # Add all input files
        for las_file in colorized_files:
            pipeline["pipeline"].append(
                {"type": "readers.las", "filename": str(las_file.absolute())}
            )

        # Add merge filter
        pipeline["pipeline"].append({"type": "filters.merge"})

        # Add writer
        pipeline["pipeline"].append(
            {
                "type": "writers.las",
                "filename": str(output_file.absolute()),
                "minor_version": "4",
                "dataformat_id": "3",  # RGB format
                "forward": "all",
            }
        )

        # Execute pipeline
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pipeline, f, indent=2)
            pipeline_file = Path(f.name)

        try:
            logger.info("Merging RGB colorized files...")
            cmd = ["pdal", "pipeline", str(pipeline_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                logger.info(f"✅ Successfully merged to: {output_file}")
                file_size_gb = output_file.stat().st_size / (1024**3)
                logger.info(f"Merged file size: {file_size_gb:.2f} GB")
                return True
            else:
                logger.error(f"Merge failed: {result.stderr}")
                return False

        finally:
            pipeline_file.unlink()

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return False


def main():
    """Main execution function."""

    # Check PDAL installation
    check_pdal_installation()

    # Configuration
    las_directory = Path("data/las_files")
    orthophoto_directory = Path("data/orthophotos")
    colorized_directory = Path("data/colorized_las_rgb")
    merged_file = Path("data/merged_rgb.las")

    print("=== PDAL RGB COLORIZATION ===")
    print(f"LAS files: {las_directory}")
    print(f"Orthophotos: {orthophoto_directory}")
    print(f"Output: {colorized_directory}")

    # Step 1: RGB colorization
    print("\n--- Step 1: Adding RGB colors ---")
    successful, failed = batch_colorize_rgb(
        las_dir=las_directory,
        orthophoto_dir=orthophoto_directory,
        output_dir=colorized_directory,
    )

    if successful == 0:
        logger.error("No files were successfully colorized. Exiting.")
        return

    # Step 2: Merge files
    print("\n--- Step 2: Merging RGB files ---")
    response = input("Merge RGB colorized files into single LAS? (y/n): ")

    if response.lower() == "y":
        success = merge_rgb_files(
            colorized_dir=colorized_directory, output_file=merged_file
        )

        if success:
            print(f"\n🎉 RGB colorization complete! Merged file: {merged_file}")
            print("\nNext steps:")
            print("1. Use your custom RGB-only preprocessing function")
            print("2. Or add synthetic infrared channel separately")
            print("\nMyria3D command:")
            print(
                f'python run.py task.task_name=predict predict.src_las="{merged_file}" datamodule.points_pre_transform._args_[0]="${{get_method:src.pctl.points_pre_transform.rgb_only_transform.rgb_only_pre_transform}}" datamodule.epsg=32633 predict.gpus=1 logger.comet.disabled=true predict.output_dir="data/test" +ckpt_path="assets/proto151_V2.0_epoch_100_Myria3DV3.1.0.ckpt"'
            )
        else:
            print(
                f"\n⚠️  Merge failed, but individual RGB files are available in: {colorized_directory}"
            )
    else:
        print(
            f"\n✅ RGB colorization complete! Files available in: {colorized_directory}"
        )


if __name__ == "__main__":
    main()
