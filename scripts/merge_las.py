#!/usr/bin/env python3
"""
Merge Multiple LAS Files into One
=================================

This script provides two methods to merge multiple colorized LAS files:
1. Using laspy (Python-based)
2. Using PDAL (more robust, handles large files better)

Choose the method that works best for your system and file sizes.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_las_with_laspy(
    input_directory: Path, output_file: Path, pattern: str = "*_colorized.las"
) -> bool:
    """
    Merge LAS files using laspy (Python method).

    Args:
        input_directory: Directory containing colorized LAS files
        output_file: Output merged LAS file path
        pattern: Pattern to match input files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("=== MERGING WITH LASPY ===")

        # Find all colorized LAS files
        las_files = list(input_directory.glob(pattern))
        logger.info(f"Found {len(las_files)} LAS files to merge")

        if len(las_files) == 0:
            logger.error(
                f"No files found matching pattern '{pattern}' in {input_directory}"
            )
            return False

        # Read first file to get header template
        logger.info("Reading first file as template...")
        first_las = laspy.read(las_files[0])
        logger.info(f"Template file: {las_files[0].name}")
        logger.info(f"Template format: {first_las.header.point_format}")
        logger.info(f"Template points: {len(first_las.points):,}")

        # Calculate total points
        total_points = 0
        logger.info("Calculating total points...")
        for las_file in tqdm(las_files, desc="Scanning files"):
            las = laspy.read(las_file)
            total_points += len(las.points)

        logger.info(f"Total points to merge: {total_points:,}")

        # Check memory requirements
        estimated_memory_gb = total_points * 50 / (1024**3)  # Rough estimate
        logger.info(f"Estimated memory required: {estimated_memory_gb:.1f} GB")

        if estimated_memory_gb > 8:
            logger.warning(
                "⚠️  Large dataset detected. Consider using PDAL method instead."
            )
            response = input("Continue with laspy method? (y/n): ")
            if response.lower() != "y":
                return False

        # Create output LAS with same format as template
        logger.info("Creating merged LAS file...")
        merged_las = laspy.create(
            point_format=first_las.header.point_format,
            file_version=first_las.header.version,
        )

        # Copy header properties from template
        merged_las.header.x_scale_factor = first_las.header.x_scale_factor
        merged_las.header.y_scale_factor = first_las.header.y_scale_factor
        merged_las.header.z_scale_factor = first_las.header.z_scale_factor
        merged_las.header.x_offset = first_las.header.x_offset
        merged_las.header.y_offset = first_las.header.y_offset
        merged_las.header.z_offset = first_las.header.z_offset

        # Initialize arrays for all points
        logger.info("Initializing point arrays...")
        all_x = np.empty(total_points, dtype=np.float64)
        all_y = np.empty(total_points, dtype=np.float64)
        all_z = np.empty(total_points, dtype=np.float64)
        all_red = np.empty(total_points, dtype=np.uint16)
        all_green = np.empty(total_points, dtype=np.uint16)
        all_blue = np.empty(total_points, dtype=np.uint16)

        # Initialize other attributes if they exist
        other_attrs = {}
        for attr in [
            "intensity",
            "return_number",
            "number_of_returns",
            "classification",
            "scan_angle_rank",
            "user_data",
            "point_source_id",
            "gps_time",
        ]:
            if hasattr(first_las, attr):
                other_attrs[attr] = np.empty(
                    total_points, dtype=getattr(first_las, attr).dtype
                )

        # Merge all files
        current_index = 0
        logger.info("Merging files...")

        for las_file in tqdm(las_files, desc="Merging"):
            las = laspy.read(las_file)
            num_points = len(las.points)

            # Copy coordinates
            all_x[current_index : current_index + num_points] = las.x
            all_y[current_index : current_index + num_points] = las.y
            all_z[current_index : current_index + num_points] = las.z

            # Copy RGB
            all_red[current_index : current_index + num_points] = las.red
            all_green[current_index : current_index + num_points] = las.green
            all_blue[current_index : current_index + num_points] = las.blue

            # Copy other attributes
            for attr, array in other_attrs.items():
                if hasattr(las, attr):
                    array[current_index : current_index + num_points] = getattr(
                        las, attr
                    )

            current_index += num_points

        # Assign arrays to merged LAS
        logger.info("Finalizing merged file...")
        merged_las.x = all_x
        merged_las.y = all_y
        merged_las.z = all_z
        merged_las.red = all_red
        merged_las.green = all_green
        merged_las.blue = all_blue

        # Assign other attributes
        for attr, array in other_attrs.items():
            setattr(merged_las, attr, array)

        # Write output file
        logger.info(f"Writing merged file: {output_file}")
        merged_las.write(output_file)

        # Verify output
        verify_las = laspy.read(output_file)
        logger.info(f"✅ Successfully merged {len(las_files)} files")
        logger.info(f"✅ Output file: {output_file}")
        logger.info(f"✅ Total points: {len(verify_las.points):,}")
        logger.info(f"✅ File size: {output_file.stat().st_size / (1024**3):.2f} GB")

        return True

    except Exception as e:
        logger.error(f"❌ Error merging with laspy: {e}")
        return False


def merge_las_with_pdal(
    input_directory: Path, output_file: Path, pattern: str = "*_colorized.las"
) -> bool:
    """
    Merge LAS files using PDAL (more robust for large files).

    Args:
        input_directory: Directory containing colorized LAS files
        output_file: Output merged LAS file path
        pattern: Pattern to match input files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("=== MERGING WITH PDAL ===")

        # Check if PDAL is installed
        try:
            result = subprocess.run(
                ["pdal", "--version"], capture_output=True, text=True
            )
            logger.info(f"PDAL version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error(
                "❌ PDAL not found. Install with: conda install -c conda-forge pdal"
            )
            return False

        # Find all colorized LAS files
        las_files = list(input_directory.glob(pattern))
        logger.info(f"Found {len(las_files)} LAS files to merge")

        if len(las_files) == 0:
            logger.error(
                f"No files found matching pattern '{pattern}' in {input_directory}"
            )
            return False

        # Create PDAL pipeline configuration
        pipeline = {"pipeline": []}

        # Add all input files as readers
        for las_file in las_files:
            pipeline["pipeline"].append(
                {"type": "readers.las", "filename": str(las_file.absolute())}
            )

        # Add merge filter (optional - PDAL merges automatically)
        pipeline["pipeline"].append({"type": "filters.merge"})

        # Add writer
        pipeline["pipeline"].append(
            {
                "type": "writers.las",
                "filename": str(output_file.absolute()),
                # No compression - standard LAS format
            }
        )

        # Save pipeline to temporary file
        pipeline_file = input_directory / "merge_pipeline.json"
        with open(pipeline_file, "w") as f:
            json.dump(pipeline, f, indent=2)

        logger.info(f"Created PDAL pipeline: {pipeline_file}")
        logger.info("Running PDAL merge...")

        # Run PDAL pipeline
        cmd = ["pdal", "pipeline", str(pipeline_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ PDAL merge completed successfully")

            # Clean up pipeline file
            pipeline_file.unlink()

            # Verify output
            if output_file.exists():
                file_size_gb = output_file.stat().st_size / (1024**3)
                logger.info(f"✅ Output file: {output_file}")
                logger.info(f"✅ File size: {file_size_gb:.2f} GB")

                # Get point count using laspy
                try:
                    verify_las = laspy.read(output_file)
                    logger.info(f"✅ Total points: {len(verify_las.points):,}")
                except:
                    logger.info("✅ File created (could not verify point count)")

                return True
            else:
                logger.error("❌ Output file was not created")
                return False
        else:
            logger.error(f"❌ PDAL failed: {result.stderr}")
            pipeline_file.unlink()  # Clean up
            return False

    except Exception as e:
        logger.error(f"❌ Error merging with PDAL: {e}")
        return False


def get_file_info(directory: Path, pattern: str = "*_colorized.las") -> None:
    """
    Display information about files to be merged.
    """
    las_files = list(directory.glob(pattern))

    if not las_files:
        logger.error(f"No files found matching pattern '{pattern}'")
        return

    logger.info("\n=== FILES TO MERGE ===")
    total_points = 0
    total_size = 0

    for las_file in las_files:
        try:
            las = laspy.read(las_file)
            points = len(las.points)
            size_mb = las_file.stat().st_size / (1024**2)

            total_points += points
            total_size += size_mb

            logger.info(f"{las_file.name}: {points:,} points, {size_mb:.1f} MB")
        except Exception as e:
            logger.error(f"Could not read {las_file.name}: {e}")

    logger.info(f"\nTOTAL: {total_points:,} points, {total_size:.1f} MB")
    logger.info(f"Estimated merged size: {total_size:.1f} MB")


if __name__ == "__main__":
    # Configuration
    input_dir = Path("data/colorized_las")
    output_file = Path("data/merged_colorized.las")

    # Display file information
    get_file_info(input_dir)

    # Choose method
    print("\nChoose merge method:")
    print("1. laspy (Python-based, good for smaller datasets)")
    print("2. PDAL (more robust, better for large datasets)")
    print("3. Show file info only")

    try:
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            success = merge_las_with_laspy(input_dir, output_file)
        elif choice == "2":
            success = merge_las_with_pdal(input_dir, output_file)
        elif choice == "3":
            logger.info("File information displayed above.")
            sys.exit(0)
        else:
            logger.error("Invalid choice")
            sys.exit(1)

        if success:
            logger.info("\n🎉 Merge completed successfully!")
            logger.info(f"Merged file: {output_file}")
        else:
            logger.error("\n❌ Merge failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Merge cancelled by user")

    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
