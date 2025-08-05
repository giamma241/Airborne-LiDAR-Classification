#!/usr/bin/env python3
"""
Batch LAS File Colorization using Orthophotos
==============================================

This script colorizes LAS files using corresponding orthophotos.
For each Fabriano_XXXXXX.las file, it uses the corresponding Fabriano_XXXXXX.tif orthophoto.

Requirements:
- laspy
- rasterio
- numpy
- tqdm (for progress bars)
- pathlib

Install with: pip install laspy rasterio numpy tqdm
"""

import logging
from pathlib import Path
from typing import Tuple

import laspy
import numpy as np
import rasterio
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_world_file(tfw_path: Path) -> Tuple[float, float, float, float, float, float]:
    """
    Read a world file (.tfw) to get georeferencing parameters.

    Returns:
        tuple: (x_scale, y_rotation, x_rotation, y_scale, x_origin, y_origin)
    """
    with open(tfw_path, "r") as f:
        lines = [float(line.strip()) for line in f.readlines()]

    if len(lines) != 6:
        raise ValueError(f"World file {tfw_path} should have exactly 6 lines")

    return tuple(lines)


def pixel_to_world(
    col: int, row: int, world_params: Tuple[float, ...]
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to world coordinates using world file parameters.
    """
    x_scale, y_rotation, x_rotation, y_scale, x_origin, y_origin = world_params

    x = x_origin + col * x_scale + row * x_rotation
    y = y_origin + col * y_rotation + row * y_scale

    return x, y


def world_to_pixel(
    x: float, y: float, world_params: Tuple[float, ...]
) -> Tuple[int, int]:
    """
    Convert world coordinates to pixel coordinates.
    """
    x_scale, y_rotation, x_rotation, y_scale, x_origin, y_origin = world_params

    # Solve the inverse transformation
    det = x_scale * y_scale - x_rotation * y_rotation
    if abs(det) < 1e-10:
        raise ValueError("World file transformation is singular")

    dx = x - x_origin
    dy = y - y_origin

    col = (y_scale * dx - x_rotation * dy) / det
    row = (x_scale * dy - y_rotation * dx) / det

    return int(round(col)), int(round(row))


def colorize_las_file(
    las_path: Path, tif_path: Path, tfw_path: Path, output_path: Path
) -> bool:
    """
    Colorize a single LAS file using orthophoto.

    Args:
        las_path: Path to input LAS file
        tif_path: Path to orthophoto TIF file
        tfw_path: Path to world file (TFW)
        output_path: Path for output colorized LAS file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Processing {las_path.name}...")

        # Read LAS file
        logger.info("  Reading LAS file...")
        las = laspy.read(las_path)

        # Read orthophoto
        logger.info("  Reading orthophoto...")
        with rasterio.open(tif_path) as src:
            # Read RGB bands (assuming bands 1, 2, 3 are R, G, B)
            if src.count < 3:
                logger.error(
                    f"  Orthophoto {tif_path} has only {src.count} bands, need at least 3 (RGB)"
                )
                return False

            rgb_data = src.read([1, 2, 3])  # Read first 3 bands as RGB
            height, width = rgb_data.shape[1], rgb_data.shape[2]

            # Get transformation from rasterio
            transform = src.transform

        # Alternative: read world file if rasterio transform doesn't work well
        if tfw_path.exists():
            logger.info("  Reading world file...")
            world_params = read_world_file(tfw_path)
        else:
            # Use rasterio transform to create world file parameters
            world_params = (
                transform.a,  # x_scale
                transform.b,  # y_rotation
                transform.d,  # x_rotation
                transform.e,  # y_scale
                transform.c,  # x_origin
                transform.f,  # y_origin
            )

        # Extract point coordinates
        x_coords = np.array(las.x)
        y_coords = np.array(las.y)

        logger.info(f"  Colorizing {len(x_coords):,} points...")

        # Initialize RGB arrays
        red_values = np.zeros(len(x_coords), dtype=np.uint16)
        green_values = np.zeros(len(x_coords), dtype=np.uint16)
        blue_values = np.zeros(len(x_coords), dtype=np.uint16)

        # Process points in batches for memory efficiency
        batch_size = 100000
        num_batches = (len(x_coords) + batch_size - 1) // batch_size

        points_colored = 0
        points_outside = 0

        for batch_idx in tqdm(range(num_batches), desc="  Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(x_coords))

            batch_x = x_coords[start_idx:end_idx]
            batch_y = y_coords[start_idx:end_idx]

            # Convert world coordinates to pixel coordinates
            for i, (x, y) in enumerate(zip(batch_x, batch_y)):
                try:
                    col, row = world_to_pixel(x, y, world_params)

                    # Check if pixel is within image bounds
                    if 0 <= col < width and 0 <= row < height:
                        # Extract RGB values
                        red_values[start_idx + i] = rgb_data[0, row, col]
                        green_values[start_idx + i] = rgb_data[1, row, col]
                        blue_values[start_idx + i] = rgb_data[2, row, col]
                        points_colored += 1
                    else:
                        points_outside += 1
                        # Keep RGB as 0 for points outside image

                except Exception:
                    points_outside += 1
                    # Keep RGB as 0 for problematic points

        logger.info(
            f"  Colored {points_colored:,} points, {points_outside:,} points outside image"
        )

        # Create new LAS file with RGB data
        logger.info("  Creating colorized LAS file...")

        # Use a simpler approach that works with different laspy versions
        try:
            # Method 1: Try to create a copy and modify it
            new_las = laspy.LasData(las.header)

            # Copy all point data
            new_las.points = las.points.copy()

            # Add RGB data - handle different laspy versions
            try:
                new_las.red = red_values
                new_las.green = green_values
                new_las.blue = blue_values
            except Exception:
                # Alternative method for older laspy versions
                points_array = new_las.points_data
                if hasattr(points_array, "red"):
                    points_array.red = red_values
                    points_array.green = green_values
                    points_array.blue = blue_values
                else:
                    # Need to upgrade point format
                    logger.info("  Upgrading point format to support RGB...")

                    # Create new header with RGB support
                    if las.header.point_format.id in [0, 1]:
                        new_format_id = 2 if las.header.point_format.id == 0 else 3
                    else:
                        new_format_id = 7  # Modern format

                    new_header = laspy.LasHeader(
                        point_format=new_format_id, version=las.header.version
                    )

                    # Copy header properties
                    for attr in [
                        "x_scale_factor",
                        "y_scale_factor",
                        "z_scale_factor",
                        "x_offset",
                        "y_offset",
                        "z_offset",
                        "min",
                        "max",
                    ]:
                        if hasattr(las.header, attr):
                            setattr(new_header, attr, getattr(las.header, attr))

                    # Create new LAS with RGB-capable format
                    new_las = laspy.LasData(new_header)

                    # Copy coordinates
                    new_las.x = las.x
                    new_las.y = las.y
                    new_las.z = las.z

                    # Copy other standard attributes
                    for attr in [
                        "intensity",
                        "return_number",
                        "number_of_returns",
                        "scan_direction_flag",
                        "edge_of_flight_line",
                        "classification",
                        "scan_angle_rank",
                        "user_data",
                        "point_source_id",
                    ]:
                        if hasattr(las, attr):
                            try:
                                setattr(new_las, attr, getattr(las, attr))
                            except:
                                pass  # Skip if not compatible

                    # Copy GPS time if available
                    if hasattr(las, "gps_time"):
                        try:
                            new_las.gps_time = las.gps_time
                        except:
                            pass

                    # Add RGB
                    new_las.red = red_values
                    new_las.green = green_values
                    new_las.blue = blue_values

        except Exception as e:
            logger.error(f"  Error creating LAS structure: {e}")
            # Try completely different approach
            logger.info("  Trying alternative LAS creation method...")

            # Simple file-based approach
            temp_las = laspy.create(point_format=7, file_version=(1, 4))

            # Set header properties
            temp_las.header.x_scale_factor = las.header.x_scale_factor
            temp_las.header.y_scale_factor = las.header.y_scale_factor
            temp_las.header.z_scale_factor = las.header.z_scale_factor
            temp_las.header.x_offset = las.header.x_offset
            temp_las.header.y_offset = las.header.y_offset
            temp_las.header.z_offset = las.header.z_offset

            # Add points
            temp_las.x = las.x
            temp_las.y = las.y
            temp_las.z = las.z
            temp_las.red = red_values
            temp_las.green = green_values
            temp_las.blue = blue_values

            # Copy other attributes if possible
            for attr in [
                "intensity",
                "return_number",
                "number_of_returns",
                "classification",
                "scan_angle_rank",
                "user_data",
                "point_source_id",
            ]:
                if hasattr(las, attr) and hasattr(temp_las, attr):
                    try:
                        setattr(temp_las, attr, getattr(las, attr))
                    except:
                        pass

            new_las = temp_las

        # Write output file
        logger.info(f"  Writing to {output_path}...")
        new_las.write(output_path)

        logger.info(f"✅ Successfully processed {las_path.name}")
        return True

    except Exception as e:
        logger.error(f"❌ Error processing {las_path.name}: {str(e)}")
        return False


def batch_colorize(
    las_dir: Path,
    orthophoto_dir: Path,
    output_dir: Path,
    las_pattern: str = "Fabriano_*.las",
) -> None:
    """
    Batch colorize all LAS files in a directory.

    Args:
        las_dir: Directory containing LAS files
        orthophoto_dir: Directory containing orthophotos (TIF) and world files (TFW)
        output_dir: Directory for colorized output files
        las_pattern: Pattern to match LAS files
    """

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all LAS files
    las_files = list(las_dir.glob(las_pattern))
    logger.info(f"Found {len(las_files)} LAS files to process")

    if not las_files:
        logger.error(
            f"No LAS files found matching pattern '{las_pattern}' in {las_dir}"
        )
        return

    successful = 0
    failed = 0

    for las_path in las_files:
        # Construct corresponding orthophoto and world file paths
        base_name = las_path.stem  # e.g., "Fabriano_000001"
        tif_path = orthophoto_dir / f"{base_name}.tif"
        tfw_path = orthophoto_dir / f"{base_name}.tfw"
        output_path = output_dir / f"{base_name}_colorized.las"

        # Check if required files exist
        if not tif_path.exists():
            logger.error(f"❌ Orthophoto not found: {tif_path}")
            failed += 1
            continue

        if not tfw_path.exists():
            logger.warning(
                f"⚠️  World file not found: {tfw_path} (will try rasterio transform)"
            )

        # Skip if output already exists
        if output_path.exists():
            logger.info(f"⏭️  Output already exists: {output_path}")
            continue

        # Process file
        if colorize_las_file(las_path, tif_path, tfw_path, output_path):
            successful += 1
        else:
            failed += 1

    logger.info("\n=== BATCH PROCESSING COMPLETE ===")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    # Since we're running from the airborne-lidar-classification directory
    las_directory = Path("data/las_files")  # Directory with your LAS files
    orthophoto_directory = Path("data/orthophotos")  # Directory with TIF and TFW files
    output_directory = Path("data/colorized_las")  # Where to save colorized files

    # Alternative if running from a subdirectory:
    # las_directory = Path("../data/las_files")
    # orthophoto_directory = Path("../data/orthophotos")
    # output_directory = Path("../data/colorized_las")

    # Verify directories exist
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Looking for LAS directory at: {las_directory.resolve()}")
    logger.info(
        f"Looking for orthophoto directory at: {orthophoto_directory.resolve()}"
    )

    if not las_directory.exists():
        logger.error(f"LAS directory not found: {las_directory.resolve()}")
        logger.info("Available directories:")
        parent_dir = las_directory.parent
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}")
        exit(1)

    if not orthophoto_directory.exists():
        logger.error(
            f"Orthophoto directory not found: {orthophoto_directory.resolve()}"
        )
        logger.info("Available directories:")
        parent_dir = orthophoto_directory.parent
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}")
        exit(1)

    logger.info("=== BATCH LAS COLORIZATION ===")
    logger.info(f"LAS files: {las_directory}")
    logger.info(f"Orthophotos: {orthophoto_directory}")
    logger.info(f"Output: {output_directory}")

    # Run batch processing
    batch_colorize(las_directory, orthophoto_directory, output_directory)
