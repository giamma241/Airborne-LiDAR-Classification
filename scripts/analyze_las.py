#!/usr/bin/env python3
"""
LAS File Analysis for Myria3D Inference
=======================================

This notebook analyzes your LAS file to understand its structure and
compatibility with Myria3D requirements.
"""

from collections import Counter

import laspy
import numpy as np

# File path to your LAS file (adjusted for notebook being in notebooks/ folder)
las_file_path = "../data/colorized_las/merged_colorized.las"

print("=== LAS FILE ANALYSIS ===\n")

try:
    # Read the LAS file
    print("📂 Loading LAS file...")
    las = laspy.read(las_file_path)
    print("✅ LAS file loaded successfully!\n")

    # Basic file information
    print("=== BASIC FILE INFORMATION ===")
    print(f"File path: {las_file_path}")
    print(f"Number of points: {len(las.points):,}")
    print(f"Point format ID: {las.header.point_format}")
    print(f"LAS version: {las.header.version}")
    # Calculate file size more robustly
    try:
        file_size_gb = (
            las.header.point_count * las.header.point_data_record_length / (1024**3)
        )
        print(f"File size: {file_size_gb:.2f} GB")
    except AttributeError:
        # Alternative method for newer laspy versions
        try:
            import os

            file_size_gb = os.path.getsize(las_file_path) / (1024**3)
            print(f"File size: {file_size_gb:.2f} GB")
        except:
            print("File size: Could not determine")
    print()

    # Header information
    print("=== HEADER INFORMATION ===")
    print(
        f"Min coordinates: X={las.header.min[0]:.2f}, Y={las.header.min[1]:.2f}, Z={las.header.min[2]:.2f}"
    )
    print(
        f"Max coordinates: X={las.header.max[0]:.2f}, Y={las.header.max[1]:.2f}, Z={las.header.max[2]:.2f}"
    )
    # print(f"Scale factors: X={las.header.x_scale_factor}, Y={las.header.y_scale_factor}, Z={las.header.z_scale_factor}")
    # print(f"Offset: X={las.header.x_offset}, Y={las.header.y_offset}, Z={las.header.z_offset}")
    print()

    # Check for coordinate system (EPSG)
    print("=== COORDINATE SYSTEM ===")
    try:
        # Check for VLRs (Variable Length Records)
        if hasattr(las, "vlrs") and las.vlrs:
            print(f"Found {len(las.vlrs)} VLR(s)")
            for i, vlr in enumerate(las.vlrs):
                print(f"VLR {i}: {vlr}")
                # Look for coordinate system info
                if hasattr(vlr, "record_id"):
                    if vlr.record_id in [34735, 34736, 34737]:  # GeoTIFF keys
                        print("  → GeoTIFF coordinate system record found")
        else:
            print("No VLRs found")

        # Check global encoding
        if hasattr(las.header, "global_encoding"):
            print(f"Global encoding: {las.header.global_encoding}")

        # Try to extract EPSG if available
        try:
            if hasattr(las, "header") and hasattr(las.header, "evlrs"):
                for evlr in las.header.evlrs:
                    if "EPSG" in str(evlr) or "WKT" in str(evlr):
                        print(f"Extended VLR with coordinate info: {evlr}")
        except:
            pass

        print("⚠️  If no EPSG found, you'll need: datamodule.epsg=YOUR_EPSG_CODE")
    except Exception as e:
        print(f"Could not read coordinate system info: {e}")
    print()

    # Available dimensions/attributes
    print("=== AVAILABLE DIMENSIONS ===")
    try:
        # Try newer laspy API first
        if hasattr(las.header, "point_format"):
            dimensions = list(las.header.point_format.dimension_names)
        else:
            # Fallback to older API
            dimensions = list(las.point_format.dimension_names)
        print(f"Available dimensions: {dimensions}")
    except AttributeError:
        # Manual dimension detection
        dimensions = []
        standard_dims = [
            "X",
            "Y",
            "Z",
            "intensity",
            "return_number",
            "number_of_returns",
            "scan_direction_flag",
            "edge_of_flight_line",
            "classification",
            "scan_angle_rank",
            "user_data",
            "point_source_id",
        ]
        color_dims = ["red", "green", "blue"]
        extra_dims = ["gps_time", "nir", "near_infrared"]

        for dim in standard_dims + color_dims + extra_dims:
            if hasattr(las, dim.lower()):
                dimensions.append(dim)

        print(f"Available dimensions (detected): {dimensions}")
    print()

    # Check for required attributes
    print("=== MYRIA3D REQUIREMENTS CHECK ===")

    # Check for coordinates
    has_xyz = all(dim in dimensions for dim in ["X", "Y", "Z"])
    print(f"✅ XYZ coordinates: {has_xyz}")

    # Check for RGB
    has_rgb = all(dim in dimensions for dim in ["red", "green", "blue"])
    print(f"{'✅' if has_rgb else '❌'} RGB colors: {has_rgb}")
    if has_rgb:
        print(f"   Red range: {las.red.min()} - {las.red.max()}")
        print(f"   Green range: {las.green.min()} - {las.green.max()}")
        print(f"   Blue range: {las.blue.min()} - {las.blue.max()}")

    # Check for Infrared/NIR
    has_nir = "nir" in dimensions or "near_infrared" in dimensions
    print(f"{'✅' if has_nir else '❌'} Near Infrared: {has_nir}")
    if has_nir:
        nir_attr = "nir" if "nir" in dimensions else "near_infrared"
        nir_data = getattr(las, nir_attr)
        print(f"   NIR range: {nir_data.min()} - {nir_data.max()}")

    # Check for intensity
    has_intensity = "intensity" in dimensions
    print(f"{'✅' if has_intensity else '❌'} Intensity: {has_intensity}")
    if has_intensity:
        print(f"   Intensity range: {las.intensity.min()} - {las.intensity.max()}")

    # Check for return information
    has_returns = all(
        dim in dimensions for dim in ["return_number", "number_of_returns"]
    )
    print(f"{'✅' if has_returns else '❌'} Return information: {has_returns}")
    if has_returns:
        print(f"   Return numbers: {set(las.return_number)}")
        print(f"   Max returns: {set(las.number_of_returns)}")

    # Check for classification
    has_classification = "classification" in dimensions
    print(
        f"{'✅' if has_classification else '❌'} Classification: {has_classification}"
    )
    if has_classification:
        class_counts = Counter(las.classification)
        print(f"   Classification codes present: {sorted(class_counts.keys())}")
        print("   Point distribution by class:")
        for class_code, count in sorted(class_counts.items()):
            percentage = (count / len(las.points)) * 100
            print(f"     Class {class_code}: {count:,} points ({percentage:.1f}%)")

    print()

    # Myria3D specific requirements
    print("=== MYRIA3D COMPATIBILITY ===")

    # Expected input features for Myria3D (based on lidar_hd_pre_transform)
    expected_features = [
        "X",
        "Y",
        "Z",
        "red",
        "green",
        "blue",
        "nir",
        "intensity",
        "return_number",
        "number_of_returns",
    ]

    print("Expected features for Myria3D:")
    missing_features = []
    for feature in expected_features:
        if feature in dimensions or (
            feature == "nir" and "near_infrared" in dimensions
        ):
            print(f"   ✅ {feature}")
        else:
            print(f"   ❌ {feature} - MISSING")
            missing_features.append(feature)

    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
        print("   This might cause issues with the default lidar_hd_pre_transform")
        print("   You may need to create a custom preprocessing function")
    else:
        print("\n✅ All expected features are present!")

    print()

    # Data quality checks
    print("=== DATA QUALITY CHECKS ===")

    # Check for NaN/invalid values
    if has_xyz:
        nan_count = (
            np.isnan(las.x).sum() + np.isnan(las.y).sum() + np.isnan(las.z).sum()
        )
        print(f"NaN coordinates: {nan_count}")

    # Check coordinate ranges (reasonable values)
    print("Coordinate extent:")
    print(f"   X: {las.header.max[0] - las.header.min[0]:.2f} units")
    print(f"   Y: {las.header.max[1] - las.header.min[1]:.2f} units")
    print(f"   Z: {las.header.max[2] - las.header.min[2]:.2f} units")

    # Sample a few points for inspection
    print("\n=== SAMPLE POINTS ===")
    sample_size = min(5, len(las.points))
    print(f"First {sample_size} points:")

    for i in range(sample_size):
        point_info = f"Point {i}: X={las.x[i]:.2f}, Y={las.y[i]:.2f}, Z={las.z[i]:.2f}"
        if has_rgb:
            point_info += f", RGB=({las.red[i]}, {las.green[i]}, {las.blue[i]})"
        if has_intensity:
            point_info += f", I={las.intensity[i]}"
        if has_classification:
            point_info += f", Class={las.classification[i]}"
        print(f"   {point_info}")

    print()

    # Recommendations
    print("=== RECOMMENDATIONS ===")

    if missing_features:
        print("🔧 REQUIRED ACTIONS:")
        print("   1. You need to create a custom preprocessing function")
        print("   2. Place it in 'src/pctl/points_pre_transform/' directory")
        print("   3. Reference it with datamodule.points_pre_transform parameter")
        print()

    print("🚀 SUGGESTED COMMAND:")
    command = f"""python run.py task.task_name=predict \\
    predict.src_las="{las_file_path}" \\
    predict.output_dir="data/test" \\
    +ckpt_path="assets/proto151_V2.0_epoch_100_Myria3DV3.1.0.ckpt" \\
    predict.gpus=1 \\
    logger.comet.disabled=true"""

    # Add EPSG if we couldn't find it
    command += " \\\n    datamodule.epsg=YOUR_EPSG_CODE  # Replace with actual EPSG"

    if missing_features:
        command += " \\\n    datamodule.points_pre_transform=your_custom_transform"

    print(command)

except FileNotFoundError:
    print(f"❌ File not found: {las_file_path}")
    print("Please check the file path and try again.")

except Exception as e:
    print(f"❌ Error reading LAS file: {e}")
    print("Make sure you have laspy installed: pip install laspy")

print("\n" + "=" * 50)
print("Analysis complete! Review the results above.")
