"""
Custom preprocessing function for RGB-only LAS files (no NIR/Infrared)
=====================================================================

This is adapted from the original lidar_hd_pre_transform to work with
LAS files that have RGB but no Infrared channel.

Place this file in: src/pctl/points_pre_transform/rgb_only_transform.py
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
from torch_geometric.data import Data

COLORS_NORMALIZATION_MAX_VALUE = 255.0 * 256.0  # 65535 for 16-bit colors
RETURN_NUMBER_NORMALIZATION_MAX_VALUE = 7.0


def rgb_only_pre_transform(points):
    """
    Turn pdal points into torch-geometric Data object for RGB-only LAS files.

    This function is adapted from lidar_hd_pre_transform to handle LAS files
    that have RGB color channels but no Infrared/NIR channel.

    Args:
        points (np.ndarray): points loaded via PDAL

    Returns:
        Data: the point cloud formatted for deep learning training.
    """

    # Positions
    pos = np.asarray(
        [points["X"], points["Y"], points["Z"]], dtype=np.float32
    ).transpose()

    # Handle return number normalization
    occluded_points = points["ReturnNumber"] > 1
    points["ReturnNumber"] = (
        (points["ReturnNumber"]) / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )
    points["NumberOfReturns"] = (
        (points["NumberOfReturns"]) / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )

    # Ensure RGB color fields exist (should be there from colorization)
    for color in ["Red", "Green", "Blue"]:
        if color not in points.dtype.names:
            print(
                f"Warning: {color} channel not found. Creating fake {color} filled with 0."
            )
            fake_color = np.zeros(points.shape[0], dtype=np.float32)
            points = append_fields(
                points, color, fake_color, dtypes=np.float32, usemask=False
            )

    # Create synthetic Infrared channel based on RGB
    # Method 1: Use average of RGB as NIR approximation
    if "Infrared" not in points.dtype.names:
        print("Creating synthetic Infrared channel from RGB average...")
        # Synthetic NIR = weighted average emphasizing red and green (vegetation-sensitive)
        synthetic_nir = (
            0.4 * points["Red"] + 0.4 * points["Green"] + 0.2 * points["Blue"]
        ).astype(np.float32)
        points = append_fields(
            points, "Infrared", synthetic_nir, dtypes=np.float32, usemask=False
        )

    # Normalize colors
    available_colors = []
    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color in points.dtype.names:
            # Check if colors are already normalized (0-1) or need normalization (0-65535)
            color_max = points[color].max()
            if color_max > 1.0:
                # Colors are in 16-bit format, normalize to 0-1
                assert color_max <= COLORS_NORMALIZATION_MAX_VALUE, (
                    f"{color} max ({color_max}) too high! Expected <= {COLORS_NORMALIZATION_MAX_VALUE}"
                )
                points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE

            # Set occluded points colors to 0 (as in original function)
            points[color][occluded_points] = 0.0
            available_colors.append(color)
            print(
                f"Normalized {color} channel (range: {points[color].min():.3f} - {points[color].max():.3f})"
            )

    # Calculate RGB average (composite color channel)
    rgb_avg = np.zeros(points.shape[0], dtype=np.float32)
    if all(c in points.dtype.names for c in ["Red", "Green", "Blue"]):
        rgb_avg = (
            np.asarray(
                [points["Red"], points["Green"], points["Blue"]], dtype=np.float32
            )
            .transpose()
            .mean(axis=1)
        )
        print(
            f"Calculated RGB average (range: {rgb_avg.min():.3f} - {rgb_avg.max():.3f})"
        )

    # Calculate NDVI using synthetic infrared
    ndvi = np.zeros(points.shape[0], dtype=np.float32)
    if "Infrared" in points.dtype.names and "Red" in points.dtype.names:
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = (points["Infrared"] - points["Red"]) / (
            points["Infrared"] + points["Red"] + 1e-6
        )
        print(f"Calculated NDVI (range: {ndvi.min():.3f} - {ndvi.max():.3f})")

    # Build feature list
    x_list = [
        points["Intensity"],
        points["ReturnNumber"],
        points["NumberOfReturns"],
    ]
    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
    ]

    # Add color channels
    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color in points.dtype.names:
            x_list.append(points[color])
            x_features_names.append(color)

    # Add computed features
    x_list += [rgb_avg, ndvi]
    x_features_names += ["rgb_avg", "ndvi"]

    # Stack features
    x = np.stack(x_list, axis=0).transpose()

    # Get classification if available
    y = points["Classification"] if "Classification" in points.dtype.names else None

    # Create Data object
    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)

    print(
        f"Created Data object with {len(x_features_names)} features: {x_features_names}"
    )
    print(f"Feature matrix shape: {x.shape}")
    print(f"Position matrix shape: {pos.shape}")

    return data


def rgb_only_pre_transform_simple(points):
    """
    Simplified version that creates a minimal synthetic NIR channel.
    Use this if the main function has issues.
    """

    # Positions
    pos = np.asarray(
        [points["X"], points["Y"], points["Z"]], dtype=np.float32
    ).transpose()

    # Simple return number normalization
    points["ReturnNumber"] = (
        points["ReturnNumber"] / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )
    points["NumberOfReturns"] = (
        points["NumberOfReturns"] / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )

    # Simple color normalization (assume 16-bit colors)
    for color in ["Red", "Green", "Blue"]:
        if color in points.dtype.names and points[color].max() > 1.0:
            points[color] = points[color] / 65535.0

    # Create very simple synthetic NIR (just copy Green channel)
    if "Infrared" not in points.dtype.names:
        points = append_fields(
            points, "Infrared", points["Green"], dtypes=np.float32, usemask=False
        )

    # Minimal feature set
    x_list = [
        points["Intensity"],
        points["ReturnNumber"],
        points["NumberOfReturns"],
        points["Red"],
        points["Green"],
        points["Blue"],
        points["Infrared"],
        (points["Red"] + points["Green"] + points["Blue"]) / 3,  # RGB avg
        np.zeros(points.shape[0], dtype=np.float32),  # Dummy NDVI
    ]

    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
        "Red",
        "Green",
        "Blue",
        "Infrared",
        "rgb_avg",
        "ndvi",
    ]

    x = np.stack(x_list, axis=0).transpose()
    y = points["Classification"] if "Classification" in points.dtype.names else None

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)

    return data
