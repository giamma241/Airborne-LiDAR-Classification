import logging

import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def compute_roughness(x, y, z, grid_size=1.0):
    """
    Computes surface roughness as the std deviation of Z in grid cells.

    Parameters:
    - x, y, z: 1D numpy arrays of coordinates
    - grid_size: resolution of grid in meters

    Returns:
    - mean and std of cell-wise roughness
    """
    logging.info("Computing roughness with grid size = %.2f", grid_size)

    xi = ((x - x.min()) / grid_size).astype(int)
    yi = ((y - y.min()) / grid_size).astype(int)

    cell_dict = {}
    for i in range(len(z)):
        key = (xi[i], yi[i])
        cell_dict.setdefault(key, []).append(z[i])

    roughness_vals = [np.std(pts) for pts in cell_dict.values() if len(pts) >= 3]

    if not roughness_vals:
        logging.warning("No valid cells found for roughness computation.")
        return None

    mean_roughness = np.mean(roughness_vals)
    std_roughness = np.std(roughness_vals)

    logging.info("Mean roughness: %.3f, Std dev: %.3f", mean_roughness, std_roughness)
    return mean_roughness, std_roughness


def compute_local_slope(x, y, z, radius=1.0):
    """
    Computes local slope using a k-d tree in a given radius neighborhood.

    Parameters:
    - x, y, z: 1D numpy arrays of coordinates
    - radius: search radius in meters

    Returns:
    - slopes: numpy array of slope values (degrees)
    """
    logging.info("Computing slope using neighborhood radius = %.2f", radius)

    coords = np.vstack((x, y)).T
    tree = cKDTree(coords)

    slopes = np.zeros_like(z)
    for i, (pt, zi) in enumerate(zip(coords, z)):
        idx = tree.query_ball_point(pt, r=radius)
        if len(idx) < 3:
            slopes[i] = 0
            continue
        dz = z[idx] - zi
        dx = np.linalg.norm(coords[idx] - pt, axis=1)
        local_slope = np.rad2deg(np.arctan2(np.abs(dz), dx + 1e-6))
        slopes[i] = np.mean(local_slope)

    logging.info("Computed slope for %d points.", len(slopes))
    return slopes
