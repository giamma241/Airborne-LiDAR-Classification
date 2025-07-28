"""
Denoising functions for LiDAR data using statistical outlier removal.
"""


def statistical_denoise(pcd, nb_neighbors=20, std_ratio=2.0):
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return pcd
