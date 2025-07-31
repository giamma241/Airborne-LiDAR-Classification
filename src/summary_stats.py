# File: src/evaluation/summary_stats.py

import logging

import laspy
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)

def summarize_las(source):
    """
    Summarizes LAS data whether from laspy.LasData or PDAL numpy array.

    Params:
    - source: laspy.LasData OR numpy structured array from PDAL
    """
    # Detect type
    is_laspy = isinstance(source, laspy.LasData)

    if is_laspy:
        las = source
        logging.info("LAS summary")
        logging.info("Version LAS: %s", las.header.version)
        logging.info("Format points: %s", las.header.point_format)
        logging.info("Total points: %d", len(las.points))
        logging.info("Bounding Box:")
        logging.info("  X: %.2f → %.2f", las.header.mins[0], las.header.maxs[0])
        logging.info("  Y: %.2f → %.2f", las.header.mins[1], las.header.maxs[1])
        logging.info("  Z: %.2f → %.2f", las.header.mins[2], las.header.maxs[2])

        z = las.z
        intensity = las.intensity
        classification = las.classification

        area = (las.header.maxs[0] - las.header.mins[0]) * (
            las.header.maxs[1] - las.header.mins[1]
        )

    else:
        arr = source
        logging.info("PDAL array summary")
        logging.info("Total points: %d", len(arr))

        x = arr["X"] * arr["X"].dtype.type(0.01)  # Assume scale = 0.01
        y = arr["Y"] * arr["Y"].dtype.type(0.01)
        z = arr["Z"] * arr["Z"].dtype.type(0.01)
        intensity = arr["Intensity"]
        classification = arr["Classification"]

        logging.info("Bounding Box:")
        logging.info("  X: %.2f → %.2f", x.min(), x.max())
        logging.info("  Y: %.2f → %.2f", y.min(), y.max())
        logging.info("  Z: %.2f → %.2f", z.min(), z.max())

        area = (x.max() - x.min()) * (y.max() - y.min())

    density = len(z) / area
    logging.info("Avg density (points/m²): %.2f", density)

    # Stats
    logging.info("Altitude (Z): min = %.2f, max = %.2f", z.min(), z.max())
    logging.info("Intensity: min = %d, max = %d", intensity.min(), intensity.max())

    classes, counts = np.unique(classification, return_counts=True)
    logging.info("Class counts:")
    for c, count in zip(classes, counts):
        logging.info("  Class %d: %d points", c, count)

# Example usage:
# from src.evaluation.summary_stats import summarize_las
# summarize_las(laspy.read("../data/processed/file.las"))
# summarize_las(pipeline.arrays[0])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize LAS or PDAL array")
    parser.add_argument("input", help="Path to .las or .laz file")
    args = parser.parse_args()

    las = laspy.read(args.input)
    summarize_las(las)