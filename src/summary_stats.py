import laspy
import numpy as np


def summarize_las(source):
    """
    Summarizes LAS data whether from laspy.LasData or PDAL numpy array.

    Parameters:
    - source: laspy.LasData OR numpy structured array from PDAL
    """
    # Detect type
    is_laspy = isinstance(source, laspy.LasData)

    if is_laspy:
        las = source
        print("LAS summary")
        print("Version LAS:", las.header.version)
        print("Format points:", las.header.point_format)
        print("Total points:", len(las.points))
        print("Bounding Box:")
        print("  X:", las.header.mins[0], "→", las.header.maxs[0])
        print("  Y:", las.header.mins[1], "→", las.header.maxs[1])
        print("  Z:", las.header.mins[2], "→", las.header.maxs[2])

        z = las.z
        intensity = las.intensity
        classification = las.classification

        area = (las.header.maxs[0] - las.header.mins[0]) * (
            las.header.maxs[1] - las.header.mins[1]
        )

    else:
        arr = source
        print("PDAL array summary")
        print("Total points:", len(arr))

        x = arr["X"] * arr["X"].dtype.type(0.01)  # Assume scale = 0.01
        y = arr["Y"] * arr["Y"].dtype.type(0.01)
        z = arr["Z"] * arr["Z"].dtype.type(0.01)
        intensity = arr["Intensity"]
        classification = arr["Classification"]

        print("Bounding Box:")
        print("  X:", x.min(), "→", x.max())
        print("  Y:", y.min(), "→", y.max())
        print("  Z:", z.min(), "→", z.max())

        area = (x.max() - x.min()) * (y.max() - y.min())

    density = len(z) / area
    print("\nAvg density (points/m²):", round(density, 2))

    # Stats
    print("\nAltitude (Z): min =", z.min(), ", max =", z.max())
    print("Intensity: min =", intensity.min(), ", max =", intensity.max())

    classes, counts = np.unique(classification, return_counts=True)
    print("\nClass counts:")
    for c, count in zip(classes, counts):
        print(f"  Class {c}: {count} points")


# Example usage:
# from src.evaluation.summary_stats import summarize_las
# summarize_las(laspy.read("../data/processed/file.las"))
# summarize_las(pipeline.arrays[0])