import json
import pdal
import os
from typing import Optional


def run_pmf(
    input_file: str,
    output_file: str,
    max_window_size: int = 33,
    slope: float = 1.0,
    initial_distance: float = 0.5,
    cell_size: float = 1.0,
) -> str:
    """Run PDAL progressive morphological filter.

    Parameters
    ----------
    input_file: str
        Path to input LAS/LAZ file.
    output_file: str
        Destination file for filtered result.
    max_window_size: int, optional
        Maximum window size used by the filter.
    slope: float, optional
        Slope for terrain.
    initial_distance: float, optional
        Initial elevation distance.
    cell_size: float, optional
        Cell size for morphological operations.

    Returns
    -------
    str
        Path to the written file.
    """
    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_file},
            {
                "type": "filters.pmf",
                "max_window_size": max_window_size,
                "slope": slope,
                "initial_distance": initial_distance,
                "cell_size": cell_size,
            },
            {"type": "writers.las", "filename": output_file},
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()
    return os.path.abspath(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Progressive Morphological Filter")
    parser.add_argument("input", help="Input LAS/LAZ file")
    parser.add_argument("output", help="Output LAS/LAZ file")
    parser.add_argument("--max-window-size", type=int, default=33)
    parser.add_argument("--slope", type=float, default=1.0)
    parser.add_argument("--initial-distance", type=float, default=0.5)
    parser.add_argument("--cell-size", type=float, default=1.0)
    args = parser.parse_args()

    run_pmf(
        args.input,
        args.output,
        max_window_size=args.max_window_size,
        slope=args.slope,
        initial_distance=args.initial_distance,
        cell_size=args.cell_size,
    )
