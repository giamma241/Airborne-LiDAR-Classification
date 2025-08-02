import json
import logging
import os

import pdal

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def run_pmf(
    input_file: str,
    output_file: str,
    cell_size: float = 1.0,
    initial_distance: float = 1.0,
    max_window_size: int = 33,
    slope: float = 1.0,
) -> str:
    """
    Runs PDAL Progressive Morphological Filter (PMF) and saves the output.

    Params:
    - input_file: Path to input LAS/LAZ file
    - output_file: Destination file for filtered result
    - max_window_size: Maximum window size used by the filter
    - slope: Slope for terrain
    - initial_distance: Initial elevation distance
    - cell_size: Cell size for morphological operations

    Returns:
    - Path to the written output file (absolute path)
    """
    logging.info("Preparing PMF pipeline for input: %s", input_file)

    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_file},
            {
                "type": "filters.pmf",
                "cell_size": cell_size,
                "initial_distance": initial_distance,
                "max_window_size": max_window_size,
                "slope": slope,
            },
            {"type": "writers.las", "filename": output_file},
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()

    abs_path = os.path.abspath(output_file)
    logging.info("PMF filter applied successfully. Output saved to: %s", abs_path)
    return abs_path


if __name__ == "__main__":
    # Example usage (manual test)
    logging.info("This script is intended to be imported as a module.")
