import json
import logging
import os

import pdal

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def run_smrf(
    input_file: str,
    output_file: str,
    window: float = 18.0,
    slope: float = 0.15,
    threshold: float = 0.5,
    window_increment: float = 16.0,
) -> str:
    """
    Runs PDAL Simple Morphological Filter (SMRF) and saves the output.

    Params:
    - input_file: Path to input LAS/LAZ file
    - output_file: Destination file for filtered result
    - window: Initial window size for the filter
    - slope: Slope parameter for underlying terrain
    - threshold: Elevation threshold
    - window_increment: Increment applied to window size each step

    Returns:
    - Path to the written output file (absolute path)
    """
    logging.info("Preparing SMRF pipeline for input: %s", input_file)

    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_file},
            {
                "type": "filters.smrf",
                "scalar": window_increment,
                "slope": slope,
                "threshold": threshold,
                "window": window,
            },
            {"type": "writers.las", "filename": output_file},
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()

    abs_path = os.path.abspath(output_file)
    logging.info("SMRF filter applied successfully. Output saved to: %s", abs_path)
    return abs_path


if __name__ == "__main__":
    # Example usage (manual test)
    logging.info("This script is intended to be imported as a module.")
