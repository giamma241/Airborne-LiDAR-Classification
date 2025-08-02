import json
import logging
import os

import pdal

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def run_csf(
    input_file: str,
    output_file: str,
    rigidness: int = 2,
    step: float = 0.65,
) -> str:
    """
    Runs PDAL Cloth Simulation Filter (CSF) and saves the output.

    Params:
    - input_file: Path to input LAS/LAZ file
    - output_file: Destination file for filtered result
    - cloth_resolution: Resolution of the cloth used in simulation
    - rigidness: Rigidness of cloth (1-3)
    - time_step: Integration time step for simulation

    Returns:
    - Path to the written output file (absolute path)
    """
    logging.info("Preparing CSF pipeline for input: %s", input_file)

    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_file},
            {
                "type": "filters.csf",
                "rigidness": rigidness,
                "step": step,
            },
            {"type": "writers.las", "filename": output_file},
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()

    abs_path = os.path.abspath(output_file)
    logging.info("CSF filter applied successfully. Output saved to: %s", abs_path)
    return abs_path


if __name__ == "__main__":
    # Example usage (manual test)
    logging.info("This script is intended to be imported as a module.")
