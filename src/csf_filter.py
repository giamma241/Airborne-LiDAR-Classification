import json
import os
import pdal


def run_csf(
    input_file: str,
    output_file: str,
    cloth_resolution: float = 0.5,
    rigidness: int = 2,
    time_step: float = 0.65,
) -> str:
    """Run PDAL Cloth Simulation Filter (CSF).

    Parameters
    ----------
    input_file: str
        Path to input LAS/LAZ file.
    output_file: str
        Destination file for filtered result.
    cloth_resolution: float, optional
        Resolution of the cloth used in simulation.
    rigidness: int, optional
        Rigidness of cloth (1-3).
    time_step: float, optional
        Integration time step for simulation.

    Returns
    -------
    str
        Path to the written file.
    """
    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_file},
            {
                "type": "filters.csf",
                "cloth_resolution": cloth_resolution,
                "rigidness": rigidness,
                "time_step": time_step,
            },
            {"type": "writers.las", "filename": output_file},
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()
    return os.path.abspath(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cloth Simulation Filter")
    parser.add_argument("input", help="Input LAS/LAZ file")
    parser.add_argument("output", help="Output LAS/LAZ file")
    parser.add_argument("--cloth-resolution", type=float, default=0.5)
    parser.add_argument("--rigidness", type=int, default=2)
    parser.add_argument("--time-step", type=float, default=0.65)
    args = parser.parse_args()

    run_csf(
        args.input,
        args.output,
        cloth_resolution=args.cloth_resolution,
        rigidness=args.rigidness,
        time_step=args.time_step,
    )
