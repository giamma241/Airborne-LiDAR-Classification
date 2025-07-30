import json
import os
import pdal


def run_smrf(
    input_file: str,
    output_file: str,
    window: float = 18.0,
    slope: float = 0.15,
    threshold: float = 0.5,
    window_increment: float = 16.0,
) -> str:
    """Run PDAL Simple Morphological Filter (SMRF).

    Parameters
    ----------
    input_file: str
        Path to input LAS/LAZ file.
    output_file: str
        Destination file for filtered result.
    window: float, optional
        Initial window size for the filter.
    slope: float, optional
        Slope parameter for underlying terrain.
    threshold: float, optional
        Elevation threshold.
    window_increment: float, optional
        Increment applied to window size each step.

    Returns
    -------
    str
        Path to the written file.
    """
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
    return os.path.abspath(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMRF ground filter")
    parser.add_argument("input", help="Input LAS/LAZ file")
    parser.add_argument("output", help="Output LAS/LAZ file")
    parser.add_argument("--window", type=float, default=18.0)
    parser.add_argument("--slope", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-increment", type=float, default=16.0)
    args = parser.parse_args()

    run_smrf(
        args.input,
        args.output,
        window=args.window,
        slope=args.slope,
        threshold=args.threshold,
        window_increment=args.window_increment,
    )
