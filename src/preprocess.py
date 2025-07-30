import json
import os

import pdal


def run_pdal_pipeline(config_path="../config/pdal_pipeline_preprocess.json"):
    """
    Loads and executes a PDAL pipeline from a JSON configuration file.
    Prints the output file name and its location after execution.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path) as f:
        pipeline_def = json.load(f)

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()

    # Try to find the output file from the pipeline definition
    output_file = None
    for stage in pipeline_def:
        if isinstance(stage, dict) and "filename" in stage:
            output_file = stage["filename"]

    if output_file:
        abs_path = os.path.abspath(output_file)
        print(f"Output file created: {output_file}")
        print(f"Stored at: {abs_path}")
    else:
        print("No output file found in the pipeline definition.")

    return pipeline


# Example usage:
if __name__ == "__main__":
    run_pdal_pipeline()
