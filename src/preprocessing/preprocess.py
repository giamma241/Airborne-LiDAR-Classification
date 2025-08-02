import json
import logging
import os

import pdal

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)

def run_pdal_pipeline(
    config_path="../config/pdal_pipeline_preprocess.json", parameters=None
):
    """
    Loads and executes a PDAL pipeline from a JSON configuration file.
    Logs the output file name and its location after execution.

    Params:
    - config_path: path to the PDAL pipeline JSON file
    - parameters: optional dictionary of parameters to inject into the pipeline
    """
    if not os.path.exists(config_path):
        logging.error("Pipeline config not found: %s", config_path)
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path) as f:
        pipeline_def = json.load(f)

    if parameters:
        pipeline_def = json.loads(json.dumps(pipeline_def).format(**parameters))

    logging.info("Executing PDAL pipeline from: %s", config_path)
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    pipeline.execute()
    logging.info("PDAL pipeline executed successfully")

    # Try to find the output file from the pipeline definition
    output_file = None
    stages = pipeline_def.get("pipeline", pipeline_def)
    for stage in stages:
        if isinstance(stage, dict) and "filename" in stage:
            output_file = stage["filename"]

    if output_file:
        abs_path = os.path.abspath(output_file)
        logging.info("Output file created: %s", output_file)
        logging.info("Stored at: %s", abs_path)
    else:
        logging.warning("No output file found in the pipeline definition.")

    return pipeline

if __name__ == "__main__":
    run_pdal_pipeline()
