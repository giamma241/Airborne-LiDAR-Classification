# Airborne LiDAR Ground and Vegetation Classification

This repository contains pipelines for processing airborne and UAV-based LiDAR data, focusing on ground and vegetation classification using both traditional and deep learning methods.

## Structure
- `data/`: Raw and processed point cloud files (.las/.laz)
- `notebooks/`: Interactive Jupyter notebooks for each stage
- `src/`: Source code modules organized by function
- `scripts/`: CLI tools to run end-to-end workflows
- `configs/`: YAML/JSON configs for filter/model parameters

## Installation
```bash
conda env create -f environment.yml
conda activate lidar-env
```

## Quickstart
```bash
python scripts/run_classical.py --input data/raw/sample.las --output data/results/sample_ground.las --method csf
```

## Contact
Your Name - your.email@example.com
"""
