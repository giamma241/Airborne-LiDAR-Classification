#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="myria3d",
    version="0.1.0",
    description="Airborne LiDAR Classification Pipeline",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "gdal>=3.4.0",
        "pdal>=3.0.0",
        "laspy>=2.0.0",
        "pyproj>=3.4.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.60.0",
        "click>=8.0.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "myria3d-train=myria3d.cli:train",
            "myria3d-predict=myria3d.cli:predict",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
