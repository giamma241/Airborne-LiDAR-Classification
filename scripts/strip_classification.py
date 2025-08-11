#!/usr/bin/env python3
"""
Batch LAS/LAZ Cleaner
=====================

Process all LAS/LAZ files in a folder using the strip_classification script.

Usage:
    python batch_strip_classification.py --input-dir data/test --output-dir data/test_clean
    python batch_strip_classification.py --input-dir data/test --output-dir data/test_clean --parallel 4
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("batch-cleaner")


def process_single_file(
    input_file: Path,
    output_dir: Path,
    auto_drop: bool = True,
    class_value: int = 1,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Process a single LAS/LAZ file."""
    output_file = output_dir / input_file.name

    cmd = [
        sys.executable,  # Use the same Python interpreter
        "strip_classification.py",
        "--in",
        str(input_file),
        "--out",
        str(output_file),
        "--class-value",
        str(class_value),
    ]

    if auto_drop:
        cmd.append("--auto-drop")

    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout per file
        )

        if result.returncode == 0:
            return True, f"✅ {input_file.name}"
        else:
            return False, f"❌ {input_file.name}: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"⏱️ {input_file.name}: Timeout (>5 minutes)"
    except Exception as e:
        return False, f"❌ {input_file.name}: {str(e)}"


def process_folder(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.la[sz]",
    auto_drop: bool = True,
    class_value: int = 1,
    parallel: int = 1,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    """Process all LAS/LAZ files in a folder."""

    # Find all files
    las_files = list(input_dir.glob("*.las"))
    laz_files = list(input_dir.glob("*.laz"))
    all_files = las_files + laz_files

    if not all_files:
        log.warning(f"No LAS/LAZ files found in {input_dir}")
        return

    log.info(f"Found {len(all_files)} files to process")
    log.info(f"Input directory: {input_dir}")
    log.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out existing files if not overwriting
    if not overwrite:
        files_to_process = []
        for f in all_files:
            output_file = output_dir / f.name
            if output_file.exists():
                log.info(f"⏭️ Skipping {f.name} (output exists)")
            else:
                files_to_process.append(f)
    else:
        files_to_process = all_files

    if not files_to_process:
        log.info("No files to process (all outputs exist)")
        return

    log.info(f"Processing {len(files_to_process)} files...")

    # Process files
    success_count = 0
    failed_count = 0

    if parallel > 1:
        # Parallel processing
        log.info(f"Using {parallel} parallel workers")

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    process_single_file, f, output_dir, auto_drop, class_value, verbose
                ): f
                for f in files_to_process
            }

            for future in as_completed(futures):
                success, message = future.result()
                log.info(message)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
    else:
        # Sequential processing
        for i, f in enumerate(files_to_process, 1):
            log.info(f"[{i}/{len(files_to_process)}] Processing {f.name}...")
            success, message = process_single_file(
                f, output_dir, auto_drop, class_value, verbose
            )
            log.info(message)
            if success:
                success_count += 1
            else:
                failed_count += 1

    # Summary
    log.info("=" * 50)
    log.info("Processing complete!")
    log.info(f"✅ Successful: {success_count}")
    if failed_count > 0:
        log.warning(f"❌ Failed: {failed_count}")

    # Calculate total size reduction
    try:
        input_size = sum(f.stat().st_size for f in files_to_process) / (1024**3)
        output_files = [output_dir / f.name for f in files_to_process]
        output_size = sum(f.stat().st_size for f in output_files if f.exists()) / (
            1024**3
        )
        reduction_pct = (1 - output_size / input_size) * 100 if input_size > 0 else 0

        log.info(f"Total input size: {input_size:.2f} GB")
        log.info(f"Total output size: {output_size:.2f} GB")
        if reduction_pct > 0:
            log.info(f"Size reduction: {reduction_pct:.1f}%")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Batch process LAS/LAZ files to remove extra dimensions and reset classification."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing LAS/LAZ files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for cleaned files",
    )
    parser.add_argument(
        "--auto-drop",
        action="store_true",
        default=True,
        help="Auto-detect and remove typical prediction dimensions (default: True)",
    )
    parser.add_argument(
        "--no-auto-drop",
        dest="auto_drop",
        action="store_false",
        help="Don't auto-detect dimensions to remove",
    )
    parser.add_argument(
        "--class-value",
        type=int,
        default=1,
        help="Value to set for Classification field (default: 1)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        log.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        log.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        auto_drop=args.auto_drop,
        class_value=args.class_value,
        parallel=args.parallel,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()