#!/usr/bin/env python3
"""
PDAL-based LAS Cleaner
======================

Obiettivo:
- Azzerare il campo standard LAS 'Classification' (non rimovibile dallo standard).
- Rimuovere eventuali extra dimensions come 'PredictedClassification', 'entropy', 'building', 'ground', ecc.

Uso:
  python strip_classification.py --in input.las --out output.las
  # opzionale: specificare quali extra-dims rimuovere
  python strip_classification.py --in input.las --out output.las --drop PredictedClassification entropy building ground
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Set

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdal-cleaner")


def check_pdal() -> None:
    try:
        r = subprocess.run(
            ["pdal", "--version"], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        log.info(f"PDAL: {r.stdout.strip()}")
    except FileNotFoundError:
        log.error("PDAL non trovato. Installa con: conda install -c conda-forge pdal")
        sys.exit(1)


def pdal_list_dims(las_path: Path) -> Set[str]:
    """Ritorna l’insieme dei nomi dimensione presenti nel file (via `pdal info`)."""
    cmd = ["pdal", "info", "--metadata", str(las_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning(
            "Impossibile leggere metadata con pdal info; proseguo senza auto-rilevazione extra-dims."
        )
        return set()
    try:
        meta = json.loads(r.stdout)
        dims = meta.get("metadata", {}).get("readers.las", {}).get("dimensions", [])
        return set(dims)
    except Exception:
        return set()


def create_cleanup_pipeline(
    input_las: Path, output_las: Path, dims_to_drop: List[str]
) -> dict:
    """
    Crea pipeline PDAL:
      readers.las -> (filters.drop?) -> filters.assign(Classification=1) -> writers.las
    """
    pipeline = [{"type": "readers.las", "filename": str(input_las.absolute())}]

    # Se ci sono extra-dims da rimuovere, usa filters.drop
    if dims_to_drop:
        pipeline.append({"type": "filters.drop", "dimensions": ",".join(dims_to_drop)})

    # Azzerare la Classification standard
    # Nota: non si può eliminare, solo impostare il valore.
    pipeline.append(
        {
            "type": "filters.assign",
            "assignment": "Classification[:] = 1",  # oppure 1 per 'Unclassified'
        }
    )

    # Scrittura
    pipeline.append(
        {
            "type": "writers.las",
            "filename": str(output_las.absolute()),
            "minor_version": "4",  # LAS 1.4
            "dataformat_id": "3",  # Point format 3 (XYZ, Intensity, Time, RGB)
            "forward": "all",  # inoltra tutte le altre dimensioni
        }
    )
    return {"pipeline": pipeline}


def run_pipeline(pipeline: dict) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f, indent=2)
        tmp = Path(f.name)
    try:
        log.debug("Pipeline:\n" + json.dumps(pipeline, indent=2))
        r = subprocess.run(
            ["pdal", "pipeline", str(tmp)], capture_output=True, text=True
        )
        if r.returncode != 0:
            log.error("PDAL pipeline fallita:\n" + r.stderr)
            return False
        return True
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Rimuove extra-dims e azzera la Classification in un LAS/LAZ."
    )
    parser.add_argument(
        "--in", dest="in_path", required=True, type=Path, help="Input LAS/LAZ"
    )
    parser.add_argument(
        "--out", dest="out_path", required=True, type=Path, help="Output LAS/LAZ"
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=[],
        help="Elenco extra-dims da rimuovere (es. PredictedClassification entropy building ground)",
    )
    parser.add_argument(
        "--auto-drop",
        action="store_true",
        help="Auto-rileva e rimuove le extra-dims tipiche di predizione (PredictedClassification, entropy, building, ground).",
    )
    args = parser.parse_args()

    check_pdal()

    if not args.in_path.exists():
        log.error(f"Input non trovato: {args.in_path}")
        sys.exit(1)

    dims_to_drop: Set[str] = set(args.drop)

    if args.auto_drop:
        present = pdal_list_dims(args.in_path)
        # set “tipico” di extra-dims generate dall’inferenza
        typical = {"PredictedClassification", "entropy", "building", "ground"}
        dims_to_drop.update(typical.intersection(present))

    pipeline = create_cleanup_pipeline(
        args.in_path, args.out_path, sorted(dims_to_drop)
    )
    ok = run_pipeline(pipeline)

    if ok and args.out_path.exists():
        size_mb = args.out_path.stat().st_size / (1024**2)
        log.info(f"✅ Pulizia completata: {args.out_path} ({size_mb:.1f} MB)")
        if dims_to_drop:
            log.info("Rimosse extra-dims: " + ", ".join(sorted(dims_to_drop)))
        log.info("Campo 'Classification' azzerato (0).")
        sys.exit(0)
    else:
        log.error("❌ Operazione fallita.")
        sys.exit(2)


if __name__ == "__main__":
    main()
