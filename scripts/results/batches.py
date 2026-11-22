"""Shared by the export_*.py scripts here: the final ablation batches, one per GPU,
and the command line every script takes.

The all-static row of each ablation batch is its end-to-end result, so these four
batches hold every result the report plots. The list matches BATCHES in the Makefile.

Every export counts a kernel run, and a speedup comparison, once: read_ablation.load_batch
marks the records that repeat another config's run (read_ablation.mark_duplicates)."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "ablation"))

FINAL_BATCHES = [
    "experiments/results/ablation/titanv/batch_20261118_090628",
    "experiments/results/ablation/rtx2080ti/batch_20261118_090535",
    "experiments/results/ablation/a100/batch_20261118_135245",
    "experiments/results/ablation/rtx4070/batch_20261118_090605",
]


def gpu_of(batch: Path) -> str:
    """experiments/results/ablation/<gpu>/batch_<timestamp> -> <gpu>."""
    return batch.resolve().parent.name


def parse_args(description: str, batches: bool) -> argparse.Namespace:
    """--out DIR (where the CSVs go, such as the report's data folder) and, for the
    scripts that read the batches, the batch folders (default: FINAL_BATCHES)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, required=True, help="folder the CSVs are written to")
    if batches:
        parser.add_argument("batch_dirs", type=Path, nargs="*",
                            help="ablation batch folders, one per GPU (default: the four final ones)")
    args = parser.parse_args()
    args.out = args.out.expanduser()
    if not args.out.is_dir():
        parser.error(f"--out {args.out} is not a folder")
    if batches:
        args.batch_dirs = [d.expanduser() for d in args.batch_dirs] or [REPO / b for b in FINAL_BATCHES]
        for d in args.batch_dirs:
            if not d.is_dir():
                parser.error(f"{d} is not a folder")
    return args
