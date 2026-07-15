# -*- coding: utf-8 -*-
"""Move CCP bundles written below a duplicated absolute output path.

The old call passed an absolute path to ``save_npz_here``, which produced files
below::

    Model/Output/home/ubuntu/work/Model/Output/ccp

This one-off recovery script moves those bundles to::

    Model/Output/ccp

Run on the server with ``MODEL_ROOT=/home/ubuntu/work/Model``. Existing
destination files are replaced because they may belong to the old type layout.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from config import DIR


def progress(message: str) -> None:
    print(message, flush=True)


def progress_interval(total: int) -> int:
    """Report roughly every 2 percent, bounded to readable intervals."""
    return max(1, min(250, total // 50))


def duplicated_output_root(output_root: Path) -> Path:
    """Reproduce the accidental ``Output/<absolute path>`` directory."""
    absolute_parts = output_root.resolve().parts
    if not absolute_parts or absolute_parts[0] != os.sep:
        raise ValueError(
            "This recovery script is intended for the Linux server's absolute paths."
        )
    return output_root.joinpath(*absolute_parts[1:])


def validate_npz(path: Path) -> None:
    """Check that a source bundle is readable and contains at least one array."""
    with np.load(path, allow_pickle=False) as bundle:
        if not bundle.files:
            raise ValueError(f"CCP bundle contains no arrays: {path}")


def remove_empty_directories(root: Path) -> None:
    """Remove only empty directories inside the known duplicated tree."""
    if not root.exists():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def move_misplaced_ccps(output_root: Path, dry_run: bool = False) -> tuple[int, int]:
    started = time.perf_counter()
    output_root = output_root.resolve()
    source_root = duplicated_output_root(output_root).resolve()
    if source_root == output_root or output_root not in source_root.parents:
        raise ValueError(
            f"Refusing unsafe source path {source_root}; expected it below {output_root}."
        )

    source_ccp = source_root / "ccp"
    progress(f"Canonical output directory: {output_root}")
    progress(f"Searching misplaced CCP directory: {source_ccp}")
    if not source_ccp.is_dir():
        raise FileNotFoundError(
            "The misplaced CCP directory was not found. Expected:\n"
            f"  {source_ccp}"
        )

    source_files = sorted(source_ccp.rglob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"No CCP .npz bundles were found below {source_ccp}.")
    total = len(source_files)
    interval = progress_interval(total)
    progress(f"Found {total} CCP bundles. Validating them before moving anything...")

    # Validate every source before replacing any destination file.
    validation_started = time.perf_counter()
    for index, source in enumerate(source_files, start=1):
        validate_npz(source)
        if index == 1 or index % interval == 0 or index == total:
            elapsed = time.perf_counter() - validation_started
            progress(
                f"Validation: {index}/{total} ({100.0 * index / total:.1f}%) "
                f"| {elapsed:.1f}s"
            )

    overwritten = 0
    action = "Checking move" if dry_run else "Moving"
    move_started = time.perf_counter()
    progress(
        "Validation completed. "
        + ("Previewing destination paths..." if dry_run else "Moving bundles...")
    )
    for index, source in enumerate(source_files, start=1):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        if destination.exists():
            overwritten += 1
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if index == 1:
            progress(f"First bundle: {source} -> {destination}")
        if index == 1 or index % interval == 0 or index == total:
            elapsed = time.perf_counter() - move_started
            progress(
                f"{action}: {index}/{total} ({100.0 * index / total:.1f}%) "
                f"| {elapsed:.1f}s | replacements={overwritten}"
            )

    if not dry_run:
        progress("Removing empty directories left inside the duplicated output tree...")
        remove_empty_directories(source_root)

    progress(f"Recovery step finished in {time.perf_counter() - started:.1f}s")

    return len(source_files), overwritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=DIR["MODEL_OUTPUT"],
        help="Canonical Model/Output directory; inferred from MODEL_ROOT by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show and validate the planned moves without changing any files.",
    )
    arguments = parser.parse_args()

    moved, overwritten = move_misplaced_ccps(
        Path(arguments.output_root), dry_run=arguments.dry_run
    )
    action = "Would move" if arguments.dry_run else "Moved"
    progress(
        f"{action} {moved} CCP bundles; "
        f"{overwritten} destination files {'would be ' if arguments.dry_run else ''}replaced."
    )


if __name__ == "__main__":
    main()
