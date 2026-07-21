"""Package the server artifacts needed to reproduce initial CCPs locally.

The ZIP uses paths relative to a model root (``Estimates/...``, ``Inputs/...``,
and ``Output/...``).  Extract it into a separate local directory and set
``MODEL_ROOT`` to that directory before running the diagnostic.  Source model
artifacts are only read; none are changed.

Example
-------
::

    python package_initial_ccp_debug.py --type 1 --x1-index 48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from config import DIR, OUT


INITIAL_CCP_PERIODS = tuple(range(9, 0, -1))
WAGE_AND_GRADUATION_FILES = (
    "wage_0.npy",
    "wage_1.npy",
    "wage_2.npy",
    "wage_3.npy",
    "wage_6.npy",
    "wage_7.npy",
    "wage_8.npy",
    "wage_9.npy",
    "wage_10.npy",
    "sigmas.npy",
    "prob_grad_twoyear.npy",
    "prob_grad_four.npy",
    "prob_grad_grad.npy",
)


def invariant_states():
    return np.array(
        np.meshgrid([1, 2, 3, 4], [1, 2, 3, 4], [0, 1], [0, 1])
    ).T.reshape(-1, 4)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_files(type_id=1, x1_index=48, include_stored_ccps=True):
    model_root = Path(DIR["MODEL"])
    files = [
        (
            model_root / "Estimates" / "auxiliary_em_results.npz",
            Path("Estimates/auxiliary_em_results.npz"),
            "auxiliary EM estimates, posterior weights, and financial process",
        ),
        (
            model_root / "Estimates" / "param_g.npy",
            Path("Estimates/param_g.npy"),
            "structural utility vector used by the diagnostic parameter audit",
        ),
    ]
    for filename in WAGE_AND_GRADUATION_FILES:
        files.append(
            (
                model_root / "Inputs" / "function_coefficients" / filename,
                Path("Inputs/function_coefficients") / filename,
                "fixed wage/graduation input loaded by model modules",
            )
        )
    for period in INITIAL_CCP_PERIODS:
        filename = f"states_t{period}.npy"
        files.append(
            (
                model_root / "Output" / "states" / filename,
                Path("Output/states") / filename,
                "dynamic state grid used for fresh CCP prediction",
            )
        )

    states = invariant_states()
    if not 0 <= int(x1_index) < len(states):
        raise ValueError(f"--x1-index must be between 0 and {len(states) - 1}.")
    if not 1 <= int(type_id) <= 16:
        raise ValueError("--type must be between 1 and 16.")
    x1 = states[int(x1_index)]
    if include_stored_ccps:
        for period in INITIAL_CCP_PERIODS:
            filename = f"ccp_t{period}_[{x1}]_em{int(type_id)}.npz"
            files.append(
                (
                    model_root / "Output" / "ccp" / str(period) / filename,
                    Path("Output/ccp") / str(period) / filename,
                    "stored CCP bundle for fresh-versus-stored comparison",
                )
            )
    return files, x1


def build_bundle(output_path, type_id=1, x1_index=48, include_stored_ccps=True):
    files, x1 = required_files(type_id, x1_index, include_stored_ccps)
    missing = [str(source) for source, _, _ in files if not source.is_file()]
    if missing:
        preview = "\n".join(missing)
        raise FileNotFoundError(
            f"Cannot build the local-debug bundle; {len(missing)} files are missing:\n"
            f"{preview}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for source, archive_path, purpose in files:
        stat = source.stat()
        manifest_files.append(
            {
                "archive_path": archive_path.as_posix(),
                "source_path": str(source),
                "purpose": purpose,
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "sha256": sha256(source),
            }
        )
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_model_root": str(DIR["MODEL"]),
        "type_id": int(type_id),
        "x1_index": int(x1_index),
        "x1": x1.tolist(),
        "stored_ccps_included": bool(include_stored_ccps),
        "extract_into": "a separate directory that will be used as MODEL_ROOT",
        "files": manifest_files,
    }

    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source, archive_path, _ in files:
            archive.write(source, archive_path.as_posix())
        archive.writestr("initial_ccp_bundle_manifest.json", json.dumps(manifest, indent=2))
    return output_path, manifest


def parse_args():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", type=int, default=1, help="Joint type ID (default: 1).")
    parser.add_argument(
        "--x1-index",
        type=int,
        default=48,
        help="Zero-based invariant-state index (default: 48, x1=[1,1,1,1]).",
    )
    parser.add_argument(
        "--without-stored-ccps",
        action="store_true",
        help="Package only inputs for fresh prediction, excluding stored CCP comparison files.",
    )
    parser.add_argument(
        "--output",
        default=OUT(
            "likelihood",
            "pipeline_diagnostics",
            f"initial_ccp_local_bundle_{timestamp}.zip",
        ),
        help="Output ZIP path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, manifest = build_bundle(
        args.output,
        type_id=args.type,
        x1_index=args.x1_index,
        include_stored_ccps=not args.without_stored_ccps,
    )
    print(f"Created {output_path}")
    print(f"Files: {len(manifest['files'])}")
    print(f"Type: {manifest['type_id']}")
    print(f"x1 index/value: {manifest['x1_index']} / {manifest['x1']}")
    print(f"Size: {output_path.stat().st_size / (1024 ** 2):.2f} MiB")
    print("No source model artifacts were modified.")


if __name__ == "__main__":
    main()
