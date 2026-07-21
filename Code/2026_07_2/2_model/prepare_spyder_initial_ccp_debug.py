"""Build a self-contained local Spyder bundle with newly predicted initial CCPs.

By default the CCP prediction matches the older choices-only auxiliary guess:
only the auxiliary choice utility ``g`` enters the logit probability.  The
script generates CCPs in a temporary directory, validates them, and packages
them with the inputs needed to solve one structural ``(type, x1)`` task.
Production CCP, VJT, EVT, and estimate files are never modified.

Server example::

    python prepare_spyder_initial_ccp_debug.py \
        --type 1 --x1-index 48 \
        --output /home/ubuntu/work/spyder_initial_ccp_debug.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import model_predict_ccps as ccp_model
import model_solution_em as solution
from config import DIR, EST, OUT


PERIODS = tuple(range(solution.T - 1, 0, -1))
FUNCTION_FILES = (
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choice_parameters(auxiliary_file):
    with np.load(auxiliary_file, allow_pickle=False) as archive:
        if "choice_parameters" not in archive.files:
            raise KeyError(f"{auxiliary_file} does not contain choice_parameters.")
        parameters = np.asarray(archive["choice_parameters"], dtype=float)
    expected = ccp_model.total_n + 2
    if parameters.shape != (expected,):
        raise ValueError(
            f"choice_parameters has shape {parameters.shape}; expected {(expected,)}."
        )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("choice_parameters contains nonfinite values.")
    return parameters


def predict_one_state(type_id, x1_index, specification, output_root):
    """Predict and save nine initial-CCP bundles outside production Output."""
    type_id = int(type_id)
    x1_index = int(x1_index)
    if type_id not in tuple(int(value) for value in ccp_model.TYPE_IDS):
        raise ValueError(f"Invalid type ID {type_id}.")
    if not 0 <= x1_index < len(solution.invariant_states):
        raise ValueError(
            f"x1 index must be between 0 and {len(solution.invariant_states) - 1}."
        )

    auxiliary_file = Path(EST("auxiliary_em_results.npz"))
    raw_choice_parameters = choice_parameters(auxiliary_file)
    utility_parameters = solution.build_param_g(
        type_id, raw_choice_parameters[: ccp_model.total_n]
    )
    full_parameters = None
    if specification == "full-auxiliary":
        full_parameters = ccp_model.load_utility_parameters(
            type_id, results_file=str(auxiliary_file)
        )

    debt = np.asarray(solution.debt_range, dtype=float)
    x1_row = np.asarray(solution.invariant_states[x1_index], dtype=int)
    inv = x1_row[None, :]
    x1_new = solution.get_x1_new(x1_row)
    output_root = Path(output_root)
    summaries = []
    generated = []

    for period in PERIODS:
        names = []
        arrays = []
        period_zero = 0
        period_nonfinite = 0
        period_min = np.inf
        period_max = -np.inf
        for x2 in solution.get_x2(period):
            x2 = np.asarray(x2, dtype=int)
            choices = solution.get_possible_choices(x2)
            home = np.flatnonzero(np.all(choices == 0, axis=1))
            if home.size != 1:
                raise ValueError(
                    f"Expected one home choice; found {home.size} for state {x2}."
                )
            home_index = int(home[0])

            if specification == "choice-only":
                g = np.asarray(
                    solution.get_all_g(
                        utility_parameters,
                        inv,
                        x1_new,
                        x2,
                        choices,
                        period,
                    ),
                    dtype=float,
                )
                auxiliary_vjt = np.broadcast_to(
                    g[None, :], (debt.size, g.size)
                )
            else:
                auxiliary_vjt = np.asarray(
                    ccp_model.get_vjt_static(
                        full_parameters,
                        inv,
                        x1_new,
                        x2,
                        choices,
                        period,
                        debt,
                        type_id,
                    ),
                    dtype=float,
                )

            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                home_log_ccp = (
                    auxiliary_vjt[:, home_index]
                    - logsumexp(auxiliary_vjt, axis=1)
                )
                home_ccp = np.exp(home_log_ccp)

            period_zero += int(np.count_nonzero(home_ccp == 0.0))
            period_nonfinite += int(np.count_nonzero(~np.isfinite(home_ccp)))
            finite = home_ccp[np.isfinite(home_ccp)]
            if finite.size:
                period_min = min(period_min, float(np.min(finite)))
                period_max = max(period_max, float(np.max(finite)))
            names.append(f"ccp_t{period}_{inv}_{x2}")
            arrays.append(np.asarray(home_ccp, dtype=float))

        if period_zero or period_nonfinite:
            raise FloatingPointError(
                f"Generated CCPs failed validation in period {period}: "
                f"zero={period_zero}, nonfinite={period_nonfinite}."
            )
        filename = f"ccp_t{period}_[{x1_row}]_em{type_id}.npz"
        relative_path = Path("Output/ccp") / str(period) / filename
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **dict(zip(names, arrays)))
        generated.append((destination, relative_path, "newly predicted initial CCP"))
        summaries.append(
            {
                "period": int(period),
                "states": len(names),
                "debt_cells": int(sum(array.size for array in arrays)),
                "zero": period_zero,
                "nonfinite": period_nonfinite,
                "minimum": period_min,
                "maximum": period_max,
                "file": relative_path.as_posix(),
            }
        )
        print(
            f"Predicted period {period}: states={len(names)}, "
            f"min={period_min:.6g}, max={period_max:.6g}",
            flush=True,
        )
    return generated, summaries, x1_row


def structural_input_files():
    model_root = Path(DIR["MODEL"])
    files = [
        (
            model_root / "Estimates" / "auxiliary_em_results.npz",
            Path("Estimates/auxiliary_em_results.npz"),
            "auxiliary EM estimates and financial-process parameters",
        ),
        (
            model_root / "Estimates" / "param_g.npy",
            Path("Estimates/param_g.npy"),
            "structural flow-utility parameters",
        ),
        (
            model_root / "Estimates" / "budgetshock_params.npy",
            Path("Estimates/budgetshock_params.npy"),
            "structural budget-shock and risk-aversion parameters",
        ),
        (
            model_root / "Output" / "cache" / "interp_dict.joblib",
            Path("Output/cache/interp_dict.joblib"),
            "terminal continuation interpolation cache",
        ),
    ]
    for filename in FUNCTION_FILES:
        files.append(
            (
                model_root / "Inputs" / "function_coefficients" / filename,
                Path("Inputs/function_coefficients") / filename,
                "fixed structural wage/graduation input",
            )
        )
    for period in range(1, solution.T + 1):
        filename = f"states_t{period}.npy"
        files.append(
            (
                model_root / "Output" / "states" / filename,
                Path("Output/states") / filename,
                "dynamic state grid",
            )
        )
    return files


def build_bundle(output_path, type_id, x1_index, specification):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = structural_input_files()
    missing = [str(source) for source, _, _ in inputs if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot prepare the Spyder bundle; required files are missing:\n"
            + "\n".join(missing)
        )

    with tempfile.TemporaryDirectory(prefix="initial_ccp_spyder_") as temp:
        generated, ccp_summary, x1 = predict_one_state(
            type_id, x1_index, specification, temp
        )
        files = inputs + generated
        manifest_files = []
        for source, archive_path, purpose in files:
            source = Path(source)
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
            "ccp_source": "newly predicted in memory; production CCP files not read",
            "ccp_specification": specification,
            "type_id": int(type_id),
            "x1_index": int(x1_index),
            "x1": x1.tolist(),
            "ccp_summary": ccp_summary,
            "extract_into": "a separate directory used as MODEL_ROOT",
            "files": manifest_files,
        }
        with zipfile.ZipFile(
            output_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for source, archive_path, _ in files:
                archive.write(source, archive_path.as_posix())
            archive.writestr(
                "spyder_initial_ccp_manifest.json", json.dumps(manifest, indent=2)
            )
    return output_path, manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", type=int, default=1, help="Joint type ID (default: 1).")
    parser.add_argument(
        "--x1-index",
        type=int,
        default=48,
        help="Zero-based invariant-state index (default: 48, x1=[1,1,1,1]).",
    )
    parser.add_argument(
        "--ccp-spec",
        choices=("choice-only", "full-auxiliary"),
        default="choice-only",
        help=(
            "Initial-CCP index. 'choice-only' reproduces the older g-only guess; "
            "'full-auxiliary' uses consumption, wages, grants, transfers, and debt."
        ),
    )
    parser.add_argument(
        "--output",
        default=OUT("likelihood", "pipeline_diagnostics", "spyder_initial_ccp_debug.zip"),
        help="Output ZIP path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, manifest = build_bundle(
        args.output, args.type, args.x1_index, args.ccp_spec
    )
    print(f"\nCreated {output_path}")
    print(f"CCP specification: {manifest['ccp_specification']}")
    print(f"Type: {manifest['type_id']}")
    print(f"x1 index/value: {manifest['x1_index']} / {manifest['x1']}")
    print(f"Size: {output_path.stat().st_size / (1024 ** 2):.2f} MiB")
    print("Production model artifacts were not modified.")


if __name__ == "__main__":
    main()
