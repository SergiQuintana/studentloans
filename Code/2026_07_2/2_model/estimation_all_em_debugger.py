"""Checked iteration-zero runner for the structural NPL estimation.

The numerical production functions remain unchanged.  This program calls
parallel debug-only entry points that use the same kernels, persist the same
CCP/VJT/EVT artifacts, and stop with a Spyder-ready snapshot at the first
invalid state (unless ``--continue-after-errors`` is requested).

Examples
--------
Targeted Bellman check using existing initial CCPs::

    python estimation_all_em_debugger.py --stage bellman \
        --types 1 --x1-indices 48 --workers 1

Rebuild and check the complete iteration-zero critical path::

    python estimation_all_em_debugger.py --stage full --workers 60
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

import model_em_algorithm as me
import model_predict_ccps as mccp
import model_solution_em as ms
from config import ENSURE_DEFAULT_TREE, EST, LIK
from latent_types import TYPE_IDS, load_em_posteriors
from model_debug_checks import DebugConfig, array_summary


def parse_integer_selection(text, valid_values, label):
    valid_values = tuple(int(value) for value in valid_values)
    if str(text).strip().lower() in {"", "all"}:
        return valid_values
    result = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    invalid = sorted(set(result) - set(valid_values))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}")
    return tuple(dict.fromkeys(result))


def _initial_ccp_worker(arguments):
    return mccp.get_all_ccps_debug(*arguments)


def _bellman_worker(arguments):
    return ms.get_all_evt_debug(*arguments)


def _bellman_initializer():
    ms.reload_budgetshock_params()


def _run_parallel(worker, arguments, workers, label, initializer=None):
    started = time.perf_counter()
    results = []
    if workers == 1:
        if initializer is not None:
            initializer()
        for number, item in enumerate(arguments, start=1):
            results.append(worker(item))
            print(f"[{label}] {number}/{len(arguments)} tasks", flush=True)
        return results
    with mp.Pool(processes=workers, initializer=initializer) as pool:
        for number, result in enumerate(
            pool.imap_unordered(worker, arguments, chunksize=1), start=1
        ):
            results.append(result)
            print(
                f"[{label}] {number}/{len(arguments)} tasks | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return results


def _summarize_prepared_arrays(loaded):
    names = (
        "choices_all", "vjt_all_types", "x1_new", "choices_array_all",
        "x_change", "x_educ", "x_first2", "x_first4", "x_firstgrad", "x_exp",
    )
    summaries = []

    def visit(name, value):
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{name}.{key}", nested)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(f"{name}[{index}]", nested)
        else:
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number):
                summaries.append({"name": name, **array_summary(array)})

    for name, value in zip(names, loaded):
        visit(name, value)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("initial-ccp", "bellman", "critical", "likelihood", "full"),
        default="critical",
        help=(
            "critical rebuilds initial CCPs and solves the Bellman recursion; "
            "full additionally prepares arrays and evaluates the likelihood once."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--types", default="all")
    parser.add_argument("--x1-indices", default="all")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diagnostic root (default: Output/likelihood/estimation_debugger/RUN_ID).",
    )
    parser.add_argument("--max-failures-per-task", type=int, default=20)
    parser.add_argument(
        "--continue-after-errors",
        action="store_true",
        help="Record additional errors instead of stopping at the first one.",
    )
    parser.add_argument(
        "--no-save-verification",
        action="store_true",
        help="Skip reopening initial-CCP bundles after they are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if args.max_failures_per_task < 1:
        raise ValueError("--max-failures-per-task must be at least one")

    ENSURE_DEFAULT_TREE(T=ms.T)
    selected_types = parse_integer_selection(args.types, TYPE_IDS, "type IDs")
    selected_x1 = parse_integer_selection(
        args.x1_indices, range(len(ms.invariant_states)), "x1 indices"
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        LIK("estimation_debugger", args.run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_config = DebugConfig(
        output_dir=str(output_dir),
        fail_fast=not args.continue_after_errors,
        max_failures=args.max_failures_per_task,
        trace_draws=True,
        verify_saved=not args.no_save_verification,
    )
    metadata = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage,
        "workers": args.workers,
        "types": selected_types,
        "x1_indices": selected_x1,
        "fail_fast": debug_config.fail_fast,
        "model_root": str(Path(ms.pathout).parent),
        "output_dir": str(output_dir),
        "production_outputs_modified": args.stage in {
            "initial-ccp", "bellman", "critical", "full"
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    stages = {
        "initial-ccp": ("initial-ccp",),
        "bellman": ("bellman",),
        "critical": ("initial-ccp", "bellman"),
        "likelihood": ("likelihood",),
        "full": ("initial-ccp", "bellman", "likelihood"),
    }[args.stage]
    all_results = []
    try:
        # Exactly the state construction performed at the start of production.
        if "initial-ccp" in stages or "bellman" in stages:
            print("Simulating/checking model state grids", flush=True)
            ms.simulate_all_states(11)

        x0 = np.asarray(np.load(EST("param_g.npy"), allow_pickle=False), dtype=float)
        debt = np.asarray(ms.debt_range, dtype=float)

        if "initial-ccp" in stages:
            print("Building checked auxiliary initial CCPs", flush=True)
            parameters = {
                type_id: mccp.load_utility_parameters(type_id)
                for type_id in selected_types
            }
            initial_arguments = [
                (
                    x1_index, ms.invariant_states, debt,
                    parameters[type_id], type_id, debug_config,
                )
                for type_id in selected_types
                for x1_index in selected_x1
            ]
            all_results.extend(
                _run_parallel(
                    _initial_ccp_worker, initial_arguments, args.workers,
                    "Initial CCP DEBUG",
                )
            )

        if "bellman" in stages:
            print("Solving checked iteration-zero Bellman recursion", flush=True)
            utility_parameters = {
                type_id: ms.build_param_g(type_id, x0)
                for type_id in selected_types
            }
            bellman_arguments = [
                (
                    x1_index, ms.invariant_states, debt, debt, 0,
                    utility_parameters[type_id], 0, 0, 0, type_id, True,
                    debug_config,
                )
                for type_id in selected_types
                for x1_index in selected_x1
            ]
            all_results.extend(
                _run_parallel(
                    _bellman_worker, bellman_arguments, args.workers,
                    "Bellman DEBUG", initializer=_bellman_initializer,
                )
            )

        if "likelihood" in stages:
            if selected_types != tuple(TYPE_IDS) or selected_x1 != tuple(
                range(len(ms.invariant_states))
            ):
                raise ValueError(
                    "The likelihood stage requires --types all and --x1-indices all "
                    "because it consumes the complete artifact grid."
                )
            print("Preparing checked likelihood inputs", flush=True)
            with mp.Pool(processes=min(args.workers, ms.T - 1)) as pool:
                pool.map(me.prepare_vjt_feasible, range(1, ms.T))
            loaded = me.load_all_arrays_feasible()
            summaries = _summarize_prepared_arrays(loaded)
            (output_dir / "prepared_array_summary.json").write_text(
                json.dumps(summaries, indent=2), encoding="utf-8"
            )
            nonfinite = [
                row for row in summaries
                if row.get("nan", 0) or row.get("positive_inf", 0)
            ]
            if nonfinite:
                (output_dir / "prepared_array_failures.json").write_text(
                    json.dumps(nonfinite, indent=2), encoding="utf-8"
                )
                raise RuntimeError(
                    f"Prepared likelihood inputs contain NaN/+Inf in "
                    f"{len(nonfinite)} arrays; see prepared_array_failures.json"
                )
            q = load_em_posteriors(EST("auxiliary_em_results.npz"))
            # Map any chosen -inf/nonfinite probability back to the observed
            # row, pubid, state, debt, choice, period, and permanent type.
            import diagnose_likelihood_inputs as likelihood_diagnostics

            cell_summary, row_details, array_details = (
                likelihood_diagnostics.diagnose_likelihood_inputs(
                    q, x0, worst_rows=50, prepared_arrays=loaded
                )
            )
            cell_summary.to_csv(output_dir / "likelihood_cell_summary.csv", index=False)
            row_details.to_csv(output_dir / "likelihood_problem_rows.csv", index=False)
            array_details.to_csv(output_dir / "likelihood_array_summary.csv", index=False)
            probability_failures = cell_summary[
                (cell_summary["positive_q_with_nonfinite_stable_log_probability"] > 0)
                | (cell_summary["positive_q_with_zero_legacy_probability"] > 0)
                | (cell_summary["chosen_invalid"] > 0)
            ]
            if not probability_failures.empty:
                probability_failures.to_csv(
                    output_dir / "likelihood_probability_failures.csv", index=False
                )
                raise RuntimeError(
                    "Chosen alternatives create nonfinite log probabilities; see "
                    "likelihood_problem_rows.csv"
                )
            print("Evaluating the structural likelihood once", flush=True)
            value, gradient = me.likelihood(x0, *loaded, q)
            likelihood_result = {
                "value": float(value),
                "gradient": array_summary(gradient),
                "value_finite": bool(np.isfinite(value)),
            }
            (output_dir / "likelihood_result.json").write_text(
                json.dumps(likelihood_result, indent=2), encoding="utf-8"
            )
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise RuntimeError(
                    "Likelihood value or gradient is nonfinite; see likelihood_result.json"
                )
    except Exception:
        (output_dir / "fatal_exception.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        print(f"Debugger stopped. See {output_dir}", flush=True)
        raise

    (output_dir / "task_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"Debugger completed successfully. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
