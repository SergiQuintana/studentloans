"""Checked runner for the complete structural NPL estimation.

The numerical production functions remain unchanged.  This program calls
parallel debug-only entry points that use the same kernels, persist the same
CCP/VJT/EVT artifacts, and stop with a Spyder-ready snapshot at the first
invalid state (unless ``--continue-after-errors`` is requested).

Examples
--------
Targeted Bellman check using existing initial CCPs::

    python estimation_all_em_debugger.py --stage bellman \
        --types 1 --x1-indices 48 --workers 1

Run and check the complete NPL estimation::

    python estimation_all_em_debugger.py --stage full --workers 60 --iterations 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import budget_shock as bs
import model_em_algorithm as me
import model_getccp_sequence as mgs
import model_predict_ccps as mccp
import model_solution_em as ms
from model_fitloans_dynamic import estimate_budget_shock_all_education
from config import ENSURE_DEFAULT_TREE, EST, LIK
from latent_types import TYPE_IDS, load_em_posteriors
from model_debug_checks import DebugConfig, array_summary


BUDGET_SMM_DRAWS = 100
BUDGET_SMM_ANNEALING_MAXFUN = 500
BUDGET_SMM_MAXITER = 1000
BUDGET_SMM_CELL_WORKERS = None
BUDGET_SMM_CELL_NUMBA_THREADS = 1


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


def _ccp_sequence_worker(arguments):
    mgs.get_ccp_sequence(*arguments)
    return {"type_id": int(arguments[3]), "x1_index": int(arguments[0])}


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


def _aggregate_task_observations(results):
    aggregated = {}
    count_fields = (
        "arrays", "size", "finite", "nan", "positive_inf", "negative_inf", "zero"
    )
    for result in results:
        for name, observation in result.get("observations", {}).items():
            target = aggregated.setdefault(
                name,
                {
                    **{field: 0 for field in count_fields},
                    "finite_min": None,
                    "finite_max": None,
                },
            )
            for field in count_fields:
                target[field] += int(observation.get(field, 0))
            minimum = observation.get("finite_min")
            maximum = observation.get("finite_max")
            if minimum is not None and (
                target["finite_min"] is None or minimum < target["finite_min"]
            ):
                target["finite_min"] = float(minimum)
            if maximum is not None and (
                target["finite_max"] is None or maximum > target["finite_max"]
            ):
                target["finite_max"] = float(maximum)
    return aggregated


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 50:
            return value.tolist()
        return array_summary(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _numeric_mapping_summaries(value, prefix="budget"):
    summaries = []
    if isinstance(value, dict):
        for key, nested in value.items():
            summaries.extend(_numeric_mapping_summaries(nested, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            summaries.extend(_numeric_mapping_summaries(nested, f"{prefix}[{index}]"))
    else:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            summaries.append({"name": prefix, **array_summary(array)})
    return summaries


def _save_budget_diagnostics(iteration_dir, result, data_summary):
    specification = bs.load(raise_if_missing=True)
    summaries = _numeric_mapping_summaries(specification)
    _write_json(iteration_dir / "budget_parameter_summary.json", summaries)
    failures = [
        row for row in summaries
        if row.get("nan", 0)
        or row.get("positive_inf", 0)
        or row.get("negative_inf", 0)
    ]
    if failures:
        _write_json(iteration_dir / "budget_parameter_failures.json", failures)
        raise RuntimeError(
            "Estimated budget parameters contain nonfinite values; see "
            "budget_parameter_failures.json"
        )

    for filename in ("budgetshock_params.npy", "budgetshock_bestx.npy"):
        source = Path(EST(filename))
        if source.is_file():
            shutil.copy2(source, iteration_dir / filename)

    if result is not None:
        result_fields = {
            key: value for key, value in dict(result).items()
            if key not in {"x", "jac", "hess", "hess_inv"}
        }
        _write_json(iteration_dir / "budget_optimizer_result.json", result_fields)
        if hasattr(result, "x"):
            np.save(iteration_dir / "budget_optimizer_x.npy", np.asarray(result.x))
        if hasattr(result, "jac") and result.jac is not None:
            np.save(iteration_dir / "budget_optimizer_jac.npy", np.asarray(result.jac))
        if hasattr(result, "fun") and not np.all(np.isfinite(np.asarray(result.fun))):
            raise RuntimeError(
                "Budget-shock optimizer returned a nonfinite objective; see "
                "budget_optimizer_result.json"
            )
    if hasattr(data_summary, "to_csv"):
        data_summary.to_csv(iteration_dir / "budget_data_summary.csv", index=False)
    elif data_summary is not None:
        _write_json(iteration_dir / "budget_data_summary.json", data_summary)
    return summaries


class OptimizerTrace:
    """Store every structural optimizer iterate without changing its decisions."""

    def __init__(self, iteration_dir):
        self.iteration_dir = Path(iteration_dir)
        self.count = 0
        self.trace_file = self.iteration_dir / "optimizer_trace.jsonl"

    def __call__(self, x):
        self.count += 1
        x = np.asarray(x, dtype=float)
        np.save(self.iteration_dir / "optimizer_latest_x.npy", x)
        with self.trace_file.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "optimizer_iteration": self.count,
                        "parameter_summary": array_summary(x),
                    }
                )
                + "\n"
            )
        # Preserve the production callback side effect as well.
        me.store(x)


class CheckedLikelihood:
    """Evaluate the production likelihood while recording every trial point."""

    def __init__(self, iteration_dir, loaded, q):
        self.iteration_dir = Path(iteration_dir)
        self.loaded = loaded
        self.q = q
        self.count = 0
        self.trace_file = self.iteration_dir / "likelihood_evaluations.jsonl"

    def __call__(self, x):
        self.count += 1
        x = np.asarray(x, dtype=float)
        value, gradient = me.likelihood(x, *self.loaded, self.q)
        gradient = np.asarray(gradient, dtype=float)
        finite = bool(np.isfinite(value) and np.all(np.isfinite(gradient)))
        with self.trace_file.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "evaluation": self.count,
                        "value": _json_safe(float(value)),
                        "finite": finite,
                        "parameters": array_summary(x),
                        "gradient": array_summary(gradient),
                    }
                )
                + "\n"
            )
        if not finite:
            np.save(self.iteration_dir / "failing_likelihood_x.npy", x)
            np.save(
                self.iteration_dir / "failing_likelihood_gradient.npy", gradient
            )
            _write_json(
                self.iteration_dir / "failing_likelihood_evaluation.json",
                {
                    "evaluation": self.count,
                    "value": float(value),
                    "parameters": array_summary(x),
                    "gradient": array_summary(gradient),
                },
            )
            import diagnose_likelihood_inputs as likelihood_diagnostics

            cell_summary, row_details, array_details = (
                likelihood_diagnostics.diagnose_likelihood_inputs(
                    self.q, x, worst_rows=100, prepared_arrays=self.loaded
                )
            )
            cell_summary.to_csv(
                self.iteration_dir / "failing_likelihood_cell_summary.csv",
                index=False,
            )
            row_details.to_csv(
                self.iteration_dir / "failing_likelihood_rows.csv", index=False
            )
            array_details.to_csv(
                self.iteration_dir / "failing_likelihood_arrays.csv", index=False
            )
            raise RuntimeError(
                f"Likelihood evaluation {self.count} is nonfinite; see "
                "failing_likelihood_evaluation.json"
            )
        return value, gradient


def _prepare_and_diagnose_likelihood(iteration_dir, workers, x0, q):
    print("Preparing checked likelihood inputs", flush=True)
    with mp.Pool(processes=min(workers, ms.T - 1)) as pool:
        pool.map(me.prepare_vjt_feasible, range(1, ms.T))
    loaded = me.load_all_arrays_feasible()
    summaries = _summarize_prepared_arrays(loaded)
    _write_json(iteration_dir / "prepared_array_summary.json", summaries)
    nonfinite = [
        row for row in summaries
        if row.get("nan", 0) or row.get("positive_inf", 0)
    ]
    if nonfinite:
        _write_json(iteration_dir / "prepared_array_failures.json", nonfinite)
        raise RuntimeError(
            f"Prepared likelihood inputs contain NaN/+Inf in {len(nonfinite)} "
            "arrays; see prepared_array_failures.json"
        )

    import diagnose_likelihood_inputs as likelihood_diagnostics

    cell_summary, row_details, array_details = (
        likelihood_diagnostics.diagnose_likelihood_inputs(
            q, x0, worst_rows=50, prepared_arrays=loaded
        )
    )
    cell_summary.to_csv(iteration_dir / "likelihood_cell_summary.csv", index=False)
    row_details.to_csv(iteration_dir / "likelihood_problem_rows.csv", index=False)
    array_details.to_csv(iteration_dir / "likelihood_array_summary.csv", index=False)
    probability_failures = cell_summary[
        (cell_summary["positive_q_with_nonfinite_stable_log_probability"] > 0)
        | (cell_summary["positive_q_with_zero_legacy_probability"] > 0)
        | (cell_summary["chosen_invalid"] > 0)
    ]
    if not probability_failures.empty:
        probability_failures.to_csv(
            iteration_dir / "likelihood_probability_failures.csv", index=False
        )
        raise RuntimeError(
            "Chosen alternatives create nonfinite log probabilities; see "
            "likelihood_problem_rows.csv"
        )
    return loaded


def _optimize_structural_likelihood(iteration_dir, x0, loaded, q):
    print("Optimizing the structural likelihood", flush=True)
    checked_likelihood = CheckedLikelihood(iteration_dir, loaded, q)
    start_value, start_gradient = checked_likelihood(x0)
    start_result = {
        "value": float(start_value),
        "value_finite": bool(np.isfinite(start_value)),
        "gradient": array_summary(start_gradient),
    }
    _write_json(iteration_dir / "likelihood_start.json", start_result)
    if not np.isfinite(start_value) or not np.all(np.isfinite(start_gradient)):
        raise RuntimeError("Starting likelihood value or gradient is nonfinite.")

    callback = OptimizerTrace(iteration_dir)
    result = minimize(
        checked_likelihood,
        x0,
        jac=True,
        options={"disp": True},
        callback=callback,
    )
    result_x = np.asarray(result.x, dtype=float)
    result_jac = np.asarray(result.jac, dtype=float)
    np.save(iteration_dir / "optimizer_result_x.npy", result_x)
    np.save(iteration_dir / "optimizer_result_jac.npy", result_jac)
    _write_json(
        iteration_dir / "optimizer_result.json",
        {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "fun": float(result.fun),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
            "njev": int(getattr(result, "njev", -1)),
            "x": array_summary(result_x),
            "jac": array_summary(result_jac),
            "callback_iterations": callback.count,
            "likelihood_evaluations": checked_likelihood.count,
        },
    )
    if (
        not np.isfinite(result.fun)
        or not np.all(np.isfinite(result_x))
        or not np.all(np.isfinite(result_jac))
    ):
        raise RuntimeError("Structural optimizer returned a nonfinite result.")
    return result


def _iteration_debug_config(args, iteration_dir):
    return DebugConfig(
        output_dir=str(iteration_dir),
        fail_fast=not args.continue_after_errors,
        max_failures=args.max_failures_per_task,
        trace_draws=True,
        verify_saved=not args.no_save_verification,
    )


def _run_full_estimation(args, output_dir):
    """Mirror the production NPL loop with checks and iteration snapshots."""
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one")

    all_types = tuple(TYPE_IDS)
    all_x1 = tuple(range(len(ms.invariant_states)))
    debt = np.asarray(ms.debt_range, dtype=float)
    q = load_em_posteriors(EST("auxiliary_em_results.npz"))
    x0 = np.asarray(np.load(EST("param_g.npy"), allow_pickle=False), dtype=float)

    print("Simulating/checking model state grids", flush=True)
    ms.simulate_all_states(11)
    print("Preparing fixed feasible/design arrays", flush=True)
    me.get_feasible()
    me.get_feasible_pubid()
    me.get_x_g_superfeasible()

    # Production only constructs the auxiliary initial CCPs before the NPL
    # loop.  Later iterations overwrite these bundles with model CCPs.
    iteration_zero_dir = output_dir / "iteration_00"
    iteration_zero_dir.mkdir(parents=True, exist_ok=True)
    initial_config = _iteration_debug_config(args, iteration_zero_dir)
    print("Building checked auxiliary initial CCPs", flush=True)
    auxiliary_parameters = {
        type_id: mccp.load_utility_parameters(type_id) for type_id in all_types
    }
    initial_arguments = [
        (
            x1_index, ms.invariant_states, debt,
            auxiliary_parameters[type_id], type_id, initial_config,
        )
        for type_id in all_types
        for x1_index in all_x1
    ]
    initial_results = _run_parallel(
        _initial_ccp_worker, initial_arguments, args.workers, "Initial CCP DEBUG"
    )
    _write_json(iteration_zero_dir / "initial_ccp_task_results.json", initial_results)
    _write_json(
        iteration_zero_dir / "initial_ccp_numeric_summary.json",
        _aggregate_task_observations(initial_results),
    )
    mccp.ensure_initial_ccps(ms.invariant_states, debt, auxiliary_parameters)

    completed_iterations = []
    for iteration in range(args.iterations):
        iteration_started = time.perf_counter()
        iteration_dir = output_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        debug_config = _iteration_debug_config(args, iteration_dir)
        ccp_real = 0 if iteration == 0 else 1
        np.save(iteration_dir / "x0_start.npy", x0)
        _write_json(
            iteration_dir / "iteration_metadata.json",
            {
                "iteration": iteration,
                "ccp_real": ccp_real,
                "ccp_source": "auxiliary_initial" if iteration == 0 else "model_updated",
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workers": args.workers,
                "budget_estimated": not args.skip_budget_estimation,
            },
        )
        print(
            f"\n========== CHECKED NPL ITERATION {iteration}/{args.iterations - 1} "
            f"| ccp_real={ccp_real} ==========",
            flush=True,
        )
        try:
            utility_parameters = {
                type_id: ms.build_param_g(type_id, x0) for type_id in all_types
            }

            if not args.skip_budget_estimation:
                print("Building sequence to estimate the budget shock", flush=True)
                sequence_arguments = [
                    (x1_index, ms.invariant_states, debt, type_id)
                    for type_id in all_types
                    for x1_index in all_x1
                ]
                sequence_results = _run_parallel(
                    _ccp_sequence_worker,
                    sequence_arguments,
                    args.workers,
                    f"Budget CCP sequence it={iteration}",
                )
                _write_json(
                    iteration_dir / "budget_ccp_sequence_tasks.json", sequence_results
                )
                print("Estimating the budget shock", flush=True)
                budget_result, budget_data = estimate_budget_shock_all_education(
                    draws=args.budget_draws,
                    maxiter=args.budget_maxiter,
                    optimizer="hybrid",
                    annealing_maxfun=args.budget_annealing_maxfun,
                    moment_spec="flow_plus_stock",
                    resource_mode="simulated",
                    restart=True,
                    ccp_workers=args.workers,
                    cell_workers=BUDGET_SMM_CELL_WORKERS,
                    cell_numba_threads=BUDGET_SMM_CELL_NUMBA_THREADS,
                    ccp_cache_mode="off",
                )
                _save_budget_diagnostics(
                    iteration_dir, budget_result, budget_data
                )
                ms.reload_budgetshock_params()
            else:
                print("Using saved budget-shock parameters", flush=True)
                _save_budget_diagnostics(iteration_dir, None, None)
                ms.reload_budgetshock_params()

            print("Solving and checking the model VJTs, CCPs, and EVTs", flush=True)
            bellman_arguments = [
                (
                    x1_index, ms.invariant_states, debt, debt, ccp_real,
                    utility_parameters[type_id], 0, 0, 0, type_id, True,
                    debug_config,
                )
                for type_id in all_types
                for x1_index in all_x1
            ]
            bellman_results = _run_parallel(
                _bellman_worker,
                bellman_arguments,
                args.workers,
                f"Bellman DEBUG it={iteration}",
                initializer=_bellman_initializer,
            )
            _write_json(
                iteration_dir / "bellman_task_results.json", bellman_results
            )
            _write_json(
                iteration_dir / "bellman_numeric_summary.json",
                _aggregate_task_observations(bellman_results),
            )

            loaded = _prepare_and_diagnose_likelihood(
                iteration_dir, args.workers, x0, q
            )
            result = _optimize_structural_likelihood(
                iteration_dir, x0, loaded, q
            )

            param_g = np.asarray(result.x, dtype=float)
            x0_next = param_g * 0.7 + x0 * 0.3
            np.save(iteration_dir / "param_g_optimizer.npy", param_g)
            np.save(iteration_dir / "x0_next_mixed.npy", x0_next)

            # Preserve all production estimation side effects and filenames.
            np.save(EST(f"estimates_it{iteration}_sigma_est.npy"), param_g)
            np.save(EST("param_g.npy"), param_g)
            with np.errstate(invalid="ignore"):
                standard_errors = np.diag(np.sqrt(np.asarray(result.hess_inv)))
            np.save(iteration_dir / "standard_errors.npy", standard_errors)
            np.save(EST(f"se_it{iteration}_sigma_est.npy"), standard_errors)
            np.save(
                EST(f"likelihood_it{iteration}_sigma_est.npy"),
                np.asarray(result.fun),
            )

            iteration_summary = {
                "iteration": iteration,
                "ccp_real": ccp_real,
                "objective": float(result.fun),
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "optimizer_message": str(result.message),
                "elapsed_seconds": time.perf_counter() - iteration_started,
                "bellman_tasks": len(bellman_results),
                "bellman_failures": int(
                    sum(item.get("failures", 0) for item in bellman_results)
                ),
            }
            _write_json(iteration_dir / "iteration_summary.json", iteration_summary)
            completed_iterations.append(iteration_summary)
            _write_json(output_dir / "completed_iterations.json", completed_iterations)
            x0 = x0_next
        except Exception:
            (iteration_dir / "fatal_exception.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            _write_json(output_dir / "completed_iterations.json", completed_iterations)
            raise

    return completed_iterations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("initial-ccp", "bellman", "critical", "likelihood", "full"),
        default="critical",
        help=(
            "critical rebuilds initial CCPs and solves the Bellman recursion; "
            "full runs the complete checked multi-iteration NPL estimation."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Aguirregabiria-Mira/NPL iterations for --stage full (default: 30).",
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
    parser.add_argument("--budget-draws", type=int, default=BUDGET_SMM_DRAWS)
    parser.add_argument("--budget-maxiter", type=int, default=BUDGET_SMM_MAXITER)
    parser.add_argument(
        "--budget-annealing-maxfun",
        type=int,
        default=BUDGET_SMM_ANNEALING_MAXFUN,
    )
    parser.add_argument(
        "--skip-budget-estimation",
        action="store_true",
        help="Use the saved budget parameters instead of reproducing production SMM.",
    )
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
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one")

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
        "iterations": args.iterations if args.stage == "full" else None,
        "budget_estimated": (
            not args.skip_budget_estimation if args.stage == "full" else None
        ),
        "budget_draws": args.budget_draws,
        "budget_maxiter": args.budget_maxiter,
        "budget_annealing_maxfun": args.budget_annealing_maxfun,
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
        if args.stage == "full":
            if selected_types != tuple(TYPE_IDS) or selected_x1 != tuple(
                range(len(ms.invariant_states))
            ):
                raise ValueError(
                    "The complete NPL estimation requires --types all and "
                    "--x1-indices all. Use critical/bellman for targeted checks."
                )
            completed = _run_full_estimation(args, output_dir)
            _write_json(output_dir / "full_estimation_results.json", completed)
            print(
                f"Checked NPL estimation completed {len(completed)} iterations. "
                f"Results: {output_dir}",
                flush=True,
            )
            return

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
