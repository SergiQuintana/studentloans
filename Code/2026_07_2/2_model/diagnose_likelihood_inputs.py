"""Read-only diagnostics for structural-likelihood inputs.

Run this after ``prepare_vjt_feasible`` and before structural optimization.  It
does not modify estimates, CCPs, VJTs, or likelihood inputs.  Diagnostic CSVs
are written below ``Model/Output/likelihood/diagnostics`` by default.

Examples
--------
Core likelihood-input checks::

    python diagnose_likelihood_inputs.py

Also inspect the home-CCP bundles used by the Bellman recursion::

    python diagnose_likelihood_inputs.py --check-ccps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import budget_shock as budget_shock
import model_em_algorithm as model
from config import EST, LIK, OUT, RDATA
from latent_types import TYPE_IDS, TYPE_NAMES, load_em_posteriors


def _shape_text(array):
    return "x".join(str(value) for value in np.shape(array))


def _finite_min(array):
    values = np.asarray(array)[np.isfinite(array)]
    return float(np.min(values)) if values.size else np.nan


def _finite_max(array):
    values = np.asarray(array)[np.isfinite(array)]
    return float(np.max(values)) if values.size else np.nan


def summarize_array(name, array, period=None):
    array = np.asarray(array)
    numeric = np.issubdtype(array.dtype, np.number)
    result = {
        "period": period,
        "array": name,
        "shape": _shape_text(array),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if numeric:
        result.update(
            {
                "finite": int(np.count_nonzero(np.isfinite(array))),
                "nan": int(np.count_nonzero(np.isnan(array))),
                "positive_inf": int(np.count_nonzero(np.isposinf(array))),
                "negative_inf": int(np.count_nonzero(np.isneginf(array))),
                "finite_min": _finite_min(array),
                "finite_max": _finite_max(array),
            }
        )
    return result


def _weighted_loglike(log_probability, weights):
    positive = np.asarray(weights) > 0.0
    if not np.any(positive):
        return 0.0
    contribution = np.asarray(log_probability)[positive] * np.asarray(weights)[positive]
    if np.any(np.isnan(contribution)):
        return np.nan
    return float(np.sum(contribution))


def analyze_likelihood_cell(vjt, g, choice_indices, type_weights):
    """Classify one period/type cell without changing its inputs."""
    vjt = np.asarray(vjt, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    choice_indices = np.asarray(choice_indices, dtype=np.int64).reshape(-1)
    type_weights = np.asarray(type_weights, dtype=np.float64).reshape(-1)

    if vjt.shape != g.shape:
        raise ValueError(f"VJT shape {vjt.shape} does not match g shape {g.shape}.")
    if vjt.shape[0] != choice_indices.size or vjt.shape[0] != type_weights.size:
        raise ValueError("VJT rows, choices, and posterior weights must align.")
    if np.any((choice_indices < 0) | (choice_indices >= vjt.shape[1])):
        raise ValueError("Observed choice indices fall outside the VJT columns.")

    rows = np.arange(vjt.shape[0])
    utility = vjt + g
    chosen_vjt = vjt[rows, choice_indices]
    chosen_g = g[rows, choice_indices]
    chosen_utility = utility[rows, choice_indices]
    home_utility = utility[:, -1]

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        log_denominator = logsumexp(utility, axis=1)
        stable_log_probability = chosen_utility - log_denominator
        normalized = utility - home_utility[:, None]
        exponentiated = np.exp(normalized)
        denominator = np.sum(exponentiated, axis=1)
        legacy_probability = exponentiated[rows, choice_indices] / denominator
        legacy_log_probability = np.log(legacy_probability)

    finite_utility = np.isfinite(utility)
    finite_choices_per_row = np.sum(finite_utility, axis=1)
    legacy_zero = legacy_probability == 0.0
    numerical_underflow = legacy_zero & np.isfinite(stable_log_probability)
    chosen_infeasible = np.isneginf(chosen_utility)
    chosen_invalid = np.isnan(chosen_utility) | np.isposinf(chosen_utility)
    positive_weight = type_weights > 0.0
    problematic = (
        numerical_underflow
        | chosen_infeasible
        | chosen_invalid
        | ~np.isfinite(stable_log_probability)
        | ~np.isfinite(legacy_probability)
        | (finite_choices_per_row == 0)
    )

    maximum_utility = np.max(utility, axis=1)
    chosen_minus_maximum = chosen_utility - maximum_utility
    chosen_minus_home = chosen_utility - home_utility

    summary = {
        "observations": int(vjt.shape[0]),
        "choices": int(vjt.shape[1]),
        "vjt_finite": int(np.count_nonzero(np.isfinite(vjt))),
        "vjt_nan": int(np.count_nonzero(np.isnan(vjt))),
        "vjt_positive_inf": int(np.count_nonzero(np.isposinf(vjt))),
        "vjt_negative_inf": int(np.count_nonzero(np.isneginf(vjt))),
        "rows_without_finite_vjt": int(np.count_nonzero(finite_choices_per_row == 0)),
        "minimum_finite_choices_per_row": int(np.min(finite_choices_per_row)),
        "maximum_finite_choices_per_row": int(np.max(finite_choices_per_row)),
        "chosen_vjt_negative_inf": int(np.count_nonzero(np.isneginf(chosen_vjt))),
        "chosen_vjt_nan": int(np.count_nonzero(np.isnan(chosen_vjt))),
        "chosen_vjt_positive_inf": int(np.count_nonzero(np.isposinf(chosen_vjt))),
        "home_utility_nonfinite": int(np.count_nonzero(~np.isfinite(home_utility))),
        "g_nonfinite": int(np.count_nonzero(~np.isfinite(g))),
        "utility_nan": int(np.count_nonzero(np.isnan(utility))),
        "utility_positive_inf": int(np.count_nonzero(np.isposinf(utility))),
        "legacy_probability_zero": int(np.count_nonzero(legacy_zero)),
        "legacy_probability_nonfinite": int(
            np.count_nonzero(~np.isfinite(legacy_probability))
        ),
        "numerical_underflow": int(np.count_nonzero(numerical_underflow)),
        "chosen_infeasible": int(np.count_nonzero(chosen_infeasible)),
        "chosen_invalid": int(np.count_nonzero(chosen_invalid)),
        "stable_log_probability_nonfinite": int(
            np.count_nonzero(~np.isfinite(stable_log_probability))
        ),
        "positive_q_with_zero_legacy_probability": int(
            np.count_nonzero(positive_weight & legacy_zero)
        ),
        "positive_q_with_nonfinite_stable_log_probability": int(
            np.count_nonzero(positive_weight & ~np.isfinite(stable_log_probability))
        ),
        "problematic_rows": int(np.count_nonzero(problematic)),
        "vjt_finite_min": _finite_min(vjt),
        "vjt_finite_max": _finite_max(vjt),
        "g_finite_min": _finite_min(g),
        "g_finite_max": _finite_max(g),
        "utility_finite_min": _finite_min(utility),
        "utility_finite_max": _finite_max(utility),
        "chosen_minus_home_min": _finite_min(chosen_minus_home),
        "chosen_minus_home_max": _finite_max(chosen_minus_home),
        "chosen_minus_maximum_min": _finite_min(chosen_minus_maximum),
        "stable_log_probability_min": _finite_min(stable_log_probability),
        "stable_log_probability_max": _finite_max(stable_log_probability),
        "legacy_probability_min": _finite_min(legacy_probability),
        "legacy_probability_max": _finite_max(legacy_probability),
        "stable_weighted_loglike": _weighted_loglike(
            stable_log_probability, type_weights
        ),
        "legacy_weighted_loglike": _weighted_loglike(
            legacy_log_probability, type_weights
        ),
        "q_positive_rows": int(np.count_nonzero(positive_weight)),
        "q_zero_rows": int(np.count_nonzero(type_weights == 0.0)),
        "q_mass": float(np.sum(type_weights)),
        "q_min_positive": _finite_min(type_weights[positive_weight]),
        "q_max": _finite_max(type_weights),
    }
    arrays = {
        "utility": utility,
        "chosen_vjt": chosen_vjt,
        "chosen_g": chosen_g,
        "chosen_utility": chosen_utility,
        "home_utility": home_utility,
        "maximum_utility": maximum_utility,
        "chosen_minus_home": chosen_minus_home,
        "chosen_minus_maximum": chosen_minus_maximum,
        "stable_log_probability": stable_log_probability,
        "legacy_probability": legacy_probability,
        "numerical_underflow": numerical_underflow,
        "chosen_infeasible": chosen_infeasible,
        "chosen_invalid": chosen_invalid,
        "problematic": problematic,
    }
    return summary, arrays


def _problem_reason(arrays, row):
    reasons = []
    if arrays["numerical_underflow"][row]:
        reasons.append("numerical_underflow")
    if arrays["chosen_infeasible"][row]:
        reasons.append("chosen_infeasible")
    if arrays["chosen_invalid"][row]:
        reasons.append("chosen_nan_or_positive_inf")
    if not np.isfinite(arrays["stable_log_probability"][row]):
        reasons.append("stable_log_probability_nonfinite")
    if not np.isfinite(arrays["legacy_probability"][row]):
        reasons.append("legacy_probability_nonfinite")
    return ";".join(reasons) if reasons else "small_stable_probability"


def _selected_worst_rows(arrays, limit):
    stable = arrays["stable_log_probability"]
    problem_rows = np.flatnonzero(arrays["problematic"])
    finite_rows = np.flatnonzero(np.isfinite(stable) & ~arrays["problematic"])
    problem_order = problem_rows[np.argsort(np.nan_to_num(stable[problem_rows], nan=-np.inf))]
    finite_order = finite_rows[np.argsort(stable[finite_rows])]
    return np.concatenate((problem_order, finite_order))[:limit]


def diagnose_likelihood_inputs(q, x0, worst_rows=10):
    (
        choices_all,
        vjt_all_types,
        x1_new,
        choices_array_all,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    ) = model.load_all_arrays_feasible()

    utility_parameters = model.build_param_g(x0)
    summaries = []
    details = []
    array_summaries = [summarize_array("q", q), summarize_array("x0", x0)]
    total_choices = model.get_total_choices()

    named_npz = {
        "x1_new": x1_new,
        "x_change": x_change,
        "x_educ": x_educ,
        "x_first2": x_first2,
        "x_first4": x_first4,
        "x_firstgrad": x_firstgrad,
        "x_exp": x_exp,
    }

    for period in range(1, model.T):
        period_index = period - 1
        choices = np.asarray(choices_all[period_index])
        choice_indices = np.asarray(choices_array_all[period_index], dtype=np.int64)
        x1_data, state, debt, loaded_choices = model.load_data_superfeasible(
            period, return_income=False
        )
        full_x1 = np.load(
            RDATA(f"invariant_state_superfeasible_t{period}.npy"),
            allow_pickle=False,
        )
        pubid = np.asarray(full_x1)[:, 0]

        if not np.array_equal(choices, loaded_choices):
            raise ValueError(f"Choice arrays are misaligned in period {period}.")
        if len(choices) != len(q):
            raise ValueError(
                f"Period {period} has {len(choices)} observations but q has {len(q)}."
            )

        valid_index = (choice_indices >= 0) & (choice_indices < len(total_choices))
        choice_mapping_mismatches = int(np.count_nonzero(~valid_index))
        if np.all(valid_index):
            choice_mapping_mismatches += int(
                np.count_nonzero(
                    ~np.all(total_choices[choice_indices] == choices, axis=1)
                )
            )

        array_summaries.extend(
            [
                summarize_array("choices", choices, period),
                summarize_array("choice_indices", choice_indices, period),
                summarize_array("state", state, period),
                summarize_array("debt_indices", debt, period),
            ]
        )
        for name, archive in named_npz.items():
            array_summaries.append(
                summarize_array(name, archive[f"period{period}"], period)
            )

        period_arrays = {
            "x1_new": x1_new[f"period{period}"],
            "x_change": x_change[f"period{period}"],
            "x_educ": x_educ[f"period{period}"],
            "x_first2": x_first2[f"period{period}"],
            "x_first4": x_first4[f"period{period}"],
            "x_firstgrad": x_firstgrad[f"period{period}"],
            "x_exp": x_exp[f"period{period}"],
        }

        for type_index, type_id in enumerate(TYPE_IDS):
            vjt = vjt_all_types[type_index][period_index]
            g = model.get_all_g(
                utility_parameters,
                period_arrays["x1_new"],
                period_arrays["x_change"],
                period_arrays["x_educ"],
                period_arrays["x_first2"],
                period_arrays["x_first4"],
                period_arrays["x_firstgrad"],
                period_arrays["x_exp"],
                period,
                type_id,
            )
            summary, arrays = analyze_likelihood_cell(
                vjt, g, choice_indices, q[:, type_index]
            )
            summary.update(
                {
                    "period": period,
                    "type_id": type_id,
                    "type_name": str(TYPE_NAMES[type_index]),
                    "choice_mapping_mismatches": choice_mapping_mismatches,
                }
            )
            summaries.append(summary)

            for row in _selected_worst_rows(arrays, worst_rows):
                details.append(
                    {
                        "period": period,
                        "type_id": type_id,
                        "type_name": str(TYPE_NAMES[type_index]),
                        "row": int(row),
                        "pubid": pubid[row],
                        "reason": _problem_reason(arrays, row),
                        "q": float(q[row, type_index]),
                        "choice_index": int(choice_indices[row]),
                        "choice": np.array2string(np.asarray(choices[row])),
                        "state": np.array2string(np.asarray(state[row])),
                        "x1": np.array2string(np.asarray(x1_data[row])),
                        "debt_index": int(debt[row]),
                        "chosen_vjt": float(arrays["chosen_vjt"][row]),
                        "chosen_g": float(arrays["chosen_g"][row]),
                        "chosen_utility": float(arrays["chosen_utility"][row]),
                        "home_utility": float(arrays["home_utility"][row]),
                        "maximum_utility": float(arrays["maximum_utility"][row]),
                        "chosen_minus_home": float(arrays["chosen_minus_home"][row]),
                        "chosen_minus_maximum": float(
                            arrays["chosen_minus_maximum"][row]
                        ),
                        "stable_log_probability": float(
                            arrays["stable_log_probability"][row]
                        ),
                        "legacy_probability": float(arrays["legacy_probability"][row]),
                    }
                )

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(details),
        pd.DataFrame(array_summaries),
    )


def diagnose_ccp_bundles(worst_rows=10):
    """Inspect stored home CCPs at observed states and all their debt points."""
    summaries = []
    details = []

    for period in range(1, model.T):
        x1, state, debt, choices = model.load_data_superfeasible(
            period, return_income=False
        )
        x1 = np.asarray(x1, dtype=np.int64)
        state = np.asarray(state, dtype=np.int64)
        debt = np.asarray(debt, dtype=np.int64).reshape(-1)
        full_x1 = np.load(
            RDATA(f"invariant_state_superfeasible_t{period}.npy"),
            allow_pickle=False,
        )
        pubid = np.asarray(full_x1)[:, 0]
        unique_x1, x1_group = np.unique(x1, axis=0, return_inverse=True)

        for type_index, type_id in enumerate(TYPE_IDS):
            counters = {
                "period": period,
                "type_id": type_id,
                "type_name": str(TYPE_NAMES[type_index]),
                "bundles_expected": int(len(unique_x1)),
                "bundles_opened": 0,
                "bundles_missing": 0,
                "keys_expected": 0,
                "keys_missing": 0,
                "observed_state_grid_ccp_values": 0,
                "observed_state_grid_ccp_zero": 0,
                "observed_state_grid_ccp_one": 0,
                "observed_state_grid_ccp_nonfinite": 0,
                "observed_state_grid_ccp_below_zero": 0,
                "observed_state_grid_ccp_above_one": 0,
                "observed_ccp_values": 0,
                "observed_ccp_zero": 0,
                "observed_ccp_nonfinite": 0,
                "observed_ccp_below_zero": 0,
                "observed_ccp_above_one": 0,
                "observed_state_grid_ccp_min": np.nan,
                "observed_state_grid_ccp_max": np.nan,
                "observed_ccp_min": np.nan,
                "observed_ccp_max": np.nan,
            }
            all_finite = []
            observed_finite = []
            type_details = []

            for group, x1i in enumerate(unique_x1):
                group_rows = np.flatnonzero(x1_group == group)
                path = Path(OUT("ccp", period, f"ccp_t{period}_[{x1i}]_em{type_id}.npz"))
                if not path.is_file():
                    counters["bundles_missing"] += 1
                    continue
                counters["bundles_opened"] += 1

                with np.load(path, allow_pickle=False) as bundle:
                    group_states = state[group_rows]
                    for x2i in np.unique(group_states, axis=0):
                        rows = group_rows[np.all(group_states == x2i, axis=1)]
                        key = f"ccp_t{period}_[{x1i}]_{x2i}"
                        counters["keys_expected"] += 1
                        if key not in bundle.files:
                            counters["keys_missing"] += 1
                            continue

                        ccp = np.asarray(bundle[key], dtype=np.float64).reshape(-1)
                        counters["observed_state_grid_ccp_values"] += int(ccp.size)
                        counters["observed_state_grid_ccp_zero"] += int(
                            np.count_nonzero(ccp == 0.0)
                        )
                        counters["observed_state_grid_ccp_one"] += int(
                            np.count_nonzero(ccp == 1.0)
                        )
                        counters["observed_state_grid_ccp_nonfinite"] += int(
                            np.count_nonzero(~np.isfinite(ccp))
                        )
                        counters["observed_state_grid_ccp_below_zero"] += int(
                            np.count_nonzero(ccp < 0.0)
                        )
                        counters["observed_state_grid_ccp_above_one"] += int(
                            np.count_nonzero(ccp > 1.0)
                        )
                        finite = ccp[np.isfinite(ccp)]
                        if finite.size:
                            all_finite.append(finite)

                        if np.any((debt[rows] < 0) | (debt[rows] >= ccp.size)):
                            raise ValueError(
                                f"Observed debt index is outside CCP key {key} in {path}."
                            )
                        observed = ccp[debt[rows]]
                        counters["observed_ccp_values"] += int(observed.size)
                        counters["observed_ccp_zero"] += int(
                            np.count_nonzero(observed == 0.0)
                        )
                        counters["observed_ccp_nonfinite"] += int(
                            np.count_nonzero(~np.isfinite(observed))
                        )
                        counters["observed_ccp_below_zero"] += int(
                            np.count_nonzero(observed < 0.0)
                        )
                        counters["observed_ccp_above_one"] += int(
                            np.count_nonzero(observed > 1.0)
                        )
                        finite_observed = observed[np.isfinite(observed)]
                        if finite_observed.size:
                            observed_finite.append(finite_observed)

                        bad = (
                            ~np.isfinite(observed)
                            | (observed <= 0.0)
                            | (observed > 1.0)
                        )
                        for local_index in np.flatnonzero(bad):
                            row = rows[local_index]
                            type_details.append(
                                {
                                    "period": period,
                                    "type_id": type_id,
                                    "type_name": str(TYPE_NAMES[type_index]),
                                    "row": int(row),
                                    "pubid": pubid[row],
                                    "ccp_home": float(observed[local_index]),
                                    "debt_index": int(debt[row]),
                                    "choice": np.array2string(np.asarray(choices[row])),
                                    "state": np.array2string(np.asarray(state[row])),
                                    "x1": np.array2string(np.asarray(x1[row])),
                                    "bundle": str(path),
                                    "key": key,
                                }
                            )

            if all_finite:
                values = np.concatenate(all_finite)
                counters["observed_state_grid_ccp_min"] = float(np.min(values))
                counters["observed_state_grid_ccp_max"] = float(np.max(values))
            if observed_finite:
                values = np.concatenate(observed_finite)
                counters["observed_ccp_min"] = float(np.min(values))
                counters["observed_ccp_max"] = float(np.max(values))
            summaries.append(counters)
            details.extend(type_details[:worst_rows])

    return pd.DataFrame(summaries), pd.DataFrame(details)


def diagnose_budget_parameters():
    """Summarize the fitted budget-shock arrays used by the Bellman solver."""
    specification = budget_shock.load(raise_if_missing=False)
    if specification is None:
        return pd.DataFrame(
            [
                {
                    "parameter": "budgetshock_params.npy",
                    "shape": "",
                    "size": 0,
                    "finite": 0,
                    "nonfinite": 0,
                    "finite_min": np.nan,
                    "finite_max": np.nan,
                    "values": "missing",
                }
            ]
        )

    rows = []
    for name, value in specification.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            continue
        finite = np.isfinite(array)
        rows.append(
            {
                "parameter": name,
                "shape": _shape_text(array),
                "size": int(array.size),
                "finite": int(np.count_nonzero(finite)),
                "nonfinite": int(np.count_nonzero(~finite)),
                "finite_min": _finite_min(array),
                "finite_max": _finite_max(array),
                "values": (
                    np.array2string(array, threshold=20)
                    if array.size <= 20
                    else ""
                ),
            }
        )

    if "debt_pen_parinc" in specification:
        parental_groups = np.column_stack(
            (np.arange(1, 5, dtype=np.int64), np.ones(4, dtype=np.int64))
        )
        group_penalties = budget_shock.debt_penalty(
            specification, parental_groups
        )
        rows.append(
            {
                "parameter": "derived_debt_penalty_group_levels",
                "shape": _shape_text(group_penalties),
                "size": int(group_penalties.size),
                "finite": int(np.count_nonzero(np.isfinite(group_penalties))),
                "nonfinite": int(np.count_nonzero(~np.isfinite(group_penalties))),
                "finite_min": _finite_min(group_penalties),
                "finite_max": _finite_max(group_penalties),
                "values": np.array2string(group_penalties),
            }
        )
    return pd.DataFrame(rows)


def _print_core_summary(summary):
    columns = [
        "period",
        "type_id",
        "type_name",
        "chosen_vjt_negative_inf",
        "utility_nan",
        "utility_positive_inf",
        "numerical_underflow",
        "positive_q_with_zero_legacy_probability",
        "positive_q_with_nonfinite_stable_log_probability",
        "chosen_minus_maximum_min",
        "stable_weighted_loglike",
        "legacy_weighted_loglike",
    ]
    print("\nStructural likelihood diagnostics by period and type")
    print(summary[columns].to_string(index=False))
    totals = summary[
        [
            "chosen_vjt_negative_inf",
            "utility_nan",
            "utility_positive_inf",
            "numerical_underflow",
            "positive_q_with_zero_legacy_probability",
            "positive_q_with_nonfinite_stable_log_probability",
            "choice_mapping_mismatches",
        ]
    ].sum()
    print("\nProblem counts summed over period/type cells")
    print(totals.to_string())


def _print_ccp_summary(summary):
    columns = [
        "period",
        "type_id",
        "type_name",
        "bundles_missing",
        "keys_missing",
        "observed_state_grid_ccp_zero",
        "observed_state_grid_ccp_nonfinite",
        "observed_ccp_zero",
        "observed_ccp_nonfinite",
        "observed_state_grid_ccp_min",
        "observed_state_grid_ccp_max",
    ]
    print("\nHome-CCP diagnostics by period and type")
    print(summary[columns].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-ccps",
        action="store_true",
        help="Also scan grouped home-CCP bundles used by the Bellman recursion.",
    )
    parser.add_argument(
        "--worst-rows",
        type=int,
        default=10,
        help="Detailed rows retained per period/type (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default=LIK("diagnostics"),
        help="Directory for diagnostic CSV files.",
    )
    parser.add_argument(
        "--posterior-file",
        default=EST("auxiliary_em_results.npz"),
        help="Full joint-type auxiliary EM result.",
    )
    parser.add_argument(
        "--parameter-file",
        default=EST("param_g.npy"),
        help="Structural utility parameter vector evaluated before optimization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worst_rows < 1:
        raise ValueError("--worst-rows must be at least one.")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Loading posterior weights from {args.posterior_file}")
    q = load_em_posteriors(args.posterior_file)
    print(f"Loading structural parameters from {args.parameter_file}")
    x0 = np.asarray(np.load(args.parameter_file, allow_pickle=False), dtype=float)

    summary, details, arrays = diagnose_likelihood_inputs(
        q, x0, worst_rows=args.worst_rows
    )
    summary_path = output_directory / "likelihood_input_summary.csv"
    detail_path = output_directory / "likelihood_worst_rows.csv"
    array_path = output_directory / "likelihood_array_summary.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(detail_path, index=False)
    arrays.to_csv(array_path, index=False)
    budget_parameters = diagnose_budget_parameters()
    budget_path = output_directory / "budgetshock_parameter_summary.csv"
    budget_parameters.to_csv(budget_path, index=False)
    _print_core_summary(summary)
    print(f"\nSaved {summary_path}")
    print(f"Saved {detail_path}")
    print(f"Saved {array_path}")
    print(f"Saved {budget_path}")

    if args.check_ccps:
        ccp_summary, ccp_details = diagnose_ccp_bundles(
            worst_rows=args.worst_rows
        )
        ccp_summary_path = output_directory / "ccp_input_summary.csv"
        ccp_detail_path = output_directory / "ccp_problem_rows.csv"
        ccp_summary.to_csv(ccp_summary_path, index=False)
        ccp_details.to_csv(ccp_detail_path, index=False)
        _print_ccp_summary(ccp_summary)
        print(f"\nSaved {ccp_summary_path}")
        print(f"Saved {ccp_detail_path}")


if __name__ == "__main__":
    main()
