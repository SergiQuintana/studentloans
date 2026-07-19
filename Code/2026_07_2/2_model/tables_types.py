# -*- coding: utf-8 -*-
"""Posterior-weighted descriptive statistics for auxiliary EM latent types.

Individuals are never assigned to a single type.  For type k, observation i
receives weight q[i, k], where q is the posterior probability produced by the
sixteen-type auxiliary EM estimator.  The resulting tables are descriptive and
do not have a causal interpretation.

Run after (or during a checkpoint of) auxiliary estimation with::

    python Code/2026_07_2/2_model/tables_types.py

By default, tables are written below
``Model/Output/tables/type_descriptives`` (or the corresponding MODEL_ROOT).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import EST, OUT, RDATA


EXPECTED_TYPE_NAMES = tuple(
    f"S{school}G{grant}T{transfer}L{loan}"
    for school in (0, 1)
    for grant in (0, 1)
    for transfer in (0, 1)
    for loan in (0, 1)
)
EDUCATION = {
    1: ("Two-year", "twograd"),
    2: ("Four-year", "fourgrad"),
    3: ("Graduate", "gradgrad"),
}


def _decode_strings(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def _load_posteriors(results_path: Path):
    """Load the joint posterior and type layout from an EM npz result."""
    with np.load(results_path, allow_pickle=False) as results:
        required = {
            "q", "type_names", "type_school", "type_grant", "type_transfer", "type_loan"
        }
        missing = required.difference(results.files)
        if missing:
            raise ValueError(
                f"{results_path} is missing required arrays: {sorted(missing)}"
            )
        q = np.asarray(results["q"], dtype=float)
        names = _decode_strings(results["type_names"])
        school = np.asarray(results["type_school"], dtype=int).reshape(-1)
        grant = np.asarray(results["type_grant"], dtype=int).reshape(-1)
        transfer = np.asarray(results["type_transfer"], dtype=int).reshape(-1)
        loan = np.asarray(results["type_loan"], dtype=int).reshape(-1)
        history_length = len(results["observed_loglike"]) if "observed_loglike" in results else 1

    if q.ndim != 2:
        raise ValueError(f"Posterior q must be a matrix; received shape {q.shape}.")
    if q.shape[1] != len(names):
        raise ValueError("The number of posterior columns does not match type_names.")
    if not (len(names) == len(school) == len(grant) == len(transfer) == len(loan)):
        raise ValueError("Type-name and type-component arrays have inconsistent lengths.")
    if len(names) != 16 or set(names) != set(EXPECTED_TYPE_NAMES):
        raise ValueError(
            "This script requires the sixteen-type schooling x grant x parental-transfer "
            "x loan "
            f"posterior; found {names}. Use auxiliary_em_checkpoint.npz, not the "
            "two-column legacy em_q_typeff2.npy file."
        )
    if not np.all(np.isfinite(q)) or np.any(q < -1e-10):
        raise ValueError("Posterior probabilities contain invalid values.")
    row_sums = q.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-7):
        raise ValueError("Posterior probabilities do not sum to one within individual.")
    q = np.clip(q, 0.0, 1.0)
    q /= q.sum(axis=1, keepdims=True)
    return q, names, school, grant, transfer, loan, max(1, history_length - 1)


def _load_analysis_panel(data_path: Path, id_path: Path, q: np.ndarray) -> pd.DataFrame:
    """Merge the balanced real-data panel to q using the EM row-order IDs."""
    ids_array = np.load(id_path)
    if ids_array.ndim == 2:
        ids_array = ids_array[:, 0]
    ids = np.asarray(ids_array).reshape(-1)
    if len(ids) != len(q):
        raise ValueError(
            f"Posterior q has {len(q)} rows but {id_path} contains {len(ids)} IDs."
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("The EM individual-ID array contains duplicates.")

    required = {
        "PUBID",
        "period",
        "educ",
        "work",
        "currentloans",
        "auxiliary_loan_flow",
        "twograd",
        "fourgrad",
        "gradgrad",
    }
    panel = pd.read_stata(data_path)
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required variables: {sorted(missing)}")

    id_frame = pd.DataFrame({"PUBID": ids, "_em_row": np.arange(len(ids), dtype=int)})
    panel = panel.merge(id_frame, on="PUBID", how="inner", validate="many_to_one")
    observed_ids = panel["_em_row"].nunique()
    if observed_ids != len(ids):
        raise ValueError(
            f"Only {observed_ids} of the {len(ids)} EM individuals were found in {data_path}."
        )
    panel = panel.sort_values(["_em_row", "period"], kind="stable").reset_index(drop=True)
    for column in ("period", "educ", "work"):
        panel[column] = pd.to_numeric(panel[column], errors="raise").astype(int)
    for column in ("currentloans", "auxiliary_loan_flow"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel


def _group_weights(q, names, school, grant, transfer, loan):
    """Return all joint types and each two-category marginal type dimension."""
    groups = []
    for column, name in enumerate(names):
        groups.append(("Joint type", name, q[:, column]))
    for dimension, components, prefix in (
        ("Schooling type", school, "S"),
        ("Grant type", grant, "G"),
        ("Parental-transfer type", transfer, "T"),
        ("Loan type", loan, "L"),
    ):
        for value in (0, 1):
            groups.append((dimension, f"{prefix}{value}", q[:, components == value].sum(axis=1)))
    return groups


def _loan_type_groups(q, loan):
    """Return the two marginal posterior loan-type weights only."""
    return [
        ("Loan type", f"L{value}", q[:, loan == value].sum(axis=1))
        for value in (0, 1)
    ]


def _weights_for_rows(panel: pd.DataFrame, individual_weights: np.ndarray) -> np.ndarray:
    return individual_weights[panel["_em_row"].to_numpy(dtype=int)]


def _weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    denominator = weights[valid].sum()
    if denominator <= 0:
        return np.nan
    return float(np.dot(values[valid], weights[valid]) / denominator)


def _weighted_quantile(values, weights, probability) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = float(probability) * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, cutoff, side="left"), len(values) - 1)])


def _weighted_correlation(left, right, weights) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(left) & np.isfinite(right) & np.isfinite(weights) & (weights > 0)
    )
    if not np.any(valid):
        return np.nan
    left, right, weights = left[valid], right[valid], weights[valid]
    weight_sum = weights.sum()
    left_centered = left - np.dot(left, weights) / weight_sum
    right_centered = right - np.dot(right, weights) / weight_sum
    covariance = np.dot(weights, left_centered * right_centered) / weight_sum
    left_variance = np.dot(weights, left_centered ** 2) / weight_sum
    right_variance = np.dot(weights, right_centered ** 2) / weight_sum
    denominator = np.sqrt(left_variance * right_variance)
    return float(covariance / denominator) if denominator > 0 else np.nan


def _loan_type_posterior_tables(panel, q, loan):
    """Summarize whether marginal L posteriors are separated or near 50/50."""
    high = q[:, loan == 1].sum(axis=1)
    low = 1.0 - high
    confidence = np.maximum(low, high)
    entropy_terms = np.zeros_like(high)
    for probability in (low, high):
        positive = probability > 0.0
        entropy_terms[positive] -= probability[positive] * np.log(probability[positive])
    normalized_entropy = entropy_terms / np.log(2.0)

    summary = pd.DataFrame(
        [
            {
                "individuals": int(len(high)),
                "posterior_expected_L0_share": float(low.mean()),
                "posterior_expected_L1_share": float(high.mean()),
                "modal_L0_share": float(np.mean(high < 0.5)),
                "modal_L1_share": float(np.mean(high >= 0.5)),
                "mean_L1_posterior": float(high.mean()),
                "sd_L1_posterior": float(high.std(ddof=0)),
                "p01_L1_posterior": float(np.quantile(high, 0.01)),
                "p05_L1_posterior": float(np.quantile(high, 0.05)),
                "p10_L1_posterior": float(np.quantile(high, 0.10)),
                "p25_L1_posterior": float(np.quantile(high, 0.25)),
                "median_L1_posterior": float(np.quantile(high, 0.50)),
                "p75_L1_posterior": float(np.quantile(high, 0.75)),
                "p90_L1_posterior": float(np.quantile(high, 0.90)),
                "p95_L1_posterior": float(np.quantile(high, 0.95)),
                "p99_L1_posterior": float(np.quantile(high, 0.99)),
                "share_L1_posterior_below_0_10": float(np.mean(high <= 0.10)),
                "share_L1_posterior_above_0_90": float(np.mean(high >= 0.90)),
                "share_L1_posterior_between_0_40_and_0_60": float(
                    np.mean((high >= 0.40) & (high <= 0.60))
                ),
                "share_modal_probability_above_0_80": float(np.mean(confidence >= 0.80)),
                "share_modal_probability_above_0_90": float(np.mean(confidence >= 0.90)),
                "mean_normalized_binary_entropy": float(normalized_entropy.mean()),
            }
        ]
    )

    bin_edges = np.asarray(
        [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    )
    bin_index = np.searchsorted(bin_edges, high, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(bin_edges) - 2)
    histogram_rows = []
    for index in range(len(bin_edges) - 1):
        count = int(np.sum(bin_index == index))
        histogram_rows.append(
            {
                "L1_posterior_bin_left": float(bin_edges[index]),
                "L1_posterior_bin_right": float(bin_edges[index + 1]),
                "right_endpoint_included": bool(index == len(bin_edges) - 2),
                "individuals": count,
                "share_of_individuals": count / len(high),
            }
        )

    people = (
        panel[["_em_row", "PUBID"]]
        .drop_duplicates("_em_row")
        .sort_values("_em_row")
        .reset_index(drop=True)
    )
    if not np.array_equal(people["_em_row"].to_numpy(dtype=int), np.arange(len(q))):
        raise ValueError("The analysis panel does not contain one ordered ID for every q row.")
    individual = people.copy()
    individual["posterior_L0"] = low
    individual["posterior_L1"] = high
    individual["modal_loan_type"] = np.where(high >= 0.5, "L1", "L0")
    individual["modal_probability"] = confidence
    individual["normalized_binary_entropy"] = normalized_entropy
    return summary, pd.DataFrame(histogram_rows), individual


def _loan_outcomes_by_period(panel, groups):
    """Annual cleaned borrowing flows and outstanding stocks by panel period."""
    enrolled = panel.loc[panel["educ"].isin(EDUCATION)].copy().reset_index(drop=True)
    rows = []
    for _, label, individual_weights in groups:
        row_weights = _weights_for_rows(enrolled, individual_weights)
        for period, indices in enrolled.groupby("period", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            selected = enrolled.loc[index]
            weights = row_weights[index]
            flow = selected["auxiliary_loan_flow"].to_numpy(dtype=float)
            stock = selected["currentloans"].to_numpy(dtype=float)
            positive_flow = flow > 0.0
            positive_stock = stock > 0.0
            rows.append(
                {
                    "loan_type": label,
                    "period": int(period),
                    "sample": "Enrolled in current period",
                    "raw_enrolled_observations": int(len(index)),
                    "posterior_weighted_enrolled_observations": float(weights.sum()),
                    "mean_annual_loan_flow": _weighted_mean(flow, weights),
                    "share_receiving_annual_loan": _weighted_mean(positive_flow, weights),
                    "mean_annual_loan_flow_among_borrowers": _weighted_mean(
                        flow[positive_flow], weights[positive_flow]
                    ),
                    "median_annual_loan_flow": _weighted_quantile(flow, weights, 0.50),
                    "p80_annual_loan_flow": _weighted_quantile(flow, weights, 0.80),
                    "mean_current_loan_stock": _weighted_mean(stock, weights),
                    "share_with_positive_current_loan_stock": _weighted_mean(
                        positive_stock, weights
                    ),
                    "mean_current_loan_stock_among_indebted": _weighted_mean(
                        stock[positive_stock], weights[positive_stock]
                    ),
                    "median_current_loan_stock": _weighted_quantile(stock, weights, 0.50),
                }
            )
    return pd.DataFrame(rows)


def _debt_by_period(panel, groups):
    # Debt accumulation is meaningful for the requested comparison only while
    # the individual is enrolled.  Non-enrolled person-years are deliberately
    # excluded from both the type means and the period-specific reference mean.
    enrolled = panel.loc[panel["educ"].isin(EDUCATION)].copy().reset_index(drop=True)
    rows = []
    population_means = (
        enrolled.groupby("period", sort=True)["currentloans"].mean().to_dict()
    )
    for dimension, label, individual_weights in groups:
        row_weights = _weights_for_rows(enrolled, individual_weights)
        for period, indices in enrolled.groupby("period", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            debt = enrolled.loc[index, "currentloans"].to_numpy(dtype=float)
            weights = row_weights[index]
            positive = debt > 0
            rows.append(
                {
                    "grouping": dimension,
                    "type": label,
                    "period": int(period),
                    "sample": "Enrolled in current period",
                    "raw_enrolled_observations": int(len(index)),
                    "posterior_weighted_enrolled_observations": float(weights.sum()),
                    "mean_current_loans": _weighted_mean(debt, weights),
                    "difference_from_unweighted_enrolled_mean": _weighted_mean(debt, weights)
                    - population_means[int(period)],
                    "share_with_positive_loans": _weighted_mean(positive, weights),
                    "mean_loans_among_indebted": _weighted_mean(debt[positive], weights[positive]),
                    "median_current_loans": _weighted_quantile(debt, weights, 0.50),
                }
            )
    return pd.DataFrame(rows)


def _graduation_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the final enrolled row immediately preceding each graduation state.

    The model records graduation in the following period's state.  Therefore,
    the analysis row is t-1 (which must be enrolled at the relevant level),
    while the debt stock in the new graduation state at t is retained in a
    separate diagnostic column.
    """
    events = []
    for education_code, (education_name, graduation_variable) in EDUCATION.items():
        graduation = panel[graduation_variable].fillna(0).gt(0)
        lagged = graduation.groupby(panel["_em_row"], sort=False).shift(fill_value=False)
        graduation_positions = np.flatnonzero((graduation & ~lagged).to_numpy())
        final_enrollment_positions = graduation_positions - 1
        valid = (
            (final_enrollment_positions >= 0)
            & (
                panel.iloc[final_enrollment_positions]["_em_row"].to_numpy()
                == panel.iloc[graduation_positions]["_em_row"].to_numpy()
            )
            & (
                panel.iloc[final_enrollment_positions]["educ"].to_numpy()
                == education_code
            )
        )
        if not np.all(valid):
            invalid_ids = panel.iloc[graduation_positions[~valid]]["PUBID"].tolist()
            raise ValueError(
                f"Graduation-state transitions for education={education_code} do not "
                "immediately follow enrollment for PUBID values "
                f"{invalid_ids[:10]}. No observations were silently discarded."
            )
        selected = panel.iloc[final_enrollment_positions].copy()
        graduation_rows = panel.iloc[graduation_positions]
        selected["education_level"] = education_name
        selected["education_code"] = education_code
        selected["final_enrollment_period"] = selected["period"].to_numpy(dtype=int)
        selected["graduation_state_period"] = graduation_rows["period"].to_numpy(dtype=int)
        selected["graduation_state_currentloans"] = graduation_rows[
            "currentloans"
        ].to_numpy(dtype=float)
        events.append(selected)
    return pd.concat(events, ignore_index=True) if events else panel.iloc[0:0].copy()


def _debt_at_graduation(panel, groups):
    events = _graduation_events(panel)
    rows = []
    for dimension, label, individual_weights in groups:
        event_weights = _weights_for_rows(events, individual_weights)
        for education_code, (education_name, _) in EDUCATION.items():
            selected = events["education_code"].eq(education_code).to_numpy()
            debt = events.loc[selected, "currentloans"].to_numpy(dtype=float)
            next_state_debt = events.loc[
                selected, "graduation_state_currentloans"
            ].to_numpy(dtype=float)
            weights = event_weights[selected]
            positive = debt > 0
            rows.append(
                {
                    "grouping": dimension,
                    "type": label,
                    "graduation_level": education_name,
                    "raw_graduation_events": int(selected.sum()),
                    "posterior_weighted_events": float(weights.sum()),
                    "debt_timing": "Final enrolled period before graduation state",
                    "mean_loans_in_final_enrollment_year": _weighted_mean(debt, weights),
                    "share_indebted_in_final_enrollment_year": _weighted_mean(
                        positive, weights
                    ),
                    "mean_loans_among_indebted_in_final_enrollment_year": _weighted_mean(
                        debt[positive], weights[positive]
                    ),
                    "median_loans_in_final_enrollment_year": _weighted_quantile(
                        debt, weights, 0.50
                    ),
                    "mean_loans_in_next_graduation_state": _weighted_mean(
                        next_state_debt, weights
                    ),
                }
            )
    return pd.DataFrame(rows)


def _loan_outcomes_at_graduation(panel, groups):
    """Flow and stock in the last enrolled year before each graduation state."""
    events = _graduation_events(panel)
    rows = []
    for _, label, individual_weights in groups:
        event_weights = _weights_for_rows(events, individual_weights)
        graduation_groups = [(None, "All graduation levels")] + [
            (education_code, education_name)
            for education_code, (education_name, _) in EDUCATION.items()
        ]
        for education_code, education_name in graduation_groups:
            selected = (
                np.ones(len(events), dtype=bool)
                if education_code is None
                else events["education_code"].eq(education_code).to_numpy()
            )
            graduation = events.loc[selected]
            weights = event_weights[selected]
            flow = graduation["auxiliary_loan_flow"].to_numpy(dtype=float)
            stock = graduation["currentloans"].to_numpy(dtype=float)
            next_stock = graduation["graduation_state_currentloans"].to_numpy(dtype=float)
            positive_flow = flow > 0.0
            positive_stock = stock > 0.0
            rows.append(
                {
                    "loan_type": label,
                    "graduation_level": education_name,
                    "raw_graduation_events": int(selected.sum()),
                    "posterior_weighted_graduation_events": float(weights.sum()),
                    "debt_timing": "Final enrolled period before graduation state",
                    "mean_annual_loan_flow_in_final_enrollment_year": _weighted_mean(
                        flow, weights
                    ),
                    "share_receiving_loan_in_final_enrollment_year": _weighted_mean(
                        positive_flow, weights
                    ),
                    "mean_annual_flow_among_borrowers_in_final_enrollment_year": (
                        _weighted_mean(flow[positive_flow], weights[positive_flow])
                    ),
                    "mean_loan_stock_in_final_enrollment_year": _weighted_mean(
                        stock, weights
                    ),
                    "share_indebted_in_final_enrollment_year": _weighted_mean(
                        positive_stock, weights
                    ),
                    "mean_stock_among_indebted_in_final_enrollment_year": (
                        _weighted_mean(stock[positive_stock], weights[positive_stock])
                    ),
                    "median_loan_stock_in_final_enrollment_year": _weighted_quantile(
                        stock, weights, 0.50
                    ),
                    "mean_loan_stock_in_next_graduation_state": _weighted_mean(
                        next_stock, weights
                    ),
                }
            )
    return pd.DataFrame(rows)


def _longest_true_spell(values) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _individual_loan_histories(panel):
    """Construct repeated-borrowing outcomes for each individual."""
    enrolled = panel.loc[panel["educ"].isin(EDUCATION)].copy()
    rows = []
    for em_row, person in enrolled.groupby("_em_row", sort=True):
        person = person.sort_values("period")
        flow = person["auxiliary_loan_flow"].to_numpy(dtype=float)
        positive = flow > 0.0
        positive_flow = flow[positive]
        rows.append(
            {
                "_em_row": int(em_row),
                "enrolled_years": int(len(person)),
                "positive_borrowing_years": int(positive.sum()),
                "share_enrolled_years_borrowing": float(positive.mean()),
                "ever_borrowed": bool(positive.any()),
                "borrowed_in_multiple_years": bool(positive.sum() >= 2),
                "longest_positive_borrowing_spell": _longest_true_spell(positive),
                "first_borrowing_period": (
                    float(person.loc[positive, "period"].iloc[0])
                    if positive.any() else np.nan
                ),
                "cumulative_annual_loan_flow": float(np.nansum(flow)),
                "mean_positive_annual_loan_flow": (
                    float(np.mean(positive_flow)) if len(positive_flow) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _loan_persistence_by_type(panel, groups):
    """Person-level repetition and consecutive-period borrowing transitions."""
    people = _individual_loan_histories(panel)
    person_rows = []
    for _, label, individual_weights in groups:
        indices = people["_em_row"].to_numpy(dtype=int)
        weights = individual_weights[indices]
        ever = people["ever_borrowed"].to_numpy(dtype=bool)
        person_rows.append(
            {
                "loan_type": label,
                "raw_ever_enrolled_individuals": int(len(people)),
                "posterior_weighted_ever_enrolled_individuals": float(weights.sum()),
                "share_ever_borrowed": _weighted_mean(ever, weights),
                "mean_positive_borrowing_years": _weighted_mean(
                    people["positive_borrowing_years"], weights
                ),
                "mean_share_enrolled_years_borrowing": _weighted_mean(
                    people["share_enrolled_years_borrowing"], weights
                ),
                "share_borrowed_in_multiple_years": _weighted_mean(
                    people["borrowed_in_multiple_years"], weights
                ),
                "mean_longest_positive_borrowing_spell": _weighted_mean(
                    people["longest_positive_borrowing_spell"], weights
                ),
                "mean_first_borrowing_period_among_ever_borrowers": _weighted_mean(
                    people.loc[ever, "first_borrowing_period"], weights[ever]
                ),
                "mean_cumulative_annual_loan_flow": _weighted_mean(
                    people["cumulative_annual_loan_flow"], weights
                ),
                "mean_positive_flow_among_ever_borrowers": _weighted_mean(
                    people.loc[ever, "mean_positive_annual_loan_flow"], weights[ever]
                ),
            }
        )

    ordered = panel.sort_values(["_em_row", "period"], kind="stable").copy()
    grouped = ordered.groupby("_em_row", sort=False)
    ordered["previous_period"] = grouped["period"].shift()
    ordered["previous_educ"] = grouped["educ"].shift()
    ordered["previous_loan_flow"] = grouped["auxiliary_loan_flow"].shift()
    consecutive_enrollment = (
        ordered["educ"].isin(EDUCATION)
        & ordered["previous_educ"].isin(EDUCATION)
        & ordered["previous_period"].eq(ordered["period"] - 1)
    )
    transitions = ordered.loc[consecutive_enrollment].copy().reset_index(drop=True)
    transition_rows = []
    for _, label, individual_weights in groups:
        weights = _weights_for_rows(transitions, individual_weights)
        previous_flow = transitions["previous_loan_flow"].to_numpy(dtype=float)
        current_flow = transitions["auxiliary_loan_flow"].to_numpy(dtype=float)
        previous_borrow = previous_flow > 0.0
        current_borrow = current_flow > 0.0
        continuation = _weighted_mean(current_borrow[previous_borrow], weights[previous_borrow])
        entry = _weighted_mean(current_borrow[~previous_borrow], weights[~previous_borrow])
        both_positive = previous_borrow & current_borrow
        transition_rows.append(
            {
                "loan_type": label,
                "transition_sample": "Consecutive panel periods enrolled in both years",
                "raw_transitions": int(len(transitions)),
                "posterior_weighted_transitions": float(weights.sum()),
                "share_borrowing_in_previous_year": _weighted_mean(previous_borrow, weights),
                "continuation_probability_P_borrow_t_given_borrow_tminus1": continuation,
                "entry_probability_P_borrow_t_given_no_borrow_tminus1": entry,
                "cessation_probability_P_no_borrow_t_given_borrow_tminus1": (
                    1.0 - continuation if np.isfinite(continuation) else np.nan
                ),
                "continuation_minus_entry": (
                    continuation - entry
                    if np.isfinite(continuation) and np.isfinite(entry) else np.nan
                ),
                "annual_flow_correlation_including_zeros": _weighted_correlation(
                    previous_flow, current_flow, weights
                ),
                "positive_flow_correlation_conditional_borrowing_both_years": (
                    _weighted_correlation(
                        previous_flow[both_positive], current_flow[both_positive],
                        weights[both_positive]
                    )
                ),
                "mean_flow_t_after_borrowing_tminus1": _weighted_mean(
                    current_flow[previous_borrow], weights[previous_borrow]
                ),
                "mean_flow_t_after_no_borrowing_tminus1": _weighted_mean(
                    current_flow[~previous_borrow], weights[~previous_borrow]
                ),
            }
        )
    return pd.DataFrame(person_rows), pd.DataFrame(transition_rows)


def _individual_education_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Create one row per individual with explicit non-completion definitions."""
    final_period = int(panel["period"].max())
    rows = []
    for em_row, person in panel.groupby("_em_row", sort=True):
        result = {"_em_row": int(em_row)}
        for education_code, (education_name, graduation_variable) in EDUCATION.items():
            short = {1: "two_year", 2: "four_year", 3: "graduate"}[education_code]
            enrolled = person["educ"].eq(education_code)
            ever = bool(enrolled.any())
            completed = bool(person[graduation_variable].fillna(0).gt(0).any())
            last_enrollment = int(person.loc[enrolled, "period"].max()) if ever else 0
            exit_observed = bool(ever and last_enrollment < final_period)
            result[f"ever_{short}"] = ever
            result[f"completed_{short}"] = completed
            result[f"noncompletion_{short}"] = bool(ever and not completed)
            result[f"observed_dropout_{short}"] = bool(ever and not completed and exit_observed)
            result[f"right_censored_{short}"] = bool(ever and not completed and not exit_observed)
            result[f"last_enrollment_{short}"] = last_enrollment
        rows.append(result)
    return pd.DataFrame(rows).sort_values("_em_row").reset_index(drop=True)


def _education_outcomes(panel, groups):
    people = _individual_education_outcomes(panel)
    rows = []
    for dimension, label, individual_weights in groups:
        weights = individual_weights[people["_em_row"].to_numpy(dtype=int)]
        for education_code, (education_name, _) in EDUCATION.items():
            short = {1: "two_year", 2: "four_year", 3: "graduate"}[education_code]
            ever = people[f"ever_{short}"].to_numpy(dtype=bool)
            completed = people[f"completed_{short}"].to_numpy(dtype=bool)
            noncompletion = people[f"noncompletion_{short}"].to_numpy(dtype=bool)
            observed_dropout = people[f"observed_dropout_{short}"].to_numpy(dtype=bool)
            right_censored = people[f"right_censored_{short}"].to_numpy(dtype=bool)
            rows.append(
                {
                    "grouping": dimension,
                    "type": label,
                    "education_level": education_name,
                    "posterior_weighted_individuals": float(weights.sum()),
                    "ever_enrolled_share": _weighted_mean(ever, weights),
                    "completion_share_among_ever_enrolled": _weighted_mean(
                        completed[ever], weights[ever]
                    ),
                    "noncompletion_share_among_ever_enrolled": _weighted_mean(
                        noncompletion[ever], weights[ever]
                    ),
                    "observed_dropout_share_among_ever_enrolled": _weighted_mean(
                        observed_dropout[ever], weights[ever]
                    ),
                    "right_censored_share_among_ever_enrolled": _weighted_mean(
                        right_censored[ever], weights[ever]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _work_rows(selected, weights, dimension, label, education_level, period):
    work = selected["work"].to_numpy(dtype=int)
    return {
        "grouping": dimension,
        "type": label,
        "education_level": education_level,
        "period": period,
        "posterior_weighted_enrolled_person_years": float(weights.sum()),
        "share_no_work": _weighted_mean(work == 0, weights),
        "share_part_time": _weighted_mean(work == 1, weights),
        "share_full_time": _weighted_mean(work == 2, weights),
        "share_any_work": _weighted_mean(work > 0, weights),
    }


def _work_while_enrolled(panel, groups):
    enrolled = panel.loc[panel["educ"].isin(EDUCATION)].copy().reset_index(drop=True)
    rows_by_level = []
    rows_by_period = []
    for dimension, label, individual_weights in groups:
        weights = _weights_for_rows(enrolled, individual_weights)
        rows_by_level.append(
            _work_rows(enrolled, weights, dimension, label, "All enrolled", "All")
        )
        for education_code, (education_name, _) in EDUCATION.items():
            selected = enrolled["educ"].eq(education_code).to_numpy()
            rows_by_level.append(
                _work_rows(
                    enrolled.loc[selected],
                    weights[selected],
                    dimension,
                    label,
                    education_name,
                    "All",
                )
            )
        for period, indices in enrolled.groupby("period", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            rows_by_period.append(
                _work_rows(
                    enrolled.loc[index],
                    weights[index],
                    dimension,
                    label,
                    "All enrolled",
                    int(period),
                )
            )
    return pd.DataFrame(rows_by_level), pd.DataFrame(rows_by_period)


def _latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{value:.6g}"
    return _latex_escape(value)


def _write_latex(table: pd.DataFrame, path: Path):
    columns = list(table.columns)
    lines = [
        r"\begin{longtable}{" + "l" * len(columns) + "}",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        r"\hline",
        r"\endfirsthead",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        r"\hline",
        r"\endhead",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_value(value) for value in row) + " \\\\")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_tables(tables, output_root: Path, iteration: int):
    iteration_directory = output_root / "iterations" / f"iteration_{iteration:04d}"
    latest_directory = output_root / "latest"
    for directory in (iteration_directory, latest_directory):
        directory.mkdir(parents=True, exist_ok=True)
        for stem, original_table in tables.items():
            table = original_table.copy()
            table.insert(0, "iteration", int(iteration))
            table.to_csv(directory / f"{stem}.csv", index=False, float_format="%.10g")
            # The person-level posterior file is an analysis dataset, not a
            # presentational table; a 5,000+ row longtable is not useful.
            if stem != "loan_type_posterior_individual":
                _write_latex(table, directory / f"{stem}.tex")
    definitions = """Definitions used by tables_types.py

All statistics use posterior probabilities q_ik as fractional weights. No
individual is assigned to a single type.

Joint types are SxGyTzLw, where S is the schooling type, G is the grant type,
T is the parental-transfer type, and L is the loan type. Zero is the baseline
and one is the high type for each dimension. Marginal rows sum posterior
weights over the other three dimensions.

annual current loans: currentloans among individuals enrolled in two-year,
four-year, or graduate education in that period. Non-enrolled person-years are
not part of these annual comparisons.
annual loan flow: auxiliary_loan_flow, the cleaned current-period borrowing
flow used to identify the auxiliary EM loan type. This is distinct from
currentloans, the outstanding stock entering the period.
debt in the final enrollment year: currentloans in the enrolled period
immediately before the corresponding graduation indicator changes from zero to
one. The next period's debt stock is retained separately as
mean_loans_in_next_graduation_state so the model timing remains transparent.
non-completion: ever enrolled at that level but the corresponding graduation
indicator is never observed by the end of the ten-period panel.
observed dropout: non-completion with the last enrollment at that level before
the final panel period. Non-completers enrolled in the final period are reported
as right-censored rather than automatically classified as observed dropouts.
work while enrolled: work=0 (no work), work=1 (part-time), work=2 (full-time),
conditional on educ being two-year, four-year, or graduate enrollment.

loan-type posterior: posterior_L1 sums q over all eight joint types whose loan
component is one. Modal types use a 0.5 cutoff. Normalized binary entropy is
zero for a certain L0/L1 posterior and one for an exactly 50/50 posterior.

loan persistence: person-level repetition is computed over enrolled periods.
Transition statistics require enrollment in two consecutive panel periods.
Continuation is P(positive annual flow at t | positive annual flow at t-1),
entry is P(positive annual flow at t | zero flow at t-1), and their difference
is a descriptive state-dependence measure. It is not a causal effect of past
borrowing. The positive-flow correlation conditions on borrowing in both years.

These are descriptive posterior-weighted associations. The same outcomes help
identify the latent types, so differences are neither out-of-sample tests nor
causal effects of type membership.
"""
    for directory in (iteration_directory, latest_directory):
        (directory / "definitions.txt").write_text(definitions, encoding="utf-8")
    return iteration_directory, latest_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=EST("auxiliary_em_checkpoint.npz"),
        help="Sixteen-type auxiliary EM checkpoint/results npz containing q.",
    )
    parser.add_argument(
        "--data",
        default=RDATA("realdata_tographs_allfields.dta"),
        help="Balanced real-data person-period Stata file.",
    )
    parser.add_argument(
        "--ids",
        default=RDATA("invariant_state_superfeasible_t1.npy"),
        help="Array whose first column gives EM individual IDs in q row order.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT("tables", "type_descriptives"),
        help="Root output directory for CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Optional iteration label; inferred from likelihood history by default.",
    )
    arguments = parser.parse_args()

    q, names, school, grant, transfer, loan, inferred_iteration = _load_posteriors(
        Path(arguments.results)
    )
    iteration = arguments.iteration if arguments.iteration is not None else inferred_iteration
    panel = _load_analysis_panel(Path(arguments.data), Path(arguments.ids), q)
    groups = _group_weights(q, names, school, grant, transfer, loan)
    loan_groups = _loan_type_groups(q, loan)
    posterior_summary, posterior_histogram, posterior_individual = (
        _loan_type_posterior_tables(panel, q, loan)
    )
    persistence_people, persistence_transitions = _loan_persistence_by_type(
        panel, loan_groups
    )
    work_by_level, work_by_period = _work_while_enrolled(panel, groups)
    tables = {
        "loan_type_posterior_summary": posterior_summary,
        "loan_type_posterior_histogram": posterior_histogram,
        "loan_type_posterior_individual": posterior_individual,
        "loan_type_annual_flow_and_stock_by_year": _loan_outcomes_by_period(
            panel, loan_groups
        ),
        "loan_type_flow_and_stock_at_graduation": _loan_outcomes_at_graduation(
            panel, loan_groups
        ),
        "loan_type_persistence_person_level": persistence_people,
        "loan_type_persistence_transitions": persistence_transitions,
        "student_loans_by_year_and_type": _debt_by_period(panel, groups),
        "student_loans_at_graduation_by_type": _debt_at_graduation(panel, groups),
        "education_completion_dropout_by_type": _education_outcomes(panel, groups),
        "work_while_enrolled_by_type": work_by_level,
        "work_while_enrolled_by_year_and_type": work_by_period,
    }
    iteration_directory, latest_directory = _save_tables(
        tables, Path(arguments.output_dir), iteration
    )
    print(f"Type descriptive tables written to {iteration_directory}", flush=True)
    print(f"Latest type descriptive tables updated in {latest_directory}", flush=True)


if __name__ == "__main__":
    main()
