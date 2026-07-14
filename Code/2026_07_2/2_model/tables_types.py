# -*- coding: utf-8 -*-
"""Posterior-weighted descriptive statistics for auxiliary EM latent types.

Individuals are never assigned to a single type.  For type k, observation i
receives weight q[i, k], where q is the posterior probability produced by the
eight-type auxiliary EM estimator.  The resulting tables are descriptive and
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


EXPECTED_TYPE_NAMES = (
    "S0G0T0",
    "S0G0T1",
    "S0G1T0",
    "S0G1T1",
    "S1G0T0",
    "S1G0T1",
    "S1G1T0",
    "S1G1T1",
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
        required = {"q", "type_names", "type_school", "type_grant", "type_transfer"}
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
        history_length = len(results["observed_loglike"]) if "observed_loglike" in results else 1

    if q.ndim != 2:
        raise ValueError(f"Posterior q must be a matrix; received shape {q.shape}.")
    if q.shape[1] != len(names):
        raise ValueError("The number of posterior columns does not match type_names.")
    if not (len(names) == len(school) == len(grant) == len(transfer)):
        raise ValueError("Type-name and type-component arrays have inconsistent lengths.")
    if len(names) != 8 or set(names) != set(EXPECTED_TYPE_NAMES):
        raise ValueError(
            "This script requires the eight-type schooling x grant x parental-transfer "
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
    return q, names, school, grant, transfer, max(1, history_length - 1)


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
    panel["currentloans"] = pd.to_numeric(panel["currentloans"], errors="coerce")
    return panel


def _group_weights(q, names, school, grant, transfer):
    """Return all joint types and each two-category marginal type dimension."""
    groups = []
    for column, name in enumerate(names):
        groups.append(("Joint type", name, q[:, column]))
    for dimension, components, prefix in (
        ("Schooling type", school, "S"),
        ("Grant type", grant, "G"),
        ("Parental-transfer type", transfer, "T"),
    ):
        for value in (0, 1):
            groups.append((dimension, f"{prefix}{value}", q[:, components == value].sum(axis=1)))
    return groups


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
            _write_latex(table, directory / f"{stem}.tex")
    definitions = """Definitions used by tables_types.py

All statistics use posterior probabilities q_ik as fractional weights. No
individual is assigned to a single type.

annual current loans: currentloans among individuals enrolled in two-year,
four-year, or graduate education in that period. Non-enrolled person-years are
not part of these annual comparisons.
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
        help="Eight-type auxiliary EM checkpoint/results npz containing q.",
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

    q, names, school, grant, transfer, inferred_iteration = _load_posteriors(
        Path(arguments.results)
    )
    iteration = arguments.iteration if arguments.iteration is not None else inferred_iteration
    panel = _load_analysis_panel(Path(arguments.data), Path(arguments.ids), q)
    groups = _group_weights(q, names, school, grant, transfer)
    work_by_level, work_by_period = _work_while_enrolled(panel, groups)
    tables = {
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
