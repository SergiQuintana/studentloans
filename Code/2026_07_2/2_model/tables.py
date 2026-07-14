# -*- coding: utf-8 -*-
"""Produce reporting tables for the auxiliary latent-type EM estimator.

The EM routine calls :func:`write_auxiliary_em_tables` after every iteration.
This file can also be run after estimation to recreate the tables from the
latest auxiliary checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from config import EST, OUT


TYPE_NAMES = (
    "S0G0T0",
    "S0G0T1",
    "S0G1T0",
    "S0G1T1",
    "S1G0T0",
    "S1G0T1",
    "S1G1T0",
    "S1G1T1",
)
SCHOOL_TYPE = ("0 (baseline)",) * 4 + ("1 (high)",) * 4
GRANT_TYPE = (
    "0 (baseline)",
    "0 (baseline)",
    "1 (high)",
    "1 (high)",
) * 2
TRANSFER_TYPE = ("0 (baseline)", "1 (high)") * 4
X1_NAMES = (
    "constant",
    "parent_income_q2",
    "parent_income_q3",
    "parent_income_q4",
    "ability_q2",
    "ability_q3",
    "ability_q4",
    "female",
    "black",
)
FINANCIAL_X_NAMES = X1_NAMES + (
    "four_year",
    "graduate",
    "part_time",
    "full_time",
)
GRANT_X_NAMES = X1_NAMES + ("part_time", "full_time")
GRANT_EDUCATION_NAMES = {1: "Two-year", 2: "Four-year", 3: "Graduate"}


def _type_layout(number_types: int):
    """Return labels/dimensions, including support for legacy checkpoints."""
    if number_types == 8:
        return {
            "names": TYPE_NAMES,
            "school": SCHOOL_TYPE,
            "grant": GRANT_TYPE,
            "transfer": TRANSFER_TYPE,
        }
    if number_types == 4:
        # In the old model the same resource type entered both equations.
        resource = ("0 (baseline)", "1 (high)", "0 (baseline)", "1 (high)")
        return {
            "names": ("LL", "LH", "HL", "HH"),
            "school": ("0 (baseline)", "0 (baseline)", "1 (high)", "1 (high)"),
            "grant": resource,
            "transfer": resource,
        }
    raise ValueError(f"Expected 8 new types (or 4 legacy types), received {number_types}.")


def _choice_parameter_names(number_parameters: int) -> list[str]:
    """Return stable block labels in the exact legacy parameter ordering."""
    block_sizes = (
        ("individual_characteristic", 162),
        ("work_choice", 28),
        ("previous_choice", 17),
        ("education_occupation_complementarity", 47),
        ("period", 140),
        ("period_work", 32),
        ("first_enrollment", 6),
        ("experience_ability", 168),
    )
    names = []
    for block, size in block_sizes:
        names.extend(f"{block}_{index + 1:03d}" for index in range(size))
    names.extend(
        (
            "high_schooling_type_associate",
            "high_schooling_type_four_year",
            "expected_consumption",
            "current_debt_nonhome",
        )
    )
    if len(names) != number_parameters:
        names = [f"choice_parameter_{index + 1:03d}" for index in range(number_parameters)]
        if number_parameters >= 4:
            names[-4:] = [
                "high_schooling_type_associate",
                "high_schooling_type_four_year",
                "expected_consumption",
                "current_debt_nonhome",
            ]
    return names


def _parameter_rows(
    iteration,
    pi,
    choice_parameters,
    measure_late,
    measure_summer,
    grant_parameters,
    transfer_parameters,
    type_layout,
):
    rows = []

    def add(component, equation, names, values, type_dimension="None"):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(names) != len(values):
            names = [f"parameter_{index + 1:03d}" for index in range(len(values))]
        for name, value in zip(names, values):
            dimension = type_dimension
            if "schooling_type" in name:
                dimension = "Schooling"
            elif "grant_type" in name:
                dimension = "Grant"
            elif "parental_transfer_type" in name:
                dimension = "Parental transfer"
            rows.append(
                {
                    "iteration": int(iteration),
                    "component": component,
                    "equation": equation,
                    "parameter": name,
                    "estimate": float(value),
                    "type_dimension": dimension,
                }
            )

    add(
        "Education choice",
        "Multinomial choice utility",
        _choice_parameter_names(len(choice_parameters)),
        choice_parameters,
    )
    measure_names = X1_NAMES + ("high_schooling_type",)
    add("Schooling measure", "Late school", measure_names, measure_late)
    add("Schooling measure", "Summer class", measure_names, measure_summer)

    grant_names = GRANT_X_NAMES + ("high_grant_type",)
    if "receipt" in grant_parameters:  # legacy pooled checkpoint
        add("Grant", "Receipt logit", FINANCIAL_X_NAMES + ("high_grant_type",), grant_parameters["receipt"])
        add("Grant", "Log positive amount", FINANCIAL_X_NAMES + ("high_grant_type",), grant_parameters["amount"])
        add("Grant", "Log positive amount", ("sigma",), (grant_parameters["sigma"],))
    else:
        for education in (1, 2, 3):
            parameters = grant_parameters[education]
            component = f"Grant: {GRANT_EDUCATION_NAMES[education]}"
            add(component, "Receipt logit", grant_names, parameters["receipt"])
            add(component, "Log positive amount", grant_names, parameters["amount"])
            add(component, "Log positive amount", ("sigma",), (parameters["sigma"],))

    transfer_names = FINANCIAL_X_NAMES + ("high_parental_transfer_type",)
    add("Transfer", "Receipt logit", transfer_names, transfer_parameters["receipt"])
    add("Transfer", "Log positive amount", transfer_names, transfer_parameters["amount"])
    add("Transfer", "Log positive amount", ("sigma",), (transfer_parameters["sigma"],))

    add(
        "Type distribution",
        "Prior probability",
        tuple(f"prior_probability_{name}" for name in type_layout["names"]),
        pi,
        "Joint type",
    )
    return pd.DataFrame(rows)


def _prior_table(iteration, pi, type_layout):
    pi = np.asarray(pi, dtype=float).reshape(-1)
    return pd.DataFrame(
        {
            "iteration": int(iteration),
            "type": type_layout["names"],
            "schooling_type": type_layout["school"],
            "grant_type": type_layout["grant"],
            "parental_transfer_type": type_layout["transfer"],
            "prior_probability": pi,
            "prior_percent": 100.0 * pi,
        }
    )


def _probability_rows(iteration, outcome, x, parameters, varying_dimension, type_layout):
    x = np.asarray(x, dtype=float)
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if x.shape[1] + 1 != len(parameters):
        raise ValueError(
            f"{outcome}: design has {x.shape[1]} columns but parameter vector "
            f"has {len(parameters)} entries."
        )
    low = float(np.mean(expit(x @ parameters[:-1])))
    high = float(np.mean(expit(x @ parameters[:-1] + parameters[-1])))
    probabilities = []
    dimension_values = {
        "Schooling": type_layout["school"],
        "Grant": type_layout["grant"],
        "Parental transfer": type_layout["transfer"],
    }[varying_dimension]
    for value in dimension_values:
        is_high = value == "1 (high)"
        probabilities.append(high if is_high else low)
    reference = probabilities[0]
    return [
        {
            "iteration": int(iteration),
            "outcome": outcome,
            "type": type_name,
            "schooling_type": school,
            "grant_type": grant,
            "parental_transfer_type": transfer,
            "varying_type_dimension": varying_dimension,
            "average_predicted_probability": probability,
            "marginal_effect_vs_all_baseline": probability - reference,
            "marginal_effect_percentage_points": 100.0 * (probability - reference),
        }
        for type_name, school, grant, transfer, probability in zip(
            type_layout["names"],
            type_layout["school"],
            type_layout["grant"],
            type_layout["transfer"],
            probabilities,
        )
    ]


def _marginal_effect_table(
    iteration,
    measure_late,
    measure_summer,
    grant_parameters,
    transfer_parameters,
    auxiliary_data,
    type_layout,
):
    rows = []
    if "receipt" in grant_parameters:  # legacy pooled checkpoint
        if "x" in auxiliary_data["grant"]:
            legacy_grant_x = auxiliary_data["grant"]["x"]
        else:
            legacy_parts = []
            for education in (1, 2, 3):
                grant_x = auxiliary_data["grant"][education]["x"]
                legacy_parts.append(
                    np.column_stack(
                        (
                            grant_x[:, :9],
                            np.full(len(grant_x), education == 2),
                            np.full(len(grant_x), education == 3),
                            grant_x[:, 9:],
                        )
                    ).astype(float)
                )
            legacy_grant_x = np.concatenate(legacy_parts, axis=0)
        rows.extend(
            _probability_rows(
                iteration,
                "Receive grant",
                legacy_grant_x,
                grant_parameters["receipt"],
                "Grant",
                type_layout,
            )
        )
    else:
        for education in (1, 2, 3):
            rows.extend(
                _probability_rows(
                    iteration,
                    f"Receive grant: {GRANT_EDUCATION_NAMES[education]}",
                    auxiliary_data["grant"][education]["x"],
                    grant_parameters[education]["receipt"],
                    "Grant",
                    type_layout,
                )
            )
    rows.extend(
        _probability_rows(
            iteration,
            "Receive transfer",
            auxiliary_data["transfer"]["x"],
            transfer_parameters["receipt"],
            "Parental transfer",
            type_layout,
        )
    )
    rows.extend(
        _probability_rows(
            iteration,
            "Late school",
            auxiliary_data["x1_measure"],
            measure_late,
            "Schooling",
            type_layout,
        )
    )
    rows.extend(
        _probability_rows(
            iteration,
            "Summer class",
            auxiliary_data["x1_measure"],
            measure_summer,
            "Schooling",
            type_layout,
        )
    )
    return pd.DataFrame(rows)


def _save_table(table, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    latex_path = directory / f"{stem}.tex"
    table.to_csv(csv_path, index=False, float_format="%.10g")
    _write_latex_table(table, latex_path)
    return csv_path, latex_path


def _latex_escape(value):
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


def _latex_value(value):
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{value:.6g}"
    return _latex_escape(value)


def _write_latex_table(table, path):
    """Write dependency-free LaTeX (pandas.to_latex now requires Jinja2)."""
    columns = list(table.columns)
    alignment = "l" * len(columns)
    lines = [
        r"\begin{longtable}{" + alignment + "}",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        r"\hline",
        r"\endfirsthead",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        r"\hline",
        r"\endhead",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_value(value) for value in row) + r" \\")
    lines.append(r"\end{longtable}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_auxiliary_em_tables(
    *,
    iteration,
    pi,
    choice_parameters,
    measure_late,
    measure_summer,
    grant_parameters,
    transfer_parameters,
    auxiliary_data,
    output_root=None,
):
    """Write iteration-specific and latest CSV/LaTeX auxiliary EM tables."""
    root = Path(output_root or OUT("tables", "auxiliary_em"))
    iteration_directory = root / "iterations" / f"iteration_{int(iteration):04d}"
    latest_directory = root / "latest"
    pi = np.asarray(pi, dtype=float).reshape(-1)
    type_layout = _type_layout(len(pi))

    tables = {
        "estimated_parameters": _parameter_rows(
            iteration,
            pi,
            choice_parameters,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            type_layout,
        ),
        "type_prior_probabilities": _prior_table(iteration, pi, type_layout),
        "type_marginal_effects": _marginal_effect_table(
            iteration,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            auxiliary_data,
            type_layout,
        ),
    }
    for stem, table in tables.items():
        _save_table(table, iteration_directory, stem)
        _save_table(table, latest_directory, stem)
    return {
        "iteration_dir": str(iteration_directory),
        "latest_dir": str(latest_directory),
    }


def _load_auxiliary_data():
    # Import here so importing this reporting module never starts or loads the
    # estimator. model_em_algorithm itself is protected by a __main__ guard.
    import model_em_algorithm as em

    (
        choices_all,
        _vjt_low,
        _vjt_high,
        x1_new,
        choices_array_all,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    ) = em.load_all_arrays_feasible(auxiliar=1)
    return em.build_auxiliary_em_data(
        choices_all,
        choices_array_all,
        x1_new,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=EST("auxiliary_em_checkpoint.npz"),
        help="Path to an auxiliary_em_checkpoint.npz file.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration label; defaults to the checkpoint likelihood-history length minus one.",
    )
    arguments = parser.parse_args()

    checkpoint = np.load(arguments.checkpoint)
    iteration = arguments.iteration
    if iteration is None:
        iteration = max(1, len(checkpoint["observed_loglike"]) - 1)
    auxiliary_data = _load_auxiliary_data()
    grant_receipt = checkpoint["grant_receipt"]
    if grant_receipt.ndim == 1:
        grant_parameters = {
            "receipt": grant_receipt,
            "amount": checkpoint["grant_amount"],
            "sigma": float(checkpoint["grant_sigma"]),
        }
    else:
        grant_parameters = {
            education: {
                "receipt": grant_receipt[index],
                "amount": checkpoint["grant_amount"][index],
                "sigma": float(checkpoint["grant_sigma"][index]),
            }
            for index, education in enumerate((1, 2, 3))
        }
    paths = write_auxiliary_em_tables(
        iteration=iteration,
        pi=checkpoint["pi"],
        choice_parameters=checkpoint["choice_parameters"],
        measure_late=checkpoint["measure_late"],
        measure_summer=checkpoint["measure_summer"],
        grant_parameters=grant_parameters,
        transfer_parameters={
            "receipt": checkpoint["transfer_receipt"],
            "amount": checkpoint["transfer_amount"],
            "sigma": float(checkpoint["transfer_sigma"]),
        },
        auxiliary_data=auxiliary_data,
    )
    print(f"Auxiliary EM tables written to {paths['iteration_dir']}", flush=True)
    print(f"Latest tables updated in {paths['latest_dir']}", flush=True)


if __name__ == "__main__":
    main()
