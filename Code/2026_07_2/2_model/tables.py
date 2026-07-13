# -*- coding: utf-8 -*-
"""Produce reporting tables for the four-type auxiliary EM estimator.

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


TYPE_NAMES = ("LL", "LH", "HL", "HH")
SCHOOL_TYPE = ("Low", "Low", "High", "High")
FINANCIAL_TYPE = ("Low", "High", "Low", "High")
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
            elif "financial_type" in name:
                dimension = "Financial resources"
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

    financial_names = FINANCIAL_X_NAMES + ("high_financial_type",)
    for source_name, parameters in (
        ("Grant", grant_parameters),
        ("Transfer", transfer_parameters),
    ):
        add(source_name, "Receipt logit", financial_names, parameters["receipt"])
        add(source_name, "Log positive amount", financial_names, parameters["amount"])
        add(source_name, "Log positive amount", ("sigma",), (parameters["sigma"],))

    add(
        "Type distribution",
        "Prior probability",
        tuple(f"prior_probability_{name}" for name in TYPE_NAMES),
        pi,
        "Joint type",
    )
    return pd.DataFrame(rows)


def _prior_table(iteration, pi):
    pi = np.asarray(pi, dtype=float).reshape(-1)
    if len(pi) != 4:
        raise ValueError("The four-type table requires four prior probabilities.")
    return pd.DataFrame(
        {
            "iteration": int(iteration),
            "type": TYPE_NAMES,
            "schooling_type": SCHOOL_TYPE,
            "financial_resources_type": FINANCIAL_TYPE,
            "prior_probability": pi,
            "prior_percent": 100.0 * pi,
        }
    )


def _probability_rows(iteration, outcome, x, parameters, varying_dimension):
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
    for school, financial in zip(SCHOOL_TYPE, FINANCIAL_TYPE):
        is_high = financial == "High" if varying_dimension == "Financial resources" else school == "High"
        probabilities.append(high if is_high else low)
    reference = probabilities[0]
    return [
        {
            "iteration": int(iteration),
            "outcome": outcome,
            "type": type_name,
            "schooling_type": school,
            "financial_resources_type": financial,
            "varying_type_dimension": varying_dimension,
            "average_predicted_probability": probability,
            "marginal_effect_vs_LL": probability - reference,
            "marginal_effect_percentage_points": 100.0 * (probability - reference),
        }
        for type_name, school, financial, probability in zip(
            TYPE_NAMES, SCHOOL_TYPE, FINANCIAL_TYPE, probabilities
        )
    ]


def _marginal_effect_table(
    iteration,
    measure_late,
    measure_summer,
    grant_parameters,
    transfer_parameters,
    auxiliary_data,
):
    rows = []
    rows.extend(
        _probability_rows(
            iteration,
            "Receive grant",
            auxiliary_data["grant"]["x"],
            grant_parameters["receipt"],
            "Financial resources",
        )
    )
    rows.extend(
        _probability_rows(
            iteration,
            "Receive transfer",
            auxiliary_data["transfer"]["x"],
            transfer_parameters["receipt"],
            "Financial resources",
        )
    )
    rows.extend(
        _probability_rows(
            iteration,
            "Late school",
            auxiliary_data["x1_measure"],
            measure_late,
            "Schooling",
        )
    )
    rows.extend(
        _probability_rows(
            iteration,
            "Summer class",
            auxiliary_data["x1_measure"],
            measure_summer,
            "Schooling",
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

    tables = {
        "estimated_parameters": _parameter_rows(
            iteration,
            pi,
            choice_parameters,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
        ),
        "type_prior_probabilities": _prior_table(iteration, pi),
        "type_marginal_effects": _marginal_effect_table(
            iteration,
            measure_late,
            measure_summer,
            grant_parameters,
            transfer_parameters,
            auxiliary_data,
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
    paths = write_auxiliary_em_tables(
        iteration=iteration,
        pi=checkpoint["pi"],
        choice_parameters=checkpoint["choice_parameters"],
        measure_late=checkpoint["measure_late"],
        measure_summer=checkpoint["measure_summer"],
        grant_parameters={
            "receipt": checkpoint["grant_receipt"],
            "amount": checkpoint["grant_amount"],
            "sigma": float(checkpoint["grant_sigma"]),
        },
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
