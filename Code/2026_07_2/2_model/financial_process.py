# -*- coding: utf-8 -*-
"""Shared education-specific grant process for solution and simulation.

Structural alternatives identify education level and work status, but not
current college type. Each grant equation therefore uses the nine invariant
controls plus part-time and full-time indicators.
"""

from pathlib import Path

import numpy as np


EDUCATION_LEVELS = (1, 2, 3)
EDUCATION_NAMES = {1: "two_year", 2: "four_year", 3: "graduate"}
N_INVARIANT = 9
N_DESIGN = 11


def _coerce_structural_design(parameters, education):
    """Convert legacy vectors to [x1(9), part-time, full-time]."""
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if len(parameters) == N_DESIGN:
        return parameters
    if len(parameters) == 13 and education in (1, 2):
        # Legacy files contained two current-college-type dummies, a variable
        # absent from the structural state and alternative definitions.
        return np.concatenate((parameters[:N_INVARIANT], parameters[-2:]))
    if len(parameters) == N_INVARIANT and education == 3:
        # The legacy graduate equation omitted work controls.
        return np.concatenate((parameters, np.zeros(2)))
    raise ValueError(
        f"Grant equation for education={education} has {len(parameters)} "
        f"coefficients; expected {N_DESIGN}. Rerun data_model_fullfields.do."
    )


def load_education_grant_process(function_coefficients_path):
    """Load receipt, positive log-amount, and residual-sigma equations."""
    root = Path(function_coefficients_path)
    receipt_files = (
        "dummy_grants_educ1.npy",
        "dummy_grants_educ2.npy",
        "dummy_grants_grad.npy",
    )
    amount_files = (
        "paramgrants_educ1.npy",
        "paramgrants_educ2.npy",
        "paramgrants_grad.npy",
    )
    sigma_files = (
        "sigma_grants_educ1.npy",
        "sigma_grants_educ2.npy",
        "sigma_grants_grad.npy",
    )
    receipt = np.vstack(
        [
            _coerce_structural_design(np.load(root / filename), level)
            for level, filename in zip(EDUCATION_LEVELS, receipt_files)
        ]
    )
    amount = np.vstack(
        [
            _coerce_structural_design(np.load(root / filename), level)
            for level, filename in zip(EDUCATION_LEVELS, amount_files)
        ]
    )
    sigma = np.zeros(3, dtype=float)
    for index, filename in enumerate(sigma_files):
        sigma_path = root / filename
        if sigma_path.exists():
            sigma[index] = float(np.asarray(np.load(sigma_path)).reshape(-1)[0])
    return {"receipt": receipt, "amount": amount, "sigma": sigma}


def expected_grant_scalar(x1, education, work, process):
    """Expected grant for one state/alternative, optimized for solver calls."""
    education = int(education)
    if education not in EDUCATION_LEVELS:
        return 0.0
    index = education - 1
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    if len(x1) != N_INVARIANT:
        raise ValueError(f"Scalar grant x1 has {len(x1)} entries; expected {N_INVARIANT}.")
    part_time = float(int(work) == 1)
    full_time = float(int(work) == 2)
    receipt = process["receipt"][index]
    amount = process["amount"][index]
    eta_receipt = (
        x1 @ receipt[:N_INVARIANT]
        + part_time * receipt[9]
        + full_time * receipt[10]
    )
    eta_amount = (
        x1 @ amount[:N_INVARIANT]
        + part_time * amount[9]
        + full_time * amount[10]
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
    conditional_mean = np.exp(
        np.clip(eta_amount + 0.5 * process["sigma"][index] ** 2, -20.0, 20.0)
    )
    return float(probability * conditional_mean)


def expected_grants_vectorized(x1, education, work, process):
    """Expected grants for arrays of agents without an agent-level loop."""
    x1 = np.asarray(x1, dtype=float)
    education = np.asarray(education, dtype=int).reshape(-1)
    work = np.asarray(work, dtype=int).reshape(-1)
    if x1.ndim != 2 or x1.shape[0] != len(education) or len(work) != len(education):
        raise ValueError("Grant inputs must have one x1, education, and work row per agent.")

    expected = np.zeros(len(education), dtype=float)
    for level in EDUCATION_LEVELS:
        selected = education == level
        if not np.any(selected):
            continue
        index = level - 1
        design = x1[selected]
        receipt = process["receipt"][index]
        amount = process["amount"][index]
        part_time = (work[selected] == 1).astype(float)
        full_time = (work[selected] == 2).astype(float)
        eta_receipt = (
            design @ receipt[:N_INVARIANT]
            + part_time * receipt[9]
            + full_time * receipt[10]
        )
        eta_amount = (
            design @ amount[:N_INVARIANT]
            + part_time * amount[9]
            + full_time * amount[10]
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
        conditional_mean = np.exp(
            np.clip(eta_amount + 0.5 * process["sigma"][index] ** 2, -20.0, 20.0)
        )
        expected[selected] = probability * conditional_mean
    return expected
