"""Single source of truth for the student-loan budget-shock specification.

The loan estimator, backward solution, and forward simulation must use the
functions in this module. Change the economic specification here, rather than
copying its parameter mappings into those three programs.

Invariant-state columns used here are:
    x1[..., 0] = parental-income quartile (1,...,4)
    x1[..., 1] = ability quartile (1,...,4)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from config import EST, ENSURE_DIR


SCHEMA_VERSION = 2
N_MEAN_PARAMETERS = 7
N_RISK_PARAMETERS = 4
N_DEBT_PENALTY_PARAMETERS = 4
DEBT_PENALTY_PARAMETERIZATION = "baseline_plus_deviations"


def unpack_estimation_vector(vector: np.ndarray, periods) -> dict[str, Any]:
    """Convert the SMM vector into the canonical named specification."""
    periods = np.asarray(periods, dtype=np.int64)
    vector = np.asarray(vector, dtype=np.float64)
    p = periods.size
    expected = N_MEAN_PARAMETERS * p + p + N_RISK_PARAMETERS + N_DEBT_PENALTY_PARAMETERS
    if vector.size != expected:
        raise ValueError(f"Budget-shock vector has {vector.size} entries; expected {expected}")

    sigma_start = N_MEAN_PARAMETERS * p
    risk_start = sigma_start + p
    penalty_start = risk_start + N_RISK_PARAMETERS
    return {
        "schema_version": SCHEMA_VERSION,
        "periods": periods,
        "mu_blocks": vector[:sigma_start].reshape(p, N_MEAN_PARAMETERS),
        "sigma_e": vector[sigma_start:risk_start],
        "risk_aversion": vector[risk_start:penalty_start],
        "debt_pen_parinc": vector[penalty_start:],
        "debt_pen_parameterization": DEBT_PENALTY_PARAMETERIZATION,
    }


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a loaded or newly constructed specification."""
    required = ("periods", "mu_blocks", "sigma_e", "debt_pen_parinc")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Budget-shock specification is missing: {missing}")

    out = dict(spec)
    out["periods"] = np.asarray(out["periods"], dtype=np.int64)
    out["mu_blocks"] = np.asarray(out["mu_blocks"], dtype=np.float64)
    out["sigma_e"] = np.asarray(out["sigma_e"], dtype=np.float64)
    out["debt_pen_parinc"] = np.asarray(out["debt_pen_parinc"], dtype=np.float64)
    if "risk_aversion" in out:
        out["risk_aversion"] = np.asarray(out["risk_aversion"], dtype=np.float64)

    p = out["periods"].size
    if out["mu_blocks"].shape != (p, N_MEAN_PARAMETERS):
        raise ValueError("mu_blocks must have shape (number of periods, 7)")
    if out["sigma_e"].shape != (p,) or np.any(out["sigma_e"] <= 0):
        raise ValueError("sigma_e must contain one positive value per period")
    if out["debt_pen_parinc"].shape not in ((3,), (N_DEBT_PENALTY_PARAMETERS,)):
        raise ValueError("debt_pen_parinc must have three legacy or four current entries")
    if "risk_aversion" in out and out["risk_aversion"].shape != (N_RISK_PARAMETERS,):
        raise ValueError("risk_aversion must have four entries")
    return out


def save(spec: dict[str, Any], raw_vector: np.ndarray | None = None,
         filename_prefix: str = "budgetshock") -> None:
    """Save the canonical bundle plus legacy files used by existing scripts."""
    spec = validate(spec)
    if "risk_aversion" not in spec:
        raise ValueError("Cannot save a fitted specification without risk_aversion")
    ENSURE_DIR(os.path.dirname(EST("budgetshock_params.npy")))

    # Keep risk_aversion both in the bundle and in the legacy separate file.
    np.save(EST("risk_aversion.npy"), spec["risk_aversion"])
    np.save(EST("budgetshock_params.npy"), spec, allow_pickle=True)
    if raw_vector is not None:
        np.save(EST(f"{filename_prefix}_bestx.npy"), np.asarray(raw_vector, dtype=np.float64))


def load(raise_if_missing: bool = True) -> dict[str, Any] | None:
    """Load estimates, accepting legacy bundles with separate risk aversion."""
    parameter_path = EST("budgetshock_params.npy")
    if not os.path.exists(parameter_path):
        if raise_if_missing:
            raise FileNotFoundError(f"Missing {parameter_path}")
        return None

    spec = np.load(parameter_path, allow_pickle=True).item()
    if "risk_aversion" not in spec:
        risk_path = EST("risk_aversion.npy")
        if os.path.exists(risk_path):
            spec["risk_aversion"] = np.load(risk_path)
    return validate(spec)


def _period_index(spec: dict[str, Any], period: int) -> int:
    match = np.flatnonzero(spec["periods"] == int(period))
    if match.size != 1:
        raise ValueError(f"Period {period} is not uniquely represented in budget-shock parameters")
    return int(match[0])


def conditional_mean_from_block(x1: np.ndarray, mu_block: np.ndarray) -> np.ndarray:
    """E[z|x1] for either one invariant state or an array of states."""
    x1 = np.asarray(x1)
    block = np.asarray(mu_block, dtype=np.float64)
    par = x1[..., 0].astype(np.int64)
    ability = x1[..., 1].astype(np.int64)
    mean = np.full(par.shape, block[0], dtype=np.float64)
    for level in range(2, 5):
        mean += block[level - 1] * (par == level)
        mean += block[level + 2] * (ability == level)
    return mean


def conditional_mean(spec: dict[str, Any], x1: np.ndarray, period: int) -> np.ndarray:
    return conditional_mean_from_block(x1, spec["mu_blocks"][_period_index(spec, period)])


def conditional_sigma(spec: dict[str, Any], period: int) -> float:
    return float(spec["sigma_e"][_period_index(spec, period)])


def risk_aversion(spec: dict[str, Any], x1: np.ndarray) -> np.ndarray:
    if "risk_aversion" not in spec:
        raise ValueError("Budget-shock specification has no risk_aversion parameters")
    par = np.asarray(x1)[..., 0].astype(np.int64)
    return spec["risk_aversion"][par - 1]


def debt_penalty_from_coefficients(x1: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Baseline plus parental-income deviations used in the SMM objective."""
    par = np.asarray(x1)[..., 0].astype(np.int64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    penalty = np.full(par.shape, coefficients[0], dtype=np.float64)
    for level in range(2, 5):
        penalty += coefficients[level - 1] * (par == level)
    return penalty


def debt_penalty(spec: dict[str, Any], x1: np.ndarray) -> np.ndarray:
    coefficients = spec["debt_pen_parinc"]
    parameterization = spec.get("debt_pen_parameterization", "group_levels")
    if parameterization == DEBT_PENALTY_PARAMETERIZATION:
        return debt_penalty_from_coefficients(x1, coefficients)
    par = np.asarray(x1)[..., 0].astype(np.int64)
    if coefficients.size == 3:
        # Old three-entry files stored the penalties for groups 2, 3, and 4;
        # parental-income group 1 had a zero penalty.
        penalty = np.zeros(par.shape, dtype=np.float64)
        for level in range(2, 5):
            penalty += coefficients[level - 2] * (par == level)
        return penalty
    # Backward compatibility: old length-four arrays represented group levels.
    return coefficients[par - 1]


def debt_penalty_design_vector(spec: dict[str, Any]) -> np.ndarray:
    """Coefficients for x1_new=[1,par2,par3,par4,...] in model_solution_em."""
    group_penalties = debt_penalty(spec, np.column_stack((np.arange(1, 5), np.ones(4))))
    vector = np.zeros(9, dtype=np.float64)
    vector[0] = group_penalties[0]
    vector[1:4] = group_penalties[1:] - group_penalties[0]
    return vector


def realization(spec: dict[str, Any], x1: np.ndarray, period: int,
                standard_draw: np.ndarray) -> np.ndarray:
    """Map fixed standard draws into budget shocks.

    The SMM estimator and forward simulation both go through this transform.
    For experiments with a different distribution, change this function and
    ``quadrature`` together; no consumer module needs its own distribution
    formula.
    """
    return (conditional_mean(spec, x1, period)
            + conditional_sigma(spec, period) * np.asarray(standard_draw, dtype=np.float64))


def draw(spec: dict[str, Any], x1: np.ndarray, period: int,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw the realized additive budget shock used in forward simulation."""
    rng = np.random.default_rng() if rng is None else rng
    shape = np.shape(conditional_mean(spec, x1, period))
    return realization(spec, x1, period, rng.standard_normal(size=shape))


def quadrature(spec: dict[str, Any], x1: np.ndarray, period: int, degree: int = 5):
    """Gauss-Hermite nodes and weights for the backward expectation."""
    x1 = np.asarray(x1)
    if x1.ndim > 1 and x1.shape[0] != 1:
        raise ValueError("quadrature expects one invariant state")
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    mean = float(np.asarray(conditional_mean(spec, x1, period)).reshape(-1)[0])
    values = np.sqrt(2.0) * conditional_sigma(spec, period) * nodes + mean
    return values, weights / np.sqrt(np.pi)
