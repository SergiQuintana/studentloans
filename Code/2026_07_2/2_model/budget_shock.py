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


SCHEMA_VERSION = 5
N_MEAN_PARAMETERS = 7
N_RISK_PARAMETERS = 4
N_DEBT_PENALTY_PARAMETERS = 4
DEBT_PENALTY_PARAMETERIZATION = "baseline_plus_deviations"
LOAN_HETEROGENEITY_MODES = ("homogeneous", "mean", "variance", "both")
INDEX_KINDS = ("model_period", "education_cell")
EDUCATION_YEAR_STATE_COLUMN = {1: 1, 2: 2, 3: 3}
EDUCATION_BUDGET_YEAR_CAP = {1: 3, 2: 5, 3: 2}
BUDGET_EDUCATION_YEAR_GROUPING = "capped_upper_years_v1"
DEBT_PENALTY_TIMING = "flow_explicit_horizon_periods_1_through_9"
BUDGET_EDUCATION_CELLS = tuple(
    (education, year)
    for education, cap in EDUCATION_BUDGET_YEAR_CAP.items()
    for year in range(1, cap + 1)
)
LEGACY_PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE = 13
PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE = 14
PARENTAL_INCOME_LOAN_TYPE_VECTOR_SIZE = 18
PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL = 6
BUDGET_RESOURCE_SCALE = 10000.0


def education_cell_code(education: int, program_year: int) -> int:
    education = int(education)
    program_year = int(program_year)
    if education not in EDUCATION_YEAR_STATE_COLUMN:
        raise ValueError("education must be 1 (two-year), 2 (four-year), or 3 (graduate)")
    if program_year < 1:
        raise ValueError("program_year must be positive")
    return 100 * education + program_year


def education_cell_from_state(state: np.ndarray, education: int):
    """Return the program-year code using pre-choice experience in ``x2``."""
    education = int(education)
    if education not in EDUCATION_YEAR_STATE_COLUMN:
        raise ValueError("education must be 1, 2, or 3")
    state = np.asarray(state)
    experience = state[..., EDUCATION_YEAR_STATE_COLUMN[education]].astype(np.int64)
    return 100 * education + experience + 1


def budget_program_year(education: int, program_year):
    """Map an actual program year to the estimated budget-shock support."""
    education = int(education)
    if education not in EDUCATION_BUDGET_YEAR_CAP:
        raise ValueError("education must be 1, 2, or 3")
    year = np.asarray(program_year, dtype=np.int64)
    if np.any(year < 1):
        raise ValueError("program_year must be positive")
    grouped = np.minimum(year, EDUCATION_BUDGET_YEAR_CAP[education])
    return int(grouped) if grouped.ndim == 0 else grouped


def budget_education_cell_code(education: int, program_year: int) -> int:
    """Return the capped education-year support code used by budget parameters."""
    return education_cell_code(
        education, budget_program_year(education, program_year)
    )


def budget_education_cell_from_state(state: np.ndarray, education: int):
    """Map the unmodified x2 state into a capped budget-shock support code."""
    actual_code = education_cell_from_state(state, education)
    actual_year = np.asarray(actual_code, dtype=np.int64) - 100 * int(education)
    grouped_year = budget_program_year(education, actual_year)
    return 100 * int(education) + grouped_year


def budget_program_year_label(education: int, program_year: int) -> str:
    """Human-readable label, marking the top grouped support with a plus."""
    grouped = int(budget_program_year(education, program_year))
    cap = EDUCATION_BUDGET_YEAR_CAP[int(education)]
    return f"{grouped}+" if grouped == cap else str(grouped)


def explicit_debt_penalty_multiplier(period, beta: float, terminal_period: int):
    """Discounted count of flow penalties through period ``terminal_period-1``.

    Model periods are one-based. With terminal_period=10, period 1 receives
    penalties for periods 1,...,9 and period 9 receives one penalty. The
    terminal continuation itself is deliberately left unchanged.
    """
    period_array = np.asarray(period, dtype=np.int64)
    remaining = int(terminal_period) - period_array
    if np.any(remaining < 0):
        raise ValueError("period cannot exceed terminal_period")
    if np.isclose(beta, 1.0):
        multiplier = remaining.astype(np.float64)
    else:
        multiplier = (1.0 - float(beta) ** remaining) / (1.0 - float(beta))
    return float(multiplier) if multiplier.ndim == 0 else multiplier


def _validate_heterogeneity_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in LOAN_HETEROGENEITY_MODES:
        raise ValueError(
            f"loan heterogeneity must be one of {LOAN_HETEROGENEITY_MODES}; "
            f"received {mode!r}"
        )
    return mode


def estimation_vector_size(periods, loan_heterogeneity: str = "homogeneous") -> int:
    """Number of SMM parameters under the requested loan-type specification."""
    p = np.asarray(periods, dtype=np.int64).size
    mode = _validate_heterogeneity_mode(loan_heterogeneity)
    extra = p * int(mode in ("mean", "both"))
    extra += p * int(mode in ("variance", "both"))
    return N_MEAN_PARAMETERS * p + p + N_RISK_PARAMETERS + N_DEBT_PENALTY_PARAMETERS + extra


def unpack_estimation_vector(
    vector: np.ndarray,
    periods,
    loan_heterogeneity: str = "homogeneous",
    index_kind: str = "model_period",
) -> dict[str, Any]:
    """Convert the SMM vector into the canonical named specification."""
    periods = np.asarray(periods, dtype=np.int64)
    vector = np.asarray(vector, dtype=np.float64)
    p = periods.size
    mode = _validate_heterogeneity_mode(loan_heterogeneity)
    index_kind = str(index_kind)
    if index_kind not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}")
    expected = estimation_vector_size(periods, mode)
    if vector.size != expected:
        raise ValueError(f"Budget-shock vector has {vector.size} entries; expected {expected}")

    sigma_start = N_MEAN_PARAMETERS * p
    risk_start = sigma_start + p
    penalty_start = risk_start + N_RISK_PARAMETERS
    base_end = penalty_start + N_DEBT_PENALTY_PARAMETERS
    cursor = base_end
    loan_mean_shift = np.zeros(p, dtype=np.float64)
    loan_log_sigma_ratio = np.zeros(p, dtype=np.float64)
    if mode in ("mean", "both"):
        loan_mean_shift = vector[cursor:cursor + p]
        cursor += p
    if mode in ("variance", "both"):
        loan_log_sigma_ratio = vector[cursor:cursor + p]

    result = {
        "schema_version": SCHEMA_VERSION,
        "debt_penalty_timing": DEBT_PENALTY_TIMING,
        "index_kind": index_kind,
        "periods": periods,
        "mu_blocks": vector[:sigma_start].reshape(p, N_MEAN_PARAMETERS),
        "sigma_e": vector[sigma_start:risk_start],
        "risk_aversion": vector[risk_start:penalty_start],
        "debt_pen_parinc": vector[penalty_start:base_end],
        "debt_pen_parameterization": DEBT_PENALTY_PARAMETERIZATION,
        "loan_heterogeneity": mode,
        "loan_mean_shift": loan_mean_shift,
        "loan_log_sigma_ratio": loan_log_sigma_ratio,
        "budget_resource_slope": np.zeros(p, dtype=np.float64),
    }
    if index_kind == "education_cell":
        result["education_year_grouping"] = BUDGET_EDUCATION_YEAR_GROUPING
    return result


def unpack_parental_income_estimation_vector(
    vector: np.ndarray,
    periods,
    index_kind: str = "education_cell",
) -> dict[str, Any]:
    """Map the fast-style vector into the canonical model schema.

    Raw order is four parinc-specific shock means, one common shock sigma,
    four parinc-specific risk-aversion levels, and four parinc-specific debt
    penalty levels, followed by one common slope on pre-choice resources in
    $10,000 units. Legacy 13-entry vectors are accepted with a zero slope. The
    returned named specification is consumed unchanged by the solution and
    simulation code.
    """
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    if periods.size != 1:
        raise ValueError("The parental-income baseline currently supports one education cell.")
    if vector.size not in (
        LEGACY_PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE,
        PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE,
    ):
        raise ValueError(
            f"Parental-income vector has {vector.size} entries; "
            f"expected {LEGACY_PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE} "
            f"(legacy) or {PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE}."
        )
    index_kind = str(index_kind)
    if index_kind not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}")

    mean_levels = vector[0:4]
    debt_levels = vector[9:13]
    result = {
        "schema_version": SCHEMA_VERSION,
        "debt_penalty_timing": DEBT_PENALTY_TIMING,
        "index_kind": index_kind,
        "periods": periods,
        "mu_blocks": np.asarray(
            [[
                mean_levels[0],
                mean_levels[1] - mean_levels[0],
                mean_levels[2] - mean_levels[0],
                mean_levels[3] - mean_levels[0],
                0.0, 0.0, 0.0,
            ]],
            dtype=np.float64,
        ),
        "sigma_e": np.asarray([vector[4]], dtype=np.float64),
        "risk_aversion": np.asarray(vector[5:9], dtype=np.float64),
        "debt_pen_parinc": np.asarray(
            [
                debt_levels[0],
                debt_levels[1] - debt_levels[0],
                debt_levels[2] - debt_levels[0],
                debt_levels[3] - debt_levels[0],
            ],
            dtype=np.float64,
        ),
        "debt_pen_parameterization": DEBT_PENALTY_PARAMETERIZATION,
        "loan_heterogeneity": "homogeneous",
        "loan_mean_shift": np.zeros(1, dtype=np.float64),
        "loan_log_sigma_ratio": np.zeros(1, dtype=np.float64),
        "budget_resource_slope": np.asarray(
            [vector[13] if vector.size == PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE else 0.0],
            dtype=np.float64,
        ),
        "estimation_parameterization": "parental_income_basic",
    }
    if index_kind == "education_cell":
        result["education_year_grouping"] = BUDGET_EDUCATION_YEAR_GROUPING
    return result


def unpack_parental_income_multicell_estimation_vector(
    vector: np.ndarray,
    periods,
    index_kind: str = "education_cell",
) -> dict[str, Any]:
    """Map cell-specific parameters plus common risk aversion to one bundle.

    Each education cell contributes six free parameters in this order: four
    parental-income shock means, one shock sigma, and one pre-choice-resource
    slope.  The final eight entries are four parental-income risk-aversion
    levels followed by four parental-income debt-penalty levels.  Both sets are
    estimated once and shared by all education cells.
    """
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    block_size = PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL * periods.size
    expected = block_size + N_RISK_PARAMETERS + N_DEBT_PENALTY_PARAMETERS
    if periods.size < 2:
        raise ValueError("The multicell parameterization requires at least two cells.")
    if np.unique(periods).size != periods.size:
        raise ValueError("Multicell education-cell codes must be unique.")
    if vector.size != expected:
        raise ValueError(
            f"Multicell parental-income vector has {vector.size} entries; "
            f"expected {expected}."
        )
    shared_risk_aversion = vector[block_size:block_size + N_RISK_PARAMETERS]
    debt_levels = vector[block_size + N_RISK_PARAMETERS:]
    index_kind = str(index_kind)
    if index_kind not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}")

    blocks = vector[:block_size].reshape(
        periods.size, PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
    )
    mean_levels = blocks[:, 0:4]
    mu_blocks = np.zeros((periods.size, N_MEAN_PARAMETERS), dtype=np.float64)
    mu_blocks[:, 0] = mean_levels[:, 0]
    mu_blocks[:, 1:4] = mean_levels[:, 1:4] - mean_levels[:, [0]]
    debt_coefficients = np.asarray(
        [
            debt_levels[0],
            debt_levels[1] - debt_levels[0],
            debt_levels[2] - debt_levels[0],
            debt_levels[3] - debt_levels[0],
        ],
        dtype=np.float64,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "debt_penalty_timing": DEBT_PENALTY_TIMING,
        "index_kind": index_kind,
        "periods": periods,
        "mu_blocks": mu_blocks,
        "sigma_e": blocks[:, 4].copy(),
        "risk_aversion": shared_risk_aversion.copy(),
        "debt_pen_parinc": debt_coefficients,
        "debt_pen_parameterization": DEBT_PENALTY_PARAMETERIZATION,
        "loan_heterogeneity": "homogeneous",
        "loan_mean_shift": np.zeros(periods.size, dtype=np.float64),
        "loan_log_sigma_ratio": np.zeros(periods.size, dtype=np.float64),
        "budget_resource_slope": blocks[:, 5].copy(),
        "estimation_parameterization": "parental_income_basic_multicell_shared_risk_debt",
    }
    if index_kind == "education_cell":
        result["education_year_grouping"] = BUDGET_EDUCATION_YEAR_GROUPING
    return result


def unpack_parental_income_loan_type_estimation_vector(
    vector: np.ndarray,
    periods,
    index_kind: str = "education_cell",
) -> dict[str, Any]:
    """Add four high-loan-type risk-aversion levels to the parinc model.

    The first 14 entries retain exactly the baseline ordering and their four
    risk-aversion levels apply to the low-loan type. Entries 15--18 are the
    corresponding high-loan-type levels. Budget-shock means remain common.
    """
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != PARENTAL_INCOME_LOAN_TYPE_VECTOR_SIZE:
        raise ValueError(
            f"Parental-income loan-type vector has {vector.size} entries; "
            f"expected {PARENTAL_INCOME_LOAN_TYPE_VECTOR_SIZE}."
        )
    spec = unpack_parental_income_estimation_vector(
        vector[:PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE],
        periods,
        index_kind=index_kind,
    )
    spec["loan_heterogeneity"] = "homogeneous"
    spec["loan_mean_shift"] = np.zeros(1, dtype=np.float64)
    spec["risk_aversion_by_loan_type"] = np.vstack(
        (spec["risk_aversion"], vector[14:18])
    ).astype(np.float64)
    spec["estimation_parameterization"] = "parental_income_loan_type"
    return spec


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a loaded or newly constructed specification."""
    required = ("periods", "mu_blocks", "sigma_e", "debt_pen_parinc")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Budget-shock specification is missing: {missing}")

    out = dict(spec)
    out["periods"] = np.asarray(out["periods"], dtype=np.int64)
    out["index_kind"] = str(out.get("index_kind", "model_period"))
    if out["index_kind"] not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}")
    grouping = out.get("education_year_grouping")
    if grouping is not None and grouping != BUDGET_EDUCATION_YEAR_GROUPING:
        raise ValueError(f"Unsupported education_year_grouping: {grouping!r}")
    timing = out.get("debt_penalty_timing")
    if timing is not None and timing != DEBT_PENALTY_TIMING:
        raise ValueError(f"Unsupported debt_penalty_timing: {timing!r}")
    out["mu_blocks"] = np.asarray(out["mu_blocks"], dtype=np.float64)
    out["sigma_e"] = np.asarray(out["sigma_e"], dtype=np.float64)
    out["debt_pen_parinc"] = np.asarray(out["debt_pen_parinc"], dtype=np.float64)
    out["loan_heterogeneity"] = _validate_heterogeneity_mode(
        out.get("loan_heterogeneity", "homogeneous")
    )
    if "risk_aversion" in out:
        out["risk_aversion"] = np.asarray(out["risk_aversion"], dtype=np.float64)
    if "risk_aversion_by_loan_type" in out:
        out["risk_aversion_by_loan_type"] = np.asarray(
            out["risk_aversion_by_loan_type"], dtype=np.float64
        )

    p = out["periods"].size
    out["loan_mean_shift"] = np.asarray(
        out.get("loan_mean_shift", np.zeros(p)), dtype=np.float64
    )
    out["loan_log_sigma_ratio"] = np.asarray(
        out.get("loan_log_sigma_ratio", np.zeros(p)), dtype=np.float64
    )
    out["budget_resource_slope"] = np.asarray(
        out.get("budget_resource_slope", np.zeros(p)), dtype=np.float64
    )
    if out["mu_blocks"].shape != (p, N_MEAN_PARAMETERS):
        raise ValueError("mu_blocks must have shape (number of periods, 7)")
    if out["sigma_e"].shape != (p,) or np.any(out["sigma_e"] <= 0):
        raise ValueError("sigma_e must contain one positive value per period")
    if out["debt_pen_parinc"].shape not in (
        (3,), (N_DEBT_PENALTY_PARAMETERS,), (p, N_DEBT_PENALTY_PARAMETERS),
    ):
        raise ValueError(
            "debt_pen_parinc must have three legacy entries, four current "
            "entries, or one four-entry row per support point"
        )
    if "risk_aversion" in out and out["risk_aversion"].shape != (N_RISK_PARAMETERS,):
        raise ValueError("risk_aversion must have four entries")
    if (
        "risk_aversion_by_loan_type" in out
        and out["risk_aversion_by_loan_type"].shape != (2, N_RISK_PARAMETERS)
    ):
        raise ValueError("risk_aversion_by_loan_type must have shape (2, 4)")
    if out["loan_mean_shift"].shape != (p,):
        raise ValueError("loan_mean_shift must contain one value per period")
    if out["loan_log_sigma_ratio"].shape != (p,):
        raise ValueError("loan_log_sigma_ratio must contain one value per period")
    if out["budget_resource_slope"].shape != (p,):
        raise ValueError("budget_resource_slope must contain one value per period")
    if not np.all(np.isfinite(out["loan_mean_shift"])):
        raise ValueError("loan_mean_shift contains non-finite values")
    if not np.all(np.isfinite(out["loan_log_sigma_ratio"])):
        raise ValueError("loan_log_sigma_ratio contains non-finite values")
    if not np.all(np.isfinite(out["budget_resource_slope"])):
        raise ValueError("budget_resource_slope contains non-finite values")
    if (
        "risk_aversion_by_loan_type" in out
        and not np.all(np.isfinite(out["risk_aversion_by_loan_type"]))
    ):
        raise ValueError("risk_aversion_by_loan_type contains non-finite values")
    return out


def save(spec: dict[str, Any], raw_vector: np.ndarray | None = None,
         filename_prefix: str = "budgetshock") -> None:
    """Save the canonical bundle plus legacy files used by existing scripts."""
    spec = validate(spec)
    if "risk_aversion" not in spec:
        raise ValueError("Cannot save a fitted specification without risk_aversion")
    ENSURE_DIR(os.path.dirname(EST("budgetshock_params.npy")))

    # A named pre-test must not overwrite the production bundle.  Only the
    # canonical ``budgetshock`` prefix updates files consumed by solution and
    # simulation code.
    if filename_prefix == "budgetshock":
        np.save(EST("risk_aversion.npy"), spec["risk_aversion"])
        np.save(EST("budgetshock_params.npy"), spec, allow_pickle=True)
    else:
        np.save(EST(f"{filename_prefix}_risk_aversion.npy"), spec["risk_aversion"])
        np.save(EST(f"{filename_prefix}_params.npy"), spec, allow_pickle=True)
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


def support_value(
    spec: dict[str, Any], period: int | None = None, *, education=None,
    state=None, program_year=None,
) -> int:
    """Map a model call to the support used by the fitted shock bundle."""
    if spec.get("index_kind", "model_period") == "model_period":
        if period is None:
            raise ValueError("period is required for a model-period shock specification")
        return int(period)
    if education is None:
        raise ValueError("education is required for an education-cell shock specification")
    grouped = (
        spec.get("education_year_grouping") == BUDGET_EDUCATION_YEAR_GROUPING
    )
    if program_year is not None:
        if grouped:
            return budget_education_cell_code(education, program_year)
        return education_cell_code(education, program_year)
    if state is None:
        raise ValueError("state or program_year is required for an education-cell specification")
    code = np.asarray(
        budget_education_cell_from_state(state, education)
        if grouped else education_cell_from_state(state, education)
    ).reshape(-1)
    if code.size != 1:
        raise ValueError("A scalar education cell is required for this model call")
    return int(code[0])


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


def _loan_type_array(loan_type, shape):
    if loan_type is None:
        return np.zeros(shape, dtype=np.float64)
    loan_type = np.asarray(loan_type)
    if not np.all(np.isin(loan_type, (0, 1))):
        raise ValueError("loan_type must contain only 0 and 1")
    return np.broadcast_to(loan_type, shape).astype(np.float64)


def conditional_mean(
    spec: dict[str, Any], x1: np.ndarray, period: int | None = None, loan_type=None,
    *, education=None, state=None, program_year=None, pre_choice_resources=None,
) -> np.ndarray:
    support = support_value(
        spec, period, education=education, state=state, program_year=program_year
    )
    base = conditional_mean_from_block(
        x1, spec["mu_blocks"][_period_index(spec, support)]
    )
    loan = _loan_type_array(loan_type, np.shape(base))
    shift = spec.get("loan_mean_shift", np.zeros(len(spec["periods"])))
    index = _period_index(spec, support)
    mean = base + float(shift[index]) * loan
    slope = float(spec.get("budget_resource_slope", np.zeros(len(spec["periods"])))[index])
    if slope != 0.0:
        if pre_choice_resources is None:
            raise ValueError(
                "pre_choice_resources is required when budget_resource_slope is nonzero"
            )
        resources = np.asarray(pre_choice_resources, dtype=np.float64)
        mean = mean + slope * (resources / BUDGET_RESOURCE_SCALE)
    return mean


def conditional_sigma(
    spec: dict[str, Any], period: int | None = None, loan_type=None, *,
    education=None, state=None, program_year=None,
):
    support = support_value(
        spec, period, education=education, state=state, program_year=program_year
    )
    index = _period_index(spec, support)
    base = float(spec["sigma_e"][index])
    if loan_type is None:
        return base
    loan = np.asarray(loan_type)
    if not np.all(np.isin(loan, (0, 1))):
        raise ValueError("loan_type must contain only 0 and 1")
    log_ratio = spec.get("loan_log_sigma_ratio", np.zeros(len(spec["periods"])))
    result = base * np.exp(float(log_ratio[index]) * loan.astype(np.float64))
    return float(result) if result.ndim == 0 else result


def risk_aversion(
    spec: dict[str, Any], x1: np.ndarray, loan_type=None,
) -> np.ndarray:
    if "risk_aversion" not in spec:
        raise ValueError("Budget-shock specification has no risk_aversion parameters")
    par = np.asarray(x1)[..., 0].astype(np.int64)
    if "risk_aversion_by_loan_type" in spec:
        if loan_type is None:
            raise ValueError(
                "loan_type is required for loan-type-specific risk aversion"
            )
        loan = np.asarray(loan_type, dtype=np.int64)
        if not np.all(np.isin(loan, (0, 1))):
            raise ValueError("loan_type must contain only 0 and 1")
        loan, par = np.broadcast_arrays(loan, par)
        return spec["risk_aversion_by_loan_type"][loan, par - 1]
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
                standard_draw: np.ndarray, loan_type=None, *, education=None,
                state=None, program_year=None, pre_choice_resources=None) -> np.ndarray:
    """Map fixed standard draws into budget shocks.

    The SMM estimator and forward simulation both go through this transform.
    For experiments with a different distribution, change this function and
    ``quadrature`` together; no consumer module needs its own distribution
    formula.
    """
    support_keywords = dict(
        education=education, state=state, program_year=program_year,
    )
    return (conditional_mean(
                spec, x1, period, loan_type=loan_type,
                pre_choice_resources=pre_choice_resources, **support_keywords,
            )
            + conditional_sigma(
                spec, period, loan_type=loan_type, **support_keywords,
            )
            * np.asarray(standard_draw, dtype=np.float64))


def draw(
    spec: dict[str, Any], x1: np.ndarray, period: int | None,
    rng: np.random.Generator | None = None, loan_type=None, *, education=None,
    state=None, program_year=None, pre_choice_resources=None,
) -> np.ndarray:
    """Draw the realized additive budget shock used in forward simulation."""
    rng = np.random.default_rng() if rng is None else rng
    keywords = dict(
        education=education, state=state, program_year=program_year,
        pre_choice_resources=pre_choice_resources,
    )
    shape = np.shape(conditional_mean(
        spec, x1, period, loan_type=loan_type, **keywords
    ))
    return realization(
        spec, x1, period, rng.standard_normal(size=shape), loan_type=loan_type,
        **keywords,
    )


def quadrature(
    spec: dict[str, Any], x1: np.ndarray, period: int, degree: int = 5,
    loan_type=None, *, education=None, state=None, program_year=None,
    pre_choice_resources=None,
):
    """Gauss-Hermite nodes and weights for the backward expectation."""
    x1 = np.asarray(x1)
    if x1.ndim > 1 and x1.shape[0] != 1:
        raise ValueError("quadrature expects one invariant state")
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    mean = float(
        np.asarray(conditional_mean(
            spec, x1, period, loan_type=loan_type, education=education,
            state=state, program_year=program_year,
            pre_choice_resources=pre_choice_resources,
        )).reshape(-1)[0]
    )
    sigma = float(np.asarray(
        conditional_sigma(
            spec, period, loan_type=loan_type, education=education,
            state=state, program_year=program_year,
        )
    ).reshape(-1)[0])
    values = np.sqrt(2.0) * sigma * nodes + mean
    return values, weights / np.sqrt(np.pi)
