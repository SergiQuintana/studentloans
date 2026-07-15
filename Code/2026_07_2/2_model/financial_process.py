# -*- coding: utf-8 -*-
"""Shared education-specific grant process for solution and simulation.

Structural alternatives identify education level and work status, but not
current college type. Each grant equation therefore uses the nine invariant
controls plus part-time and full-time indicators.
"""

from pathlib import Path

import numpy as np

from latent_types import validate_saved_layout

try:
    from numba import njit
except ImportError:  # Keep loading/reporting utilities usable without Numba.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator


EDUCATION_LEVELS = (1, 2, 3)
EDUCATION_NAMES = {1: "two_year", 2: "four_year", 3: "graduate"}
N_INVARIANT = 9
N_DESIGN = 11


def _binary_type(values, size=None, label="financial_type"):
    """Validate a scalar or vector zero/one type indicator."""
    values = np.asarray(values)
    if values.ndim == 0:
        value = int(values)
        if value not in (0, 1) or float(values) != value:
            raise ValueError(f"{label} must be 0 or 1.")
        if size is None:
            return value
        return np.full(int(size), value, dtype=np.int64)
    values = values.reshape(-1)
    if size is not None and len(values) != int(size):
        raise ValueError(f"{label} must contain one value per observation.")
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"{label} must contain only 0 and 1.")
    return values.astype(np.int64)


def _result_file(path):
    path = Path(path)
    if path.is_dir():
        path = path / "auxiliary_em_results.npz"
    return path


def load_auxiliary_financial_process(results_path):
    """Load the grant and transfer hurdle models estimated by the auxiliary EM.

    ``results_path`` may be the results file itself or the estimates directory
    containing ``auxiliary_em_results.npz``. The saved type layout is validated
    before any parameters are returned.
    """
    path = _result_file(results_path)
    required = {
        "type_names",
        "type_school",
        "type_grant",
        "type_transfer",
        "type_loan",
        "grant_education_levels",
        "grant_receipt",
        "grant_amount",
        "grant_sigma",
        "transfer_receipt",
        "transfer_amount",
        "transfer_sigma",
    }
    with np.load(path, allow_pickle=False) as results:
        missing = sorted(required.difference(results.files))
        if missing:
            raise ValueError(
                f"Auxiliary EM results {path} are missing: {', '.join(missing)}."
            )
        validate_saved_layout(
            results["type_names"],
            results["type_school"],
            results["type_grant"],
            results["type_transfer"],
            results["type_loan"],
        )
        education_levels = np.asarray(
            results["grant_education_levels"], dtype=np.int64
        )
        grant_receipt = np.asarray(results["grant_receipt"], dtype=float)
        grant_amount = np.asarray(results["grant_amount"], dtype=float)
        grant_sigma = np.asarray(results["grant_sigma"], dtype=float).reshape(-1)
        transfer_receipt = np.asarray(results["transfer_receipt"], dtype=float).reshape(-1)
        transfer_amount = np.asarray(results["transfer_amount"], dtype=float).reshape(-1)
        transfer_sigma = float(np.asarray(results["transfer_sigma"]).reshape(()))

    if not np.array_equal(education_levels, np.asarray(EDUCATION_LEVELS)):
        raise ValueError(
            "Auxiliary grant education levels must be ordered as [1, 2, 3]."
        )
    expected_grant_shape = (len(EDUCATION_LEVELS), N_DESIGN + 1)
    if grant_receipt.shape != expected_grant_shape:
        raise ValueError(
            f"Grant receipt parameters have shape {grant_receipt.shape}; "
            f"expected {expected_grant_shape}."
        )
    if grant_amount.shape != expected_grant_shape:
        raise ValueError(
            f"Grant amount parameters have shape {grant_amount.shape}; "
            f"expected {expected_grant_shape}."
        )
    if grant_sigma.shape != (len(EDUCATION_LEVELS),):
        raise ValueError("Grant sigma must contain one value per education level.")
    expected_transfer_size = N_INVARIANT + 4 + 1
    if transfer_receipt.shape != (expected_transfer_size,):
        raise ValueError(
            f"Transfer receipt parameters contain {len(transfer_receipt)} values; "
            f"expected {expected_transfer_size}."
        )
    if transfer_amount.shape != (expected_transfer_size,):
        raise ValueError(
            f"Transfer amount parameters contain {len(transfer_amount)} values; "
            f"expected {expected_transfer_size}."
        )
    arrays = (
        grant_receipt,
        grant_amount,
        grant_sigma,
        transfer_receipt,
        transfer_amount,
        np.asarray([transfer_sigma]),
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Auxiliary financial parameters contain non-finite values.")
    if np.any(grant_sigma <= 0.0) or transfer_sigma <= 0.0:
        raise ValueError("Financial amount standard deviations must be positive.")

    return {
        "grant": {
            "receipt": grant_receipt,
            "amount": grant_amount,
            "sigma": grant_sigma,
        },
        "transfer": {
            "receipt": transfer_receipt,
            "amount": transfer_amount,
            "sigma": transfer_sigma,
        },
    }


def prepare_type_financial_parameters(process, grant_type, transfer_type):
    """Preselect one type's coefficients for repeated Bellman evaluations.

    The auxiliary EM stores each type shift as the last coefficient. Because
    the first entry of ``x1`` is a constant, setup can absorb the shift into
    the intercept. The returned tuple contains only contiguous numeric arrays
    and is therefore suitable for the Numba hot path.
    """
    grant_type = _binary_type(grant_type, label="grant_type")
    transfer_type = _binary_type(transfer_type, label="transfer_type")

    grant = process["grant"]
    transfer = process["transfer"]
    grant_receipt_raw = np.asarray(grant["receipt"], dtype=np.float64)
    grant_amount_raw = np.asarray(grant["amount"], dtype=np.float64)
    transfer_receipt_raw = np.asarray(transfer["receipt"], dtype=np.float64)
    transfer_amount_raw = np.asarray(transfer["amount"], dtype=np.float64)

    if grant_receipt_raw.shape != (3, N_DESIGN + 1):
        raise ValueError("Typed grant receipt parameters must have shape (3, 12).")
    if grant_amount_raw.shape != (3, N_DESIGN + 1):
        raise ValueError("Typed grant amount parameters must have shape (3, 12).")
    if transfer_receipt_raw.shape != (N_INVARIANT + 5,):
        raise ValueError("Typed transfer receipt parameters must contain 14 values.")
    if transfer_amount_raw.shape != (N_INVARIANT + 5,):
        raise ValueError("Typed transfer amount parameters must contain 14 values.")

    grant_receipt = grant_receipt_raw[:, :N_DESIGN].copy()
    grant_amount = grant_amount_raw[:, :N_DESIGN].copy()
    transfer_receipt = transfer_receipt_raw[:-1].copy()
    transfer_amount = transfer_amount_raw[:-1].copy()

    # x1[0] is one, so adding the type shift to the intercept is exact.
    grant_receipt[:, 0] += grant_type * grant_receipt_raw[:, -1]
    grant_amount[:, 0] += grant_type * grant_amount_raw[:, -1]
    transfer_receipt[0] += transfer_type * transfer_receipt_raw[-1]
    transfer_amount[0] += transfer_type * transfer_amount_raw[-1]

    return (
        np.ascontiguousarray(grant_receipt),
        np.ascontiguousarray(grant_amount),
        np.ascontiguousarray(grant["sigma"], dtype=np.float64),
        np.ascontiguousarray(transfer_receipt),
        np.ascontiguousarray(transfer_amount),
        float(transfer["sigma"]),
    )


@njit(cache=True)
def expected_financial_help_numba(
    x1,
    education,
    work,
    grant_receipt,
    grant_amount,
    grant_sigma,
    transfer_receipt,
    transfer_amount,
    transfer_sigma,
):
    """Expected grant plus parental transfer for one Bellman alternative.

    All validation and type selection occur before this hot-path function.
    ``x1`` has nine entries; education is 0--3 and work is 0--2. Transfers
    remain eligible only for two- and four-year enrollment, matching the EM.
    """
    if education < 1 or education > 3:
        return 0.0

    education_index = education - 1
    part_time = 1.0 if work == 1 else 0.0
    full_time = 1.0 if work == 2 else 0.0

    grant_receipt_index = 0.0
    grant_amount_index = 0.0
    for column in range(N_INVARIANT):
        grant_receipt_index += x1[column] * grant_receipt[education_index, column]
        grant_amount_index += x1[column] * grant_amount[education_index, column]
    grant_receipt_index += (
        part_time * grant_receipt[education_index, 9]
        + full_time * grant_receipt[education_index, 10]
    )
    grant_amount_index += (
        part_time * grant_amount[education_index, 9]
        + full_time * grant_amount[education_index, 10]
    )

    if grant_receipt_index >= 0.0:
        grant_probability = 1.0 / (1.0 + np.exp(-grant_receipt_index))
    else:
        grant_exp = np.exp(grant_receipt_index)
        grant_probability = grant_exp / (1.0 + grant_exp)
    grant_log_mean = grant_amount_index + 0.5 * grant_sigma[education_index] ** 2
    grant_log_mean = min(20.0, max(-20.0, grant_log_mean))
    expected_grant = grant_probability * np.exp(grant_log_mean)

    if education not in (1, 2):
        return expected_grant

    transfer_receipt_index = 0.0
    transfer_amount_index = 0.0
    for column in range(N_INVARIANT):
        transfer_receipt_index += x1[column] * transfer_receipt[column]
        transfer_amount_index += x1[column] * transfer_amount[column]
    four_year = 1.0 if education == 2 else 0.0
    graduate = 1.0 if education == 3 else 0.0
    transfer_receipt_index += (
        four_year * transfer_receipt[9]
        + graduate * transfer_receipt[10]
        + part_time * transfer_receipt[11]
        + full_time * transfer_receipt[12]
    )
    transfer_amount_index += (
        four_year * transfer_amount[9]
        + graduate * transfer_amount[10]
        + part_time * transfer_amount[11]
        + full_time * transfer_amount[12]
    )

    if transfer_receipt_index >= 0.0:
        transfer_probability = 1.0 / (1.0 + np.exp(-transfer_receipt_index))
    else:
        transfer_exp = np.exp(transfer_receipt_index)
        transfer_probability = transfer_exp / (1.0 + transfer_exp)
    transfer_log_mean = transfer_amount_index + 0.5 * transfer_sigma ** 2
    transfer_log_mean = min(20.0, max(-20.0, transfer_log_mean))
    expected_transfer = transfer_probability * np.exp(transfer_log_mean)
    return expected_grant + expected_transfer


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


def expected_grant_scalar(x1, education, work, process, grant_type=0):
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
    grant_type = _binary_type(grant_type, label="grant_type")
    typed = len(receipt) == N_DESIGN + 1 and len(amount) == N_DESIGN + 1
    if len(receipt) not in (N_DESIGN, N_DESIGN + 1) or len(amount) not in (
        N_DESIGN,
        N_DESIGN + 1,
    ):
        raise ValueError("Grant process has an invalid coefficient layout.")
    if (len(receipt) == N_DESIGN + 1) != (len(amount) == N_DESIGN + 1):
        raise ValueError("Grant receipt and amount layouts do not agree.")
    eta_receipt = (
        x1 @ receipt[:N_INVARIANT]
        + part_time * receipt[9]
        + full_time * receipt[10]
        + (grant_type * receipt[-1] if typed else 0.0)
    )
    eta_amount = (
        x1 @ amount[:N_INVARIANT]
        + part_time * amount[9]
        + full_time * amount[10]
        + (grant_type * amount[-1] if typed else 0.0)
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
    conditional_mean = np.exp(
        np.clip(eta_amount + 0.5 * process["sigma"][index] ** 2, -20.0, 20.0)
    )
    return float(probability * conditional_mean)


def expected_grants_vectorized(x1, education, work, process, grant_type=0):
    """Expected grants for arrays of agents without an agent-level loop."""
    x1 = np.asarray(x1, dtype=float)
    education = np.asarray(education, dtype=int).reshape(-1)
    work = np.asarray(work, dtype=int).reshape(-1)
    if x1.ndim != 2 or x1.shape[0] != len(education) or len(work) != len(education):
        raise ValueError("Grant inputs must have one x1, education, and work row per agent.")
    grant_type = _binary_type(grant_type, len(education), "grant_type")

    expected = np.zeros(len(education), dtype=float)
    for level in EDUCATION_LEVELS:
        selected = education == level
        if not np.any(selected):
            continue
        index = level - 1
        design = x1[selected]
        receipt = process["receipt"][index]
        amount = process["amount"][index]
        typed = len(receipt) == N_DESIGN + 1 and len(amount) == N_DESIGN + 1
        if len(receipt) not in (N_DESIGN, N_DESIGN + 1) or len(amount) not in (
            N_DESIGN,
            N_DESIGN + 1,
        ):
            raise ValueError("Grant process has an invalid coefficient layout.")
        if (len(receipt) == N_DESIGN + 1) != (len(amount) == N_DESIGN + 1):
            raise ValueError("Grant receipt and amount layouts do not agree.")
        part_time = (work[selected] == 1).astype(float)
        full_time = (work[selected] == 2).astype(float)
        eta_receipt = (
            design @ receipt[:N_INVARIANT]
            + part_time * receipt[9]
            + full_time * receipt[10]
            + (grant_type[selected] * receipt[-1] if typed else 0.0)
        )
        eta_amount = (
            design @ amount[:N_INVARIANT]
            + part_time * amount[9]
            + full_time * amount[10]
            + (grant_type[selected] * amount[-1] if typed else 0.0)
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
        conditional_mean = np.exp(
            np.clip(eta_amount + 0.5 * process["sigma"][index] ** 2, -20.0, 20.0)
        )
        expected[selected] = probability * conditional_mean
    return expected


def draw_grants_vectorized(
    x1, education, work, process, grant_type=0,
    receipt_uniform=None, amount_standard_normal=None,
):
    """Draw grant receipt and positive amounts using supplied common draws."""
    x1 = np.asarray(x1, dtype=float)
    education = np.asarray(education, dtype=np.int64).reshape(-1)
    work = np.asarray(work, dtype=np.int64).reshape(-1)
    n = len(education)
    grant_type = _binary_type(grant_type, n, "grant_type")
    receipt_uniform = np.asarray(receipt_uniform, dtype=float).reshape(-1)
    amount_standard_normal = np.asarray(
        amount_standard_normal, dtype=float
    ).reshape(-1)
    if x1.shape != (n, N_INVARIANT):
        raise ValueError(f"Grant x1 must have shape {(n, N_INVARIANT)}.")
    if receipt_uniform.shape != (n,) or amount_standard_normal.shape != (n,):
        raise ValueError("Grant common draws must contain one value per observation.")

    out = np.zeros(n, dtype=float)
    for level in EDUCATION_LEVELS:
        selected = education == level
        if not np.any(selected):
            continue
        index = level - 1
        receipt = np.asarray(process["receipt"][index], dtype=float)
        amount = np.asarray(process["amount"][index], dtype=float)
        part_time = (work[selected] == 1).astype(float)
        full_time = (work[selected] == 2).astype(float)
        eta_receipt = (
            x1[selected] @ receipt[:N_INVARIANT]
            + part_time * receipt[9]
            + full_time * receipt[10]
            + grant_type[selected] * receipt[-1]
        )
        eta_amount = (
            x1[selected] @ amount[:N_INVARIANT]
            + part_time * amount[9]
            + full_time * amount[10]
            + grant_type[selected] * amount[-1]
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
        received = receipt_uniform[selected] < probability
        amount_draw = np.exp(
            np.clip(
                eta_amount
                + float(process["sigma"][index]) * amount_standard_normal[selected],
                -20.0,
                20.0,
            )
        )
        out[selected] = received * amount_draw
    return out


def _transfer_alt_features(education, work):
    education = np.asarray(education, dtype=np.int64).reshape(-1)
    work = np.asarray(work, dtype=np.int64).reshape(-1)
    return np.column_stack(
        (
            education == 2,
            education == 3,
            work == 1,
            work == 2,
        )
    ).astype(float)


def expected_transfer_scalar(x1, education, work, process, transfer_type=0):
    """Expected parental transfer for one state and structural alternative."""
    education = int(education)
    if education not in (1, 2):
        return 0.0
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    if len(x1) != N_INVARIANT:
        raise ValueError(f"Scalar transfer x1 has {len(x1)} entries; expected {N_INVARIANT}.")
    transfer_type = _binary_type(transfer_type, label="transfer_type")
    receipt = np.asarray(process["receipt"], dtype=float).reshape(-1)
    amount = np.asarray(process["amount"], dtype=float).reshape(-1)
    expected_size = N_INVARIANT + 4 + 1
    if len(receipt) != expected_size or len(amount) != expected_size:
        raise ValueError("Transfer process must contain 13 base coefficients and one type shift.")
    alt = _transfer_alt_features([education], [work])[0]
    eta_receipt = x1 @ receipt[:N_INVARIANT] + alt @ receipt[N_INVARIANT:-1]
    eta_amount = x1 @ amount[:N_INVARIANT] + alt @ amount[N_INVARIANT:-1]
    eta_receipt += transfer_type * receipt[-1]
    eta_amount += transfer_type * amount[-1]
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
    conditional_mean = np.exp(
        np.clip(eta_amount + 0.5 * float(process["sigma"]) ** 2, -20.0, 20.0)
    )
    return float(probability * conditional_mean)


def expected_transfers_vectorized(
    x1, education, work, process, transfer_type=0
):
    """Expected parental transfers for arrays of agents or alternatives."""
    x1 = np.asarray(x1, dtype=float)
    education = np.asarray(education, dtype=np.int64).reshape(-1)
    work = np.asarray(work, dtype=np.int64).reshape(-1)
    if x1.ndim != 2 or x1.shape[0] != len(education) or len(work) != len(education):
        raise ValueError(
            "Transfer inputs must have one x1, education, and work row per observation."
        )
    transfer_type = _binary_type(transfer_type, len(education), "transfer_type")
    receipt = np.asarray(process["receipt"], dtype=float).reshape(-1)
    amount = np.asarray(process["amount"], dtype=float).reshape(-1)
    expected_size = N_INVARIANT + 4 + 1
    if len(receipt) != expected_size or len(amount) != expected_size:
        raise ValueError("Transfer process must contain 13 base coefficients and one type shift.")

    expected = np.zeros(len(education), dtype=float)
    selected = np.isin(education, (1, 2))
    if not np.any(selected):
        return expected
    alt = _transfer_alt_features(education[selected], work[selected])
    eta_receipt = (
        x1[selected] @ receipt[:N_INVARIANT]
        + alt @ receipt[N_INVARIANT:-1]
        + transfer_type[selected] * receipt[-1]
    )
    eta_amount = (
        x1[selected] @ amount[:N_INVARIANT]
        + alt @ amount[N_INVARIANT:-1]
        + transfer_type[selected] * amount[-1]
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
    conditional_mean = np.exp(
        np.clip(eta_amount + 0.5 * float(process["sigma"]) ** 2, -20.0, 20.0)
    )
    expected[selected] = probability * conditional_mean
    return expected


def draw_transfers_vectorized(
    x1, education, work, process, transfer_type=0,
    receipt_uniform=None, amount_standard_normal=None,
):
    """Draw parental-transfer receipt and amounts using supplied common draws."""
    x1 = np.asarray(x1, dtype=float)
    education = np.asarray(education, dtype=np.int64).reshape(-1)
    work = np.asarray(work, dtype=np.int64).reshape(-1)
    n = len(education)
    transfer_type = _binary_type(transfer_type, n, "transfer_type")
    receipt_uniform = np.asarray(receipt_uniform, dtype=float).reshape(-1)
    amount_standard_normal = np.asarray(
        amount_standard_normal, dtype=float
    ).reshape(-1)
    if x1.shape != (n, N_INVARIANT):
        raise ValueError(f"Transfer x1 must have shape {(n, N_INVARIANT)}.")
    if receipt_uniform.shape != (n,) or amount_standard_normal.shape != (n,):
        raise ValueError("Transfer common draws must contain one value per observation.")

    out = np.zeros(n, dtype=float)
    selected = np.isin(education, (1, 2))
    if not np.any(selected):
        return out
    receipt = np.asarray(process["receipt"], dtype=float)
    amount = np.asarray(process["amount"], dtype=float)
    alt = _transfer_alt_features(education[selected], work[selected])
    eta_receipt = (
        x1[selected] @ receipt[:N_INVARIANT]
        + alt @ receipt[N_INVARIANT:-1]
        + transfer_type[selected] * receipt[-1]
    )
    eta_amount = (
        x1[selected] @ amount[:N_INVARIANT]
        + alt @ amount[N_INVARIANT:-1]
        + transfer_type[selected] * amount[-1]
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta_receipt, -40.0, 40.0)))
    received = receipt_uniform[selected] < probability
    amount_draw = np.exp(
        np.clip(
            eta_amount
            + float(process["sigma"]) * amount_standard_normal[selected],
            -20.0,
            20.0,
        )
    )
    out[selected] = received * amount_draw
    return out
