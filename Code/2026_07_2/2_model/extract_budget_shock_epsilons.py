"""Invert the budget-shock epsilons that rationalize observed loans.

Design document: Agents_Readme/Tasks/epsilon_inversion/README.md.

For every enrolled person-period in the production education cells, and for
every latent joint type, this script computes the interval [xi_lo, xi_hi] of
budget-shock realizations under which the SMM debt kernel's optimal grid
choice equals the OBSERVED debt choice, holding fixed:

  - the ITERATION-1 continuation values (the auxiliary-model CCP sequences,
    ``evt_ccp_dense_initial`` -- the tree the first budget SMM reads);
  - the OBSERVED pre-choice budget (``observed_budget``; the shock is the
    residual that adjusts for whatever budget is fed in);
  - a debt penalty FROZEN AT ZERO and no kappa / no Spec B shift (the
    current need_mixture_v2 production configuration);
  - one value of risk aversion per output file (the sigma grid; terminal
    values are sigma-dependent and recomputed per value).

Sign convention (the CODE convention -- see the README, section 2): the
shock is ADDED to consumption resources, so zero-loan observations yield a
LOWER bound (xi_hi = +inf) and at-cap observations an UPPER bound
(xi_lo = -inf).

The script is read-only with respect to all model state: it writes ONLY its
own output files under Model/Estimates/epsilon_inversion/. Run on the
server (needs the superfeasible data, auxiliary_em_results.npz, the initial
CCP sequences, and the continuation library):

    python3 extract_budget_shock_epsilons.py
    python3 extract_budget_shock_epsilons.py --sigmas 0.5 1.0 --tolerance 0.5

Output: one ``epsilons_sigma_<value>.npz`` per sigma value with per-row
metadata (N,), the EM posterior q (N, 16), and per-type inversion results
(16, N): xi_lo, xi_hi, and a status code
(0 interior / 1 zero / 2 cap / 3 infeasible / 4 degenerate-window).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time

import numpy as np

import budget_shock as bs
import model_fitloans_dynamic as mfd
import model_getccp_sequence_fast as mgsf
from config import EST, ENSURE_DIR
from latent_types import N_TYPES, TYPE_IDS, TYPE_LOAN

STATUS_INTERIOR = 0
STATUS_ZERO = 1
STATUS_CAP = 2
STATUS_INFEASIBLE = 3
STATUS_DEGENERATE = 4
STATUS_LEGEND = (
    "0=interior 1=zero(lower bound only) 2=cap(upper bound only) "
    "3=infeasible(not rationalizable) 4=degenerate(window has one point)"
)

DEFAULT_SIGMA_GRID = tuple(np.round(np.arange(0.25, 2.751, 0.25), 2))
DEFAULT_TOLERANCE = 0.5  # dollars; intervals are grid-spacing wide anyway
OUTPUT_DIRECTORY = "epsilon_inversion"


def kernel_utility(consumption, sigma):
    """The SMM kernel's CRRA flow utility, vectorized, -inf below the floor.

    Replicates ``solve_all_draws_debt_idx_pooled`` exactly: consumption below
    CONSUMPTION_FLOOR is infeasible, and the sigma==1 branch matches the
    kernel's |sigma-1| < 1e-8 log branch.
    """
    consumption = np.asarray(consumption, dtype=np.float64)
    utility = np.full(consumption.shape, -np.inf, dtype=np.float64)
    feasible = consumption >= mfd.CONSUMPTION_FLOOR
    scaled = 0.00001 * consumption[feasible]
    if abs(float(sigma) - 1.0) < 1e-8:
        utility[feasible] = 0.1 * np.log(scaled)
    else:
        utility[feasible] = 0.1 * (scaled ** (1.0 - sigma)) / (1.0 - sigma)
    return utility


def kernel_argmax(epsilon, budget, continuation, window_mask, hi_idx, sigma):
    """The kernel's optimal grid index for one epsilon per observation.

    ``epsilon``/``budget``/``hi_idx`` are (N,), ``continuation`` and
    ``window_mask`` are (N, B). Rows with no feasible consumption fall back
    to the top of the window, exactly like the production kernel.
    """
    consumption = (
        budget[:, None] + epsilon[:, None] + mfd.debt_range[None, :]
    )
    total = kernel_utility(consumption, sigma) + continuation
    total = np.where(window_mask, total, -np.inf)
    best = np.argmax(total, axis=1)
    none_feasible = ~np.isfinite(np.max(total, axis=1))
    return np.where(none_feasible, hi_idx, best).astype(np.int64)


def _bisect_boundary(
    predicate_high, budget, continuation, window_mask, hi_idx, sigma,
    lower, upper, j_obs, tolerance, mode,
):
    """Monotone bisection of one interval endpoint, vectorized over rows.

    The kernel argmax is a NONINCREASING step function of epsilon. For
    ``mode == "hi"`` the boundary separates {argmax >= j_obs} (below) from
    {argmax < j_obs} (above); for ``mode == "lo"`` it separates
    {argmax > j_obs} (below) from {argmax <= j_obs} (above). ``lower`` and
    ``upper`` must bracket the boundary row by row; rows where
    ``predicate_high`` is False are left untouched (their endpoint is
    infinite or the row is not being solved).
    """
    lower = lower.copy()
    upper = upper.copy()
    active = predicate_high.copy()
    while np.any(active) and np.max(upper[active] - lower[active]) > tolerance:
        midpoint = 0.5 * (lower + upper)
        choice = kernel_argmax(
            midpoint, budget, continuation, window_mask, hi_idx, sigma
        )
        if mode == "hi":
            below = choice >= j_obs
        else:
            below = choice > j_obs
        lower = np.where(active & below, midpoint, lower)
        upper = np.where(active & ~below, midpoint, upper)
    return 0.5 * (lower + upper)


def invert_cell_type(
    budget, continuation, b_idx, max_idx, j_obs, sigma, tolerance,
):
    """Rationalizing epsilon intervals for one latent type, all observations.

    Returns (xi_lo, xi_hi, status), each (N,) / (N,) / (N,) int8. The
    conventions follow the README: zero-loan rows (j_obs == b_idx) report
    only the lower endpoint, cap rows (j_obs == max_idx) only the upper one,
    and rows whose observed choice is never optimal for any epsilon are
    flagged infeasible.
    """
    n = budget.size
    grid_size = mfd.debt_range.size
    columns = np.arange(grid_size)[None, :]
    window_mask = (columns >= b_idx[:, None]) & (columns <= max_idx[:, None])

    xi_lo = np.full(n, -np.inf)
    xi_hi = np.full(n, np.inf)
    status = np.full(n, STATUS_INTERIOR, dtype=np.int8)

    out_of_window = (j_obs < b_idx) | (j_obs > max_idx)
    degenerate = (b_idx == max_idx) & ~out_of_window
    status[degenerate] = STATUS_DEGENERATE
    status[out_of_window] = STATUS_INFEASIBLE
    xi_lo[out_of_window] = np.nan
    xi_hi[out_of_window] = np.nan

    is_zero = (j_obs == b_idx) & ~degenerate & ~out_of_window
    is_cap = (j_obs == max_idx) & ~degenerate & ~out_of_window
    status[is_zero] = STATUS_ZERO
    status[is_cap] = STATUS_CAP

    solvable = ~out_of_window & ~degenerate

    # Bracket construction. Below eps_all_infeasible every consumption is
    # under the floor and the kernel falls back to the top of the window, so
    # the argmax starts at max_idx; as epsilon grows, the argmax decreases
    # to j_limit, the argmax of the continuation alone (utility differences
    # vanish as consumption grows).
    eps_floor = (
        mfd.CONSUMPTION_FLOOR - budget - mfd.debt_range[max_idx] - 1.0
    )
    continuation_only = np.where(window_mask, continuation, -np.inf)
    j_limit = np.argmax(continuation_only, axis=1).astype(np.int64)

    # A margin comfortably past the last preference-relevant epsilon. The
    # doubling loop below verifies the bracket and widens it if needed.
    bracket_width = np.full(n, 1.0e7)

    # Upper endpoint: finite iff the argmax eventually drops below j_obs,
    # i.e. iff j_limit < j_obs. (For zero rows j_obs == b_idx <= j_limit,
    # so xi_hi stays +inf automatically -- the zero flag's one-sided bound.)
    needs_hi = solvable & (j_limit < j_obs)
    if np.any(needs_hi):
        lower = eps_floor.copy()
        upper = eps_floor + bracket_width
        for _ in range(20):
            choice = kernel_argmax(
                upper, budget, continuation, window_mask, max_idx, sigma
            )
            unbracketed = needs_hi & (choice >= j_obs)
            if not np.any(unbracketed):
                break
            lower = np.where(unbracketed, upper, lower)
            upper = np.where(unbracketed, upper + bracket_width, upper)
        xi_hi_solved = _bisect_boundary(
            needs_hi, budget, continuation, window_mask, max_idx, sigma,
            lower, upper, j_obs, tolerance, mode="hi",
        )
        xi_hi = np.where(needs_hi, xi_hi_solved, xi_hi)

    # Lower endpoint: finite iff j_obs is below the top of the window (the
    # argmax starts at max_idx via the floor fallback, so for cap rows
    # xi_lo stays -inf -- the cap flag's one-sided bound).
    needs_lo = solvable & (j_obs < max_idx)
    if np.any(needs_lo):
        lower = eps_floor.copy()
        upper = eps_floor + bracket_width
        for _ in range(20):
            choice = kernel_argmax(
                upper, budget, continuation, window_mask, max_idx, sigma
            )
            unbracketed = needs_lo & (choice > j_obs)
            if not np.any(unbracketed):
                break
            lower = np.where(unbracketed, upper, lower)
            upper = np.where(unbracketed, upper + bracket_width, upper)
        xi_lo_solved = _bisect_boundary(
            needs_lo, budget, continuation, window_mask, max_idx, sigma,
            lower, upper, j_obs, tolerance, mode="lo",
        )
        xi_lo = np.where(needs_lo, xi_lo_solved, xi_lo)

    # Verification: the midpoint of every claimed finite interval must
    # reproduce the observed choice; step-function skips (a j never optimal
    # for ANY epsilon) and empty intervals are flagged infeasible.
    check = solvable.copy()
    midpoint = np.where(
        np.isfinite(xi_lo) & np.isfinite(xi_hi), 0.5 * (xi_lo + xi_hi),
        np.where(np.isfinite(xi_lo), xi_lo + 2.0 * tolerance,
                 xi_hi - 2.0 * tolerance),
    )
    choice = kernel_argmax(
        midpoint, budget, continuation, window_mask, max_idx, sigma
    )
    failed = check & (choice != j_obs)
    status[failed] = STATUS_INFEASIBLE
    xi_lo[failed] = np.nan
    xi_hi[failed] = np.nan
    empty = check & ~failed & (xi_lo >= xi_hi)
    status[empty] = STATUS_INFEASIBLE
    xi_lo[empty] = np.nan
    xi_hi[empty] = np.nan
    return xi_lo, xi_hi, status


def load_pooled_observations(ccp_workers):
    """Load every production education cell and pool it, iteration-1 CCPs.

    Points the sequence readers at the auxiliary (initial) tree BEFORE any
    load, exactly like iterations 0-1 of the production driver, and forces
    fresh reads (no cell cache) so the tree selection cannot be bypassed.
    """
    mfd.CCP_SEQUENCE_FORMAT = "dense"
    mgsf.DENSE_ROOT = mgsf.initial_dense_root()
    print(f"[epsilons] CCP sequence tree: {mgsf.DENSE_ROOT}")

    interp_dict = mfd.get_interp_dict_cached(force_rebuild=False)
    cells = mfd.discover_observed_education_cells()
    if not cells:
        raise ValueError("No observed education cells were found.")
    print(f"[epsilons] education cells: {cells}")

    pooled = {name: [] for name in (
        "individual_index", "period", "cell_code", "education",
        "program_year", "parinc", "begin_debt", "observed_flow",
        "budget", "annual_cap", "j_obs", "b_idx", "max_idx",
    )}
    q_rows, ccp_rows, terminal_data_rows = [], [], []
    for education, program_year in cells:
        for period in range(1, mfd.T):
            pack = mfd.load_education_cell(
                period, interp_dict, education=education,
                program_year=program_year, ccp_workers=ccp_workers,
                ccp_cache_mode="off",
            )
            n = len(pack["x1"])
            if not n:
                continue
            j_obs = np.searchsorted(mfd.debt_range, pack["debtchoice"])
            j_obs = np.clip(j_obs, 0, mfd.debt_range.size - 1)
            if not np.allclose(
                mfd.debt_range[j_obs], pack["debtchoice"], atol=1e-6
            ):
                raise ValueError(
                    "Observed debt choices are not on the debt grid in "
                    f"education {education}, year {program_year}, "
                    f"period {period}."
                )
            b_idx, max_idx = mfd.precompute_bounds_indices(
                pack["debt"].astype(np.float64),
                pack["state"].astype(np.int64),
                pack["choice"].astype(np.int64),
            )
            pooled["individual_index"].append(pack["individual_index"])
            pooled["period"].append(np.full(n, period, dtype=np.int64))
            pooled["cell_code"].append(
                np.full(n, int(pack["cell_code"]), dtype=np.int64)
            )
            pooled["education"].append(np.full(n, education, dtype=np.int64))
            pooled["program_year"].append(
                np.full(n, program_year, dtype=np.int64)
            )
            pooled["parinc"].append(pack["parinc"])
            pooled["begin_debt"].append(pack["debt"])
            pooled["observed_flow"].append(pack["loan_flow"])
            pooled["budget"].append(pack["observed_budget"])
            pooled["annual_cap"].append(pack["annual_cap"])
            pooled["j_obs"].append(j_obs.astype(np.int64))
            pooled["b_idx"].append(b_idx.astype(np.int64))
            pooled["max_idx"].append(max_idx.astype(np.int64))
            q_rows.append(pack["q"])
            ccp_rows.append(np.asarray(pack["ccp_by_type"], dtype=np.float64))
            terminal_data_rows.append((n, period, pack["terminal_data"]))
            print(
                f"  loaded education={education} year={program_year} "
                f"period={period}: {n} observations"
            )

    pooled = {name: np.concatenate(parts) for name, parts in pooled.items()}
    pooled["q"] = np.concatenate(q_rows, axis=0)
    pooled["ccp_by_type"] = np.concatenate(ccp_rows, axis=1)
    pooled["terminal_blocks"] = terminal_data_rows
    total = pooled["budget"].size
    print(f"[epsilons] pooled observations: {total}")
    return pooled


def build_terminal(pooled, sigma):
    """Sigma-dependent terminal rows and beta^(T-t) factors, pooled order."""
    total = pooled["budget"].size
    terminal = np.zeros((total, mfd.debt_range.size), dtype=np.float64)
    beta_term = np.zeros(total, dtype=np.float64)
    cache = {}
    offset = 0
    for n, period, terminal_data in pooled["terminal_blocks"]:
        rows = np.arange(n)
        terminal[offset:offset + n] = mfd.get_relevant_terminal_subset_cached(
            terminal_data, float(sigma), rows, cache,
        )
        beta_term[offset:offset + n] = float(mfd.beta ** (mfd.T - period))
        offset += n
    return terminal, beta_term


# Per-process state for the sigma-level pool. Populated in the parent BEFORE
# the pool is created, so fork-based workers inherit the pooled arrays
# copy-on-write instead of pickling ~100 MB per task.
_SIGMA_CONTEXT = {}


def _extract_one_sigma(sigma):
    """Run the full inversion for one sigma value and write its npz file."""
    pooled = _SIGMA_CONTEXT["pooled"]
    posterior_high = _SIGMA_CONTEXT["posterior_high"]
    type_indices = _SIGMA_CONTEXT["type_indices"]
    tolerance = _SIGMA_CONTEXT["tolerance"]
    output_dir = _SIGMA_CONTEXT["output_dir"]
    started = time.perf_counter()
    terminal, beta_term = build_terminal(pooled, sigma)
    total = pooled["budget"].size
    xi_lo = np.full((N_TYPES, total), np.nan)
    xi_hi = np.full((N_TYPES, total), np.nan)
    status = np.full((N_TYPES, total), STATUS_INFEASIBLE, dtype=np.int8)
    for type_index in type_indices:
        continuation = (
            pooled["ccp_by_type"][type_index]
            + beta_term[:, None] * terminal
        )
        lo, hi, flags = invert_cell_type(
            pooled["budget"], continuation,
            pooled["b_idx"], pooled["max_idx"], pooled["j_obs"],
            float(sigma), float(tolerance),
        )
        xi_lo[type_index] = lo
        xi_hi[type_index] = hi
        status[type_index] = flags
        counts = {
            name: int(np.sum(flags == code))
            for name, code in (
                ("interior", STATUS_INTERIOR), ("zero", STATUS_ZERO),
                ("cap", STATUS_CAP), ("infeasible", STATUS_INFEASIBLE),
                ("degenerate", STATUS_DEGENERATE),
            )
        }
        print(
            f"  sigma={sigma:g} type={TYPE_IDS[type_index]}: {counts}",
            flush=True,
        )

    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sigma": float(sigma),
        "tolerance": float(tolerance),
        "ccp_tree": str(mgsf.DENSE_ROOT),
        "status_legend": STATUS_LEGEND,
        "type_ids": [int(value) for value in TYPE_IDS],
        "type_loan": [int(value) for value in TYPE_LOAN],
        "sign_convention": (
            "shock is ADDED to resources; zero -> lower bound only "
            "(xi_hi=+inf); cap -> upper bound only (xi_lo=-inf)"
        ),
    }
    output_path = f"{output_dir}/epsilons_sigma_{sigma:.2f}.npz"
    np.savez_compressed(
        output_path,
        xi_lo=xi_lo, xi_hi=xi_hi, status=status,
        q=pooled["q"], posterior_high=posterior_high,
        individual_index=pooled["individual_index"],
        period=pooled["period"], cell_code=pooled["cell_code"],
        education=pooled["education"],
        program_year=pooled["program_year"],
        parinc=pooled["parinc"], begin_debt=pooled["begin_debt"],
        observed_flow=pooled["observed_flow"], budget=pooled["budget"],
        annual_cap=pooled["annual_cap"], j_obs=pooled["j_obs"],
        b_idx=pooled["b_idx"], max_idx=pooled["max_idx"],
        debt_grid=mfd.debt_range,
        metadata=np.frombuffer(
            json.dumps(metadata).encode("utf-8"), dtype=np.uint8
        ),
    )
    print(
        f"[epsilons] wrote {output_path} "
        f"({time.perf_counter() - started:,.1f} s)",
        flush=True,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Invert budget-shock epsilons that rationalize observed loans."
    )
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMA_GRID),
        help="Risk-aversion grid (default 0.25:0.25:2.75).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Bisection tolerance for interval endpoints, dollars.",
    )
    parser.add_argument(
        "--ccp-workers", type=int, default=mfd.DEFAULT_CCP_WORKERS,
        help="Workers for the CCP-path load.",
    )
    parser.add_argument(
        "--types", type=int, nargs="+", default=None,
        help="Subset of joint-type indices 0-15 (default: all sixteen).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Processes for the sigma-level parallelism (default: one per "
            "sigma value). Sigmas are independent -- each writes its own "
            "file -- so the full grid runs in roughly the time of one."
        ),
    )
    args = parser.parse_args()

    type_indices = (
        list(range(N_TYPES)) if args.types is None else sorted(set(args.types))
    )
    if any(index < 0 or index >= N_TYPES for index in type_indices):
        raise ValueError(f"Type indices must lie in 0..{N_TYPES - 1}.")

    output_dir = ENSURE_DIR(EST(OUTPUT_DIRECTORY))
    pooled = load_pooled_observations(args.ccp_workers)
    posterior_high = np.ascontiguousarray(
        pooled["q"][:, np.flatnonzero(TYPE_LOAN == 1)].sum(axis=1)
    )

    # Populate the shared context BEFORE creating the pool: fork-based
    # workers inherit the arrays copy-on-write, so nothing large is pickled.
    _SIGMA_CONTEXT["pooled"] = pooled
    _SIGMA_CONTEXT["posterior_high"] = posterior_high
    _SIGMA_CONTEXT["type_indices"] = type_indices
    _SIGMA_CONTEXT["tolerance"] = float(args.tolerance)
    _SIGMA_CONTEXT["output_dir"] = output_dir

    sigmas = list(args.sigmas)
    workers = (
        len(sigmas) if args.workers is None
        else max(1, min(int(args.workers), len(sigmas)))
    )
    if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        print(f"[epsilons] {workers} sigma workers (fork)")
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            pool.map(_extract_one_sigma, sigmas)
    else:
        if workers > 1:
            print("[epsilons] fork unavailable; running sigmas sequentially")
        for sigma in sigmas:
            _extract_one_sigma(sigma)


if __name__ == "__main__":
    main()
