"""ALTERNATIVE budget-shock estimation: SMM with inner mean-inversion (TEST).

Sergi's request (2026-07-31): test the BLP-style inner inversion. The ten
cell need-mean INTERCEPTS are not searched by the optimizer; at every
objective evaluation each cell's intercept is solved (Brent root-finding on
a monotone function) so the cell's simulated mean NEW BORROWING (annual
flow clipped at zero, averaged over all enrolled person-periods and draws,
zeros included) matches the observed cell mean exactly (tolerance $2).
dfols searches only the remaining free parameters.

Everything else mirrors the production configuration exactly:
need_mixture_v2 moments, debt penalties frozen at zero, risk aversion
frozen at 2.0, dfols, 100 draws, warm start from budgetshock_bestx.npy.
No iteration caps: dfols runs to its own convergence (rhoend).

PRODUCTION FILES ARE NOT TOUCHED:
  - this script only monkeypatches model_fitloans_dynamic IN MEMORY;
  - all outputs are saved with the prefix ``budgetshock_meaninv_*``.

Run on the server (Linux, fork start method: the patches propagate into the
persistent cell workers; on non-fork platforms the script forces serial
cell evaluation so the patches still apply):

    python3 studentloans/Code/2026_07_2/2_model/run_fitloans_mean_inversion.py
"""

import multiprocessing as mp
import time

import numpy as np
from scipy.optimize import brentq

import budget_shock as bs
import model_fitloans_dynamic as mfd

# ---------------------------------------------------------------------------
# Configuration: mirror estimation_all_em.py's budget stage exactly.
# ---------------------------------------------------------------------------
mfd.ESTIMATE_NEW_BORROWING_COST = False
mfd.ESTIMATE_LOAN_TYPE_DEBT_PENALTY = False
mfd.ESTIMATE_NEED_MIXTURE = True
mfd.FREEZE_DEBT_PENALTY = True
mfd.FREEZE_RISK_AVERSION = True
mfd.FIXED_RISK_AVERSION = 2.0

MOMENT_SPEC = "need_mixture_v2"
DRAWS = 100
CELL_NUMBA_THREADS = 9
DFOLS_MAXFUN = 10000          # effectively uncapped; dfols stops at rhoend
TARGET_TOLERANCE = 2.0        # dollars, on the simulated-vs-observed mean
BRACKET = 60000.0             # initial intercept bracket half-width
BRACKET_MAX = 240000.0        # maximum half-width after expansion

# ---------------------------------------------------------------------------
# Patch 1: remove the ten cell mean-intercepts from the optimizer's search
# space (slot 0 of each 6-entry cell block). Their pinned placeholder value
# is irrelevant: the inversion wrapper overrides the intercept at every
# evaluation.
# ---------------------------------------------------------------------------
_original_mask = mfd.multicell_free_parameter_mask


def _mask_with_inverted_intercepts(n_parameters, n_cells):
    mask = _original_mask(n_parameters, n_cells)
    per_cell = bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
    for cell_index in range(int(n_cells)):
        mask[cell_index * per_cell] = False
    return mask


mfd.multicell_free_parameter_mask = _mask_with_inverted_intercepts

# ---------------------------------------------------------------------------
# Patch 2: the inner mean-inversion around the per-cell evaluation.
# ---------------------------------------------------------------------------
_original_evaluate = mfd._evaluate_sampled_parental_income_cell
_cell_targets = {}
_inversion_counters = {}


def _observed_mean_new_borrowing(sample_by_period):
    """Observed mean of max(annual flow, 0) over all enrolled person-periods."""
    flows = np.concatenate(
        [np.asarray(pack["loan_flow"], dtype=np.float64)
         for pack in sample_by_period]
    )
    return float(np.mean(np.clip(flows, 0.0, None)))


def _simulated_mean_new_borrowing(flow_by_draw):
    return float(np.mean(np.clip(flow_by_draw, 0.0, None)))


def _inverting_evaluate(
    full_params, data_moments, sample_by_period, cell_code, education,
    program_year, moment_spec, primary_moment_weight,
    new_borrowing_costs=None, loan_type_debt_penalty_shift=None,
    need_mixture=None, return_flows=False,
):
    cell_key = int(cell_code)
    if cell_key not in _cell_targets:
        _cell_targets[cell_key] = _observed_mean_new_borrowing(sample_by_period)
        _inversion_counters[cell_key] = 0
    target = _cell_targets[cell_key]

    kernel_calls = [0]

    def simulated_mean(intercept):
        trial = np.asarray(full_params, dtype=np.float64).copy()
        trial[0] = intercept
        result = _original_evaluate(
            trial, data_moments, sample_by_period, cell_code, education,
            program_year, moment_spec, primary_moment_weight,
            new_borrowing_costs=new_borrowing_costs,
            loan_type_debt_penalty_shift=loan_type_debt_penalty_shift,
            need_mixture=need_mixture, return_flows=True,
        )
        kernel_calls[0] += 1
        return _simulated_mean_new_borrowing(result[5])

    # simulated mean is monotone NON-INCREASING in the intercept (a higher
    # shock mean gives everyone more resources -> weakly smaller loans).
    # f(mu) = simulated - target is therefore non-increasing: bracket a sign
    # change, expanding if needed, then Brent.
    center = float(full_params[0])
    half_width = BRACKET
    low, high = center - half_width, center + half_width
    f_low, f_high = (simulated_mean(low) - target,
                     simulated_mean(high) - target)
    while f_low * f_high > 0.0 and half_width < BRACKET_MAX:
        half_width *= 2.0
        low, high = center - half_width, center + half_width
        f_low, f_high = (simulated_mean(low) - target,
                         simulated_mean(high) - target)

    fallback = None
    if f_low * f_high > 0.0:
        # Target unreachable inside the maximal bracket: take the closer
        # endpoint and say so. This is itself a diagnostic.
        best_intercept = low if abs(f_low) <= abs(f_high) else high
        fallback = "unreachable-target"
    elif abs(f_low) <= TARGET_TOLERANCE:
        best_intercept = low
    elif abs(f_high) <= TARGET_TOLERANCE:
        best_intercept = high
    else:
        best_intercept = brentq(
            lambda mu: simulated_mean(mu) - target, low, high, xtol=1.0,
        )

    _inversion_counters[cell_key] += 1
    if fallback or _inversion_counters[cell_key] % 25 == 1:
        achieved = simulated_mean(best_intercept)
        kernel_calls[0] -= 1  # reporting call, do not count as search cost
        print(
            f"[mean-inv] cell {cell_key}: mu*={best_intercept:,.1f} "
            f"sim={achieved:,.2f} target={target:,.2f} "
            f"kernel_calls={kernel_calls[0]}"
            + (f" FALLBACK={fallback}" if fallback else ""),
            flush=True,
        )

    final = np.asarray(full_params, dtype=np.float64).copy()
    final[0] = best_intercept
    return _original_evaluate(
        final, data_moments, sample_by_period, cell_code, education,
        program_year, moment_spec, primary_moment_weight,
        new_borrowing_costs=new_borrowing_costs,
        loan_type_debt_penalty_shift=loan_type_debt_penalty_shift,
        need_mixture=need_mixture, return_flows=return_flows,
    )


mfd._evaluate_sampled_parental_income_cell = _inverting_evaluate

# ---------------------------------------------------------------------------
# Patch 3: redirect every save to the meaninv prefix so production files
# (budgetshock_params.npy, budgetshock_bestx.npy, ...) are never overwritten.
# ---------------------------------------------------------------------------
_original_save = mfd.save_budgetshock_estimates


def _redirected_save(*args, **kwargs):
    kwargs["filename_prefix"] = "budgetshock_meaninv"
    return _original_save(*args, **kwargs)


mfd.save_budgetshock_estimates = _redirected_save


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
def main():
    fork_available = "fork" in mp.get_all_start_methods()
    cell_workers = None if fork_available else 1
    if not fork_available:
        print(
            "[mean-inv] fork unavailable: forcing serial cell evaluation "
            "so the in-memory patches apply."
        )

    print("=" * 100)
    print("[mean-inv RUN CONFIG] alternative estimation with inner mean-inversion")
    print(f"  moment spec                = {MOMENT_SPEC}")
    print(f"  draws                      = {DRAWS}")
    print(f"  optimizer                  = dfols, maxfun={DFOLS_MAXFUN} (uncapped in practice)")
    print(f"  inverted parameters        = 10 cell need-mean intercepts")
    print(f"  inversion target           = observed mean of max(flow, 0), zeros included, per cell")
    print(f"  inversion tolerance        = ${TARGET_TOLERANCE}")
    print(f"  frozen                     = 4 debt penalties at 0; 4 risk aversions at {mfd.FIXED_RISK_AVERSION}")
    print(f"  outputs                    = Model/Estimates/budgetshock_meaninv_*")
    print("=" * 100, flush=True)

    started = time.perf_counter()
    result, summary = mfd.estimate_budget_shock_all_education(
        draws=DRAWS,
        maxiter=100000,
        optimizer="dfols",
        moment_spec=MOMENT_SPEC,
        resource_mode="simulated",
        restart=True,
        cell_workers=cell_workers,
        cell_numba_threads=CELL_NUMBA_THREADS,
        dfols_maxfun=DFOLS_MAXFUN,
    )
    elapsed = time.perf_counter() - started
    print(f"\n[mean-inv] finished in {elapsed / 3600.0:.2f} hours", flush=True)
    print("[mean-inv] targets by cell:", flush=True)
    for cell_key, target in sorted(_cell_targets.items()):
        print(f"  cell {cell_key}: observed mean new borrowing = {target:,.2f}")
    return result, summary


if __name__ == "__main__":
    main()
