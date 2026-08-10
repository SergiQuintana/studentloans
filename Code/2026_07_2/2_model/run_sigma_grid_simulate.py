# -*- coding: utf-8 -*-
"""Baseline simulation for one sigma-grid point (called by run_sigma_grid.py).

Mirrors the estimated-baseline half of simulation_fit_conterfactual_em.py:
solve every invariant state in simulation mode with the just-estimated
parameters, then simulate ``--samples`` cohorts (each cohort is one simulated
replica of every initial individual). The only differences from the original
driver are (i) the number of cohorts is a command-line argument, and (ii) the
cohort pool is capped at ``--workers`` processes instead of one process per
cohort, so 100 cohorts do not spawn 100 concurrent processes.

Everything the simulation writes goes to the usual trees
(choice/, state/, epsilon/, welfare/, grad_prob/, types/ under Model/Output);
run_sigma_grid.py moves them into the per-sigma archive afterwards. The
epsilon tree holds the per-cohort taste draws that the counterfactual
pipeline reloads, so archived sigmas remain counterfactual-ready without the
solution matrices.

Standalone use:
    python3 run_sigma_grid_simulate.py --samples 100 --workers 50
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

import model_solution_em as ms  # noqa: E402
import model_simulation_em as msim  # noqa: E402
# Importing the counterfactual driver also creates the runtime directories
# (vjt/evt trees and the simulation output trees) at module import.
from simulation_fit_conterfactual_em import solve_simulation_values  # noqa: E402
from latent_types import TYPE_IDS, load_em_posteriors  # noqa: E402
from config import EST  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100,
                        help="simulated cohorts (replicas per initial "
                             "individual)")
    parser.add_argument("--workers", type=int, default=50,
                        help="maximum concurrent cohort simulations")
    arguments = parser.parse_args()

    debt_range = ms.get_debt_range()
    uparams = [ms.load_param_g(type_id, real=1) for type_id in TYPE_IDS]
    q = load_em_posteriors(EST("auxiliary_em_results.npz"))

    print(f"[sigma-grid sim] solving simulation values "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    started = time.perf_counter()
    solve_simulation_values(conterfactual=0, maxdebt=True,
                            uparams=uparams, debt_range=debt_range)
    print(f"[sigma-grid sim] solve done after "
          f"{(time.perf_counter() - started) / 3600.0:.2f} hours", flush=True)

    # ``sigma_u`` is the legacy positional argument of simulate_choices; the
    # simulation reloads risk aversion from the canonical budget-shock bundle
    # (which carries the frozen sigma of this grid point).
    sigma_u = 1.4
    cohort_args = [
        (cohort, sigma_u, 0, q, True, uparams)
        for cohort in range(1, arguments.samples + 1)
    ]
    n_workers = max(1, min(arguments.workers, arguments.samples))
    print(f"[sigma-grid sim] simulating {arguments.samples} cohorts with "
          f"{n_workers} workers", flush=True)
    started = time.perf_counter()
    with multiprocessing.Pool(n_workers) as pool_obj:
        pool_obj.starmap(msim.simulate_choices, cohort_args, chunksize=1)
    print(f"[sigma-grid sim] simulation done after "
          f"{(time.perf_counter() - started) / 3600.0:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
