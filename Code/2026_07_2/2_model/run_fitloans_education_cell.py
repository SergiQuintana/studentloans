"""Command-line entry point for the dynamic one-education-cell pre-test."""

import argparse
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads

from config import EST
from model_fitloans_dynamic import (
    CCP_CACHE_MODES,
    DEFAULT_CCP_WORKERS,
    DEFAULT_EDUCATION_CELL_MAXITER,
    DEFAULT_PRIMARY_MOMENT_WEIGHT,
    EDUCATION_CELL_SPECIFICATIONS,
    PARENTAL_INCOME_MOMENT_SPECS,
    TYPE_INTEGRATION_MODES,
    UNCAPPED_EDUCATION_CELL_MAXITER,
    estimate_budget_shock_education_cell,
)
from prepare_fitloans_ccp_sequences import prepare_fitloans_ccp_sequences


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--education", type=int, default=2)
    parser.add_argument("--program-year", type=int, default=1)
    parser.add_argument(
        "--specification",
        choices=EDUCATION_CELL_SPECIFICATIONS,
        default="parental_income_basic",
        help=(
            "parental_income_basic estimates the parinc shock/risk parameters, "
            "optionally debt penalties, and imposes a common shock across latent types."
        ),
    )
    parser.add_argument(
        "--heterogeneity",
        choices=("homogeneous", "mean", "variance", "both"),
        default="homogeneous",
    )
    parser.add_argument(
        "--type-integration",
        choices=TYPE_INTEGRATION_MODES,
        default="sampled",
        help="Draw one persistent posterior joint type, or retain exact integration for validation.",
    )
    parser.add_argument(
        "--moment-spec",
        choices=PARENTAL_INCOME_MOMENT_SPECS,
        default="fast_stock",
        help="fast_stock exactly matches the four model_fitloans_fast moment definitions.",
    )
    parser.add_argument(
        "--primary-moment-weight",
        type=float,
        default=DEFAULT_PRIMARY_MOMENT_WEIGHT,
        help=(
            "Loss weight on mean positive loans and share indebted in every "
            "parinc group; std and p80 each retain weight 1 (default: 4)."
        ),
    )
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Threads used by the fused debt solver; default uses all Numba-available threads.",
    )
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument(
        "--maxiter", type=int, default=DEFAULT_EDUCATION_CELL_MAXITER,
        help="Maximum Nelder-Mead iterations (default: 5000).",
    )
    parser.add_argument(
        "--no-maxiter", action="store_true",
        help="Use an effectively unlimited iteration cap; convergence still stops the optimizer.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--ccp-processes", type=int, default=10)
    parser.add_argument(
        "--ccp-workers",
        type=int,
        default=DEFAULT_CCP_WORKERS,
        help="Processes used to extract the 16 type-specific CCP paths.",
    )
    parser.add_argument(
        "--ccp-cache-mode",
        choices=CCP_CACHE_MODES,
        default="reuse",
        help=(
            "Education-cell test cache: off always reads current sequences; "
            "reuse validates/reuses; rebuild replaces it."
        ),
    )
    parser.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Use only for debugging when every derived input is already verified.",
    )
    parser.add_argument(
        "--fixed-common",
        default=None,
        help="Path to a 16-parameter homogeneous bestx file.",
    )
    parser.add_argument(
        "--initial",
        default=None,
        help="Optional path to a saved bestx array used to restart the optimizer.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive; use --no-maxiter for no practical cap.")
    if not np.isfinite(args.primary_moment_weight) or args.primary_moment_weight <= 0.0:
        raise ValueError("--primary-moment-weight must be positive and finite.")
    if args.numba_threads is not None:
        if args.numba_threads <= 0:
            raise ValueError("--numba-threads must be positive.")
        set_num_threads(args.numba_threads)
    print(f"Numba debt-solver threads: {get_num_threads()}")
    if args.specification == "parental_income_basic" and args.heterogeneity != "homogeneous":
        raise ValueError(
            "--specification parental_income_basic requires --heterogeneity homogeneous."
        )
    if args.specification == "parental_income_basic" and args.fixed_common:
        raise ValueError("--fixed-common is only available with --specification joint_type.")
    if args.specification == "joint_type" and args.type_integration != "exact":
        raise ValueError("--specification joint_type requires --type-integration exact.")
    if not args.skip_preparation:
        print("Checking/building CCP continuation sequences")
        prepare_fitloans_ccp_sequences(processes=args.ccp_processes)

    fixed_common = None
    if args.fixed_common:
        path = Path(args.fixed_common)
        if not path.is_absolute():
            path = Path(EST(path.name))
        fixed_common = np.asarray(np.load(path), dtype=float)

    initial = None
    if args.initial:
        path = Path(args.initial)
        if not path.is_absolute():
            path = Path(EST(path.name))
        initial = np.asarray(np.load(path), dtype=float).reshape(-1)
        print(f"Restarting optimization from {path}")

    maxiter = (
        UNCAPPED_EDUCATION_CELL_MAXITER if args.no_maxiter else args.maxiter
    )

    result, _ = estimate_budget_shock_education_cell(
        education=args.education,
        program_year=args.program_year,
        specification=args.specification,
        type_integration=args.type_integration,
        moment_spec=args.moment_spec,
        primary_moment_weight=args.primary_moment_weight,
        shock_heterogeneity=args.heterogeneity,
        draws=args.draws,
        n_sample=args.n_sample,
        maxiter=maxiter,
        seed=args.seed,
        save=args.save,
        initial=initial,
        fixed_common=fixed_common,
        ccp_workers=args.ccp_workers,
        ccp_cache_mode=args.ccp_cache_mode,
    )
    print("\nOptimization finished")
    print("success:", result.success)
    print("message:", result.message)
    print("objective:", result.fun)
    print("parameters:", result.x)


if __name__ == "__main__":
    main()
