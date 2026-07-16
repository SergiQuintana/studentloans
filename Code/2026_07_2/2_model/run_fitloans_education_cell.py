"""Command-line entry point for the dynamic one-education-cell pre-test."""

import argparse
from pathlib import Path

import numpy as np

from config import EST
from model_fitloans_dynamic import (
    CCP_CACHE_MODES,
    DEFAULT_CCP_WORKERS,
    EDUCATION_CELL_SPECIFICATIONS,
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
            "parental_income_basic estimates 13 parinc-only parameters and "
            "imposes a common budget shock across latent types."
        ),
    )
    parser.add_argument(
        "--heterogeneity",
        choices=("homogeneous", "mean", "variance", "both"),
        default="homogeneous",
    )
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=500)
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
    return parser


def main():
    args = build_parser().parse_args()
    if args.specification == "parental_income_basic" and args.heterogeneity != "homogeneous":
        raise ValueError(
            "--specification parental_income_basic requires --heterogeneity homogeneous."
        )
    if args.specification == "parental_income_basic" and args.fixed_common:
        raise ValueError("--fixed-common is only available with --specification joint_type.")
    if not args.skip_preparation:
        print("Checking/building CCP continuation sequences")
        prepare_fitloans_ccp_sequences(processes=args.ccp_processes)

    fixed_common = None
    if args.fixed_common:
        path = Path(args.fixed_common)
        if not path.is_absolute():
            path = Path(EST(path.name))
        fixed_common = np.asarray(np.load(path), dtype=float)

    result, _ = estimate_budget_shock_education_cell(
        education=args.education,
        program_year=args.program_year,
        specification=args.specification,
        shock_heterogeneity=args.heterogeneity,
        draws=args.draws,
        n_sample=args.n_sample,
        maxiter=args.maxiter,
        seed=args.seed,
        save=args.save,
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
