"""Organize saved education-cell estimates and reproduce their model-fit table.

This program never estimates parameters and never overwrites estimate arrays.
By default it evaluates the SMM objective once at the saved parameter vector,
using the requested common-random-number settings, and writes everything shown
in the terminal to a durable text report under Model/Output/fitloans_reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import budget_shock as bs
from config import EST, OUT


MEAN_LABELS = (
    "mean intercept (parinc=1, ability=1)",
    "mean deviation: parinc=2",
    "mean deviation: parinc=3",
    "mean deviation: parinc=4",
    "mean deviation: ability=2",
    "mean deviation: ability=3",
    "mean deviation: ability=4",
)


class Tee:
    """Write identical output to the terminal and a report file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def result_prefix(
    education, program_year, heterogeneity, specification,
    type_integration="sampled", moment_spec="flow_plus_stock", resource_mode="simulated",
):
    if specification in ("parental_income_basic", "parental_income_loan_type"):
        return (
            f"budgetshock_educ{education}_year{program_year}_{specification}_"
            f"{type_integration}_{moment_spec}"
            f"{'_observed_resources' if resource_mode == 'observed' else ''}"
        )
    return f"budgetshock_educ{education}_year{program_year}_{heterogeneity}"


def load_saved_bundle(prefix, heterogeneity, education, program_year, specification):
    paths = {
        "bestx": Path(EST(f"{prefix}_bestx.npy")),
        "params": Path(EST(f"{prefix}_params.npy")),
        "risk_aversion": Path(EST(f"{prefix}_risk_aversion.npy")),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing saved estimate arrays:\n  " + "\n  ".join(missing))

    bestx = np.asarray(np.load(paths["bestx"]), dtype=np.float64).reshape(-1)
    saved_spec = bs.validate(np.load(paths["params"], allow_pickle=True).item())
    separate_risk = np.asarray(np.load(paths["risk_aversion"]), dtype=np.float64)
    cell_code = bs.budget_education_cell_code(education, program_year)
    if specification == "parental_income_basic":
        vector_spec = bs.unpack_parental_income_estimation_vector(
            bestx, [cell_code], index_kind="education_cell"
        )
    elif specification == "parental_income_loan_type":
        vector_spec = bs.unpack_parental_income_loan_type_estimation_vector(
            bestx, [cell_code], index_kind="education_cell"
        )
    else:
        vector_spec = bs.unpack_estimation_vector(
            bestx, [cell_code], loan_heterogeneity=heterogeneity,
            index_kind="education_cell",
        )

    fields = (
        "periods", "mu_blocks", "sigma_e", "risk_aversion",
        "debt_pen_parinc", "loan_mean_shift", "loan_log_sigma_ratio",
        "budget_resource_slope",
    )
    for field in fields:
        if not np.allclose(saved_spec[field], vector_spec[field], equal_nan=True):
            raise ValueError(f"{paths['params']} disagrees with bestx for {field}.")
    if specification == "parental_income_loan_type":
        if not np.allclose(
            saved_spec["risk_aversion_by_loan_type"],
            vector_spec["risk_aversion_by_loan_type"],
        ):
            raise ValueError(
                f"{paths['params']} disagrees with bestx for loan-type risk aversion."
            )
    if not np.allclose(separate_risk, saved_spec["risk_aversion"]):
        raise ValueError("The separate risk-aversion array disagrees with the parameter bundle.")
    return paths, bestx, saved_spec


def print_parameter_report(paths, bestx, spec):
    print("=" * 78)
    print("SAVED EDUCATION-CELL LOAN ESTIMATE")
    print("=" * 78)
    print(f"bestx array:          {paths['bestx']}")
    print(f"parameter bundle:     {paths['params']}")
    print(f"risk-aversion array:  {paths['risk_aversion']}")
    print(f"schema version:       {spec['schema_version']}")
    print(f"index kind:           {spec['index_kind']}")
    print(f"education-cell code:  {int(spec['periods'][0])}")
    print(f"loan heterogeneity:   {spec['loan_heterogeneity']}")
    print(f"estimation spec:      {spec.get('estimation_parameterization', 'joint_type')}")
    if "smm_type_integration" in spec:
        print(f"type integration:     {spec['smm_type_integration']}")
        print(f"moment specification: {spec['smm_moment_spec']}")
        print(
            "primary moment weight: "
            f"{spec.get('smm_primary_moment_weight', 1.0)}"
        )
        print(f"current resources:    {spec.get('smm_resource_mode', 'simulated')}")
        print(f"estimation draws:     {spec['smm_draws']}")
        print(f"estimation seed:      {spec['smm_seed']}")
        print(f"estimation n_sample:  {spec['smm_n_sample']}")
    print(f"raw vector length:    {bestx.size}")
    print(
        "budget-resource slope: "
        f"{float(spec['budget_resource_slope'][0]):.6f} dollars of shock mean "
        "per $10,000 of pre-choice resources"
    )

    print("\nBudget-shock conditional-mean coefficients")
    for label, value in zip(MEAN_LABELS, spec["mu_blocks"][0]):
        print(f"  {label:<43} {value:>14.6f}")

    sigma_low = float(spec["sigma_e"][0])
    mean_shift = float(spec["loan_mean_shift"][0])
    sigma_ratio = float(np.exp(spec["loan_log_sigma_ratio"][0]))
    print("\nLoan-type shock differences")
    print(f"  low-loan type mean shift                   {0.0:>14.6f}")
    print(f"  high-loan type mean shift                  {mean_shift:>14.6f}")
    print(f"  low-loan type sigma                        {sigma_low:>14.6f}")
    print(f"  high-loan type sigma                       {sigma_low * sigma_ratio:>14.6f}")
    print(f"  high/low sigma ratio                       {sigma_ratio:>14.6f}")

    if "risk_aversion_by_loan_type" in spec:
        print("\nRisk-aversion levels by loan type and model parinc=x1[:,0]")
        for loan_type, type_name in ((0, "low"), (1, "high")):
            for level, value in enumerate(
                spec["risk_aversion_by_loan_type"][loan_type], start=1
            ):
                print(
                    f"  {type_name}-loan, parinc={level}"
                    f"                         {value:>14.6f}"
                )
    else:
        print("\nRisk-aversion levels by model parinc=x1[:,0]")
        for level, value in enumerate(spec["risk_aversion"], start=1):
            print(f"  parinc={level}                              {value:>14.6f}")

    coefficients = np.asarray(spec["debt_pen_parinc"], dtype=np.float64)
    implied = coefficients[0] + np.r_[0.0, coefficients[1:]]
    print("\nDebt-penalty parameterization: baseline plus parinc deviations")
    print(f"  baseline coefficient (parinc=1)            {coefficients[0]:>14.6f}")
    for level in range(2, 5):
        print(
            f"  deviation coefficient (parinc={level})           "
            f"{coefficients[level - 1]:>14.6f}"
        )
    print("  Implied debt penalties:")
    for level, value in enumerate(implied, start=1):
        print(f"    parinc={level}                            {value:>14.6f}")

    print("\nRaw bestx order")
    if spec.get("estimation_parameterization") in (
        "parental_income_basic", "parental_income_loan_type",
    ):
        raw_labels = [f"shock mean level: parinc={level}" for level in range(1, 5)]
        raw_labels += ["common shock sigma"]
        raw_labels += [f"risk aversion: parinc={level}" for level in range(1, 5)]
        raw_labels += [f"debt penalty level: parinc={level}" for level in range(1, 5)]
        if bestx.size > bs.LEGACY_PARENTAL_INCOME_ESTIMATION_VECTOR_SIZE:
            raw_labels += ["common pre-choice-resource slope (per $10,000)"]
        if spec.get("estimation_parameterization") == "parental_income_loan_type":
            raw_labels += [
                f"risk aversion: high-loan type, parinc={level}"
                for level in range(1, 5)
            ]
    else:
        raw_labels = list(MEAN_LABELS)
        raw_labels += ["sigma: low-loan type"]
        raw_labels += [f"risk aversion: parinc={level}" for level in range(1, 5)]
        raw_labels += ["debt penalty: baseline"]
        raw_labels += [f"debt penalty deviation: parinc={level}" for level in range(2, 5)]
        if spec["loan_heterogeneity"] in ("mean", "both"):
            raw_labels += ["high-loan type mean shift"]
        if spec["loan_heterogeneity"] in ("variance", "both"):
            raw_labels += ["log(high/low sigma ratio)"]
    for index, (label, value) in enumerate(zip(raw_labels, bestx)):
        print(f"  [{index:>2}] {label:<43} {value:>14.6f}")


def reevaluate_model_fit(bestx, args):
    # Importing the dynamic model is deliberately delayed: printing the saved
    # parameter organization alone does not need to load the model machinery.
    import model_fitloans_dynamic as fit

    print("\n" + "=" * 78)
    print("MODEL FIT REEVALUATED ONCE AT SAVED PARAMETERS")
    print("=" * 78)
    print(f"draws:       {args.draws}")
    print(f"seed:        {args.seed}")
    print(f"n_sample:    {args.n_sample if args.n_sample is not None else 'all'}")
    print(f"ccp workers: {args.ccp_workers}")
    print(f"ccp cache:   {args.ccp_cache_mode}")
    print(f"primary moment weight: {args.primary_moment_weight:g}")
    print(f"current resources: {args.resource_mode}")
    print(
        "This reproduces the original reported fit exactly only when these "
        "settings equal those used in estimation."
    )

    interp_dict = fit.get_interp_dict_cached(force_rebuild=False)
    packs = []
    for period in range(1, fit.T):
        print(f"[load education cell] model period={period}")
        pack = fit.load_education_cell(
            period,
            interp_dict,
            education=args.education,
            program_year=args.program_year,
            ccp_workers=args.ccp_workers,
            ccp_cache_mode=args.ccp_cache_mode,
        )
        if len(pack["x1"]):
            print(f"  retained {len(pack['x1'])} enrolled observations")
            packs.append(pack)
    if not packs:
        raise ValueError("No observations were found for the requested education cell.")

    data_moments, data_new_share, data_weights, labels = (
        fit._pooled_observed_cell_moments(
            packs, specification=args.specification, moment_spec=args.moment_spec
        )
    )
    integrated_data = None
    if args.specification == "parental_income_loan_type":
        integrated_data = fit._pooled_observed_cell_moments(
            packs, specification="parental_income_basic",
            moment_spec=args.moment_spec,
        )
    if args.specification in (
        "parental_income_basic", "parental_income_loan_type",
    ) and args.type_integration == "sampled":
        packs = fit.assign_persistent_joint_types(packs, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    sampled = []
    for pack in packs:
        if args.n_sample is None or len(pack["x1"]) <= args.n_sample:
            sampled.append(pack)
        else:
            indices = rng.choice(len(pack["x1"]), args.n_sample, replace=True)
            sampled.append(fit._subset_cell_pack(pack, indices))
    sample_by_period = fit.prepare_education_cell_crns(
        sampled, draws=args.draws, seed=args.seed,
        type_integration=args.type_integration,
        resource_mode=args.resource_mode,
    )
    cell_code = bs.budget_education_cell_code(
        args.education, args.program_year
    )

    # The objective prints its detailed table every tenth call. This is a
    # reporting-only process, so make the single evaluation the tenth call.
    fit.EVAL_COUNTER = 9
    objective = (
        fit.minimize_distance_education_cell_parental_income
        if args.specification in (
            "parental_income_basic", "parental_income_loan_type",
        )
        else fit.minimize_distance_education_cell
    )
    objective_args = (
        data_moments, data_new_share, data_weights, labels, sample_by_period,
        cell_code, args.education, args.program_year,
    )
    if args.specification == "joint_type":
        objective_args += (args.heterogeneity,)
    else:
        objective_args += (
            args.type_integration, args.moment_spec, args.primary_moment_weight,
            args.specification, integrated_data,
        )
    loss = objective(bestx, *objective_args)
    print(f"Reevaluated SMM loss: {loss:.10f}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--education", type=int, default=2)
    parser.add_argument("--program-year", type=int, default=1)
    parser.add_argument(
        "--specification",
        choices=("parental_income_basic", "parental_income_loan_type", "joint_type"),
        default="parental_income_basic",
    )
    parser.add_argument(
        "--heterogeneity",
        choices=bs.LOAN_HETEROGENEITY_MODES,
        default="homogeneous",
    )
    parser.add_argument(
        "--type-integration", choices=("sampled", "exact"), default="sampled"
    )
    parser.add_argument(
        "--moment-spec",
        choices=("fast_stock", "flow_stock", "fast_flow", "flow_plus_stock"),
        default="flow_plus_stock"
    )
    parser.add_argument(
        "--primary-moment-weight",
        type=float,
        default=None,
        help="Override the saved primary-moment loss weight when reevaluating.",
    )
    parser.add_argument(
        "--resource-mode",
        choices=("simulated", "observed"),
        default="simulated",
    )
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ccp-workers", type=int, default=16)
    parser.add_argument(
        "--ccp-cache-mode",
        choices=("off", "reuse", "rebuild"),
        default="reuse",
    )
    parser.add_argument(
        "--parameters-only",
        action="store_true",
        help="Organize saved arrays without loading data or reevaluating model fit.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional report path. The default is under Model/Output/fitloans_reports.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.program_year = int(
        bs.budget_program_year(args.education, args.program_year)
    )
    if args.specification in (
        "parental_income_basic", "parental_income_loan_type",
    ) and args.heterogeneity != "homogeneous":
        raise ValueError(
            "Parental-income specifications require --heterogeneity homogeneous."
        )
    if args.specification == "joint_type" and args.type_integration != "exact":
        raise ValueError("--specification joint_type requires --type-integration exact.")
    prefix = result_prefix(
        args.education, args.program_year, args.heterogeneity, args.specification,
        args.type_integration, args.moment_spec, args.resource_mode,
    )
    report_path = Path(
        args.report_path
        or OUT("fitloans_reports", f"{prefix}_report.txt")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    terminal = sys.stdout
    with report_path.open("w", encoding="utf-8", buffering=1) as report:
        sys.stdout = Tee(terminal, report)
        try:
            paths, bestx, spec = load_saved_bundle(
                prefix, args.heterogeneity, args.education, args.program_year,
                args.specification,
            )
            if args.primary_moment_weight is None:
                args.primary_moment_weight = float(
                    spec.get("smm_primary_moment_weight", 1.0)
                )
            if not np.isfinite(args.primary_moment_weight) or args.primary_moment_weight <= 0:
                raise ValueError("--primary-moment-weight must be positive and finite.")
            print_parameter_report(paths, bestx, spec)
            if not args.parameters_only:
                reevaluate_model_fit(bestx, args)
            print(f"\nDurable report: {report_path}")
        finally:
            sys.stdout = terminal

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
