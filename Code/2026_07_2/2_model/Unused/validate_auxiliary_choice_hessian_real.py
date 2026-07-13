"""Unused read-only real-data validation for the auxiliary exact Hessian."""

import argparse
import time

import numpy as np

import model_em_algorithm as model


def build_real_problem():
    (
        choices_all,
        _vjt_low,
        _vjt_high,
        x1_new,
        choices_array_all,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    ) = model.load_all_arrays_feasible(auxiliar=1)
    data = model.build_auxiliary_em_data(
        choices_all,
        choices_array_all,
        x1_new,
        x_change,
        x_educ,
        x_first2,
        x_first4,
        x_firstgrad,
        x_exp,
    )
    design_start = time.perf_counter()
    model.prepare_auxiliary_choice_design(data)
    design_seconds = time.perf_counter() - design_start
    grant = model.initialize_financial_source(data["grant"])
    transfer = model.initialize_financial_source(data["transfer"])
    consumption = model.build_expected_consumption(
        None, grant, transfer, data, data["total_choices"]
    )
    rng = np.random.default_rng(20260713)
    q = rng.dirichlet(np.ones(4), size=data["n_individuals"])
    return data, consumption, q, rng, design_seconds


def main(full_hessian=False):
    data, consumption, q, rng, design_seconds = build_real_problem()
    evaluator = model.AuxiliaryChoiceNewtonEvaluator(
        data,
        consumption,
        q,
        hessian_workers=4,
    )
    vectors = [
        np.zeros(model.total_n_multi),
        rng.normal(0.0, 0.002, size=model.total_n_multi),
    ]
    validation_start = time.perf_counter()
    reports = model.validate_auxiliary_choice_newton_evaluator(
        evaluator,
        vectors,
        directions=1,
        epsilon=1.0e-5,
        compare_legacy_gradient=True,
    )
    validation_seconds = time.perf_counter() - validation_start
    print("sparse-design build seconds:", design_seconds)
    print("stored sparse nonzeros:", sum(
        p["choice_design_base"].nnz + p["choice_design_school_type"].nnz
        for p in data["periods"]
    ))
    print("validation seconds:", validation_seconds)
    for index, report in enumerate(reports):
        print(f"parameterization {index}:", report)

    timing_parameters = vectors[-1] + 1.0e-8
    score_start = time.perf_counter()
    evaluator.fun_and_jac(timing_parameters)
    score_seconds = time.perf_counter() - score_start
    direction = rng.normal(size=model.total_n_multi)
    direction /= np.linalg.norm(direction)
    hessp_start = time.perf_counter()
    evaluator.hessp(timing_parameters, direction)
    hessp_seconds = time.perf_counter() - hessp_start
    print("one likelihood-plus-gradient seconds:", score_seconds)
    print("one exact Hessian-vector product seconds:", hessp_seconds)

    if full_hessian:
        hessian_start = time.perf_counter()
        hessian = evaluator.hess(vectors[-1])
        hessian_seconds = time.perf_counter() - hessian_start
        print("full Hessian seconds:", hessian_seconds)
        print("full Hessian symmetry error:", np.max(np.abs(hessian - hessian.T)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hessian", action="store_true")
    arguments = parser.parse_args()
    main(full_hessian=arguments.full_hessian)
