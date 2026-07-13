"""Deterministic checks for the auxiliary multinomial-logit Hessian."""

import numpy as np

import model_em_algorithm as model


def synthetic_auxiliary_problem(n=6, seed=20260713):
    rng = np.random.default_rng(seed)
    choices = model.get_total_choices().astype(int)
    j = len(choices)
    periods = []
    consumption_low = []
    consumption_high = []

    for period in range(1, model.T):
        x_change = np.zeros((n, 1 + model.fields + model.occupations + 1))
        x_change[np.arange(n), rng.integers(0, x_change.shape[1], size=n)] = 1.0
        period_data = {
            "period": period,
            "chosen_index": rng.integers(0, j, size=n),
            "x1": rng.normal(size=(n, 9)),
            "feasible": np.ones((n, j), dtype=bool),
            "debt_dollars": rng.uniform(0.0, 30000.0, size=n),
            "x_change": x_change,
            "x_educ": rng.normal(size=(n, 9)),
            "x_first2": rng.normal(size=(n, 1)),
            "x_first4": rng.normal(size=(n, 4)),
            # Graduate enrollment cannot occur before period 6; the legacy
            # score relies on the corresponding cached regressor being zero.
            "x_firstgrad": (
                rng.normal(size=(n, 1)) if period > 5 else np.zeros((n, 1))
            ),
            "x_exp": rng.normal(size=(n, 24)),
        }
        periods.append(period_data)
        low = rng.normal(20000.0, 5000.0, size=(n, j))
        high = low + rng.normal(3000.0, 500.0, size=(n, j))
        low[:, -1] = 0.0
        high[:, -1] = 0.0
        consumption_low.append(low)
        consumption_high.append(high)

    data = {
        "n_individuals": n,
        "total_choices": choices,
        "periods": periods,
        "nonhome": np.any(choices != 0, axis=1).astype(float),
    }
    model.prepare_auxiliary_choice_design(data)
    q = rng.dirichlet(np.ones(4), size=n)
    return data, (consumption_low, consumption_high), q, rng


def run_checks():
    data, expected_consumption, q, rng = synthetic_auxiliary_problem()
    evaluator = model.AuxiliaryChoiceNewtonEvaluator(
        data,
        expected_consumption,
        q,
        hessian_workers=2,
        hessian_block_size=32,
    )
    parameter_vectors = [
        np.zeros(model.total_n_multi),
        rng.normal(0.0, 0.01, size=model.total_n_multi),
        rng.normal(0.0, 0.05, size=model.total_n_multi),
    ]
    reports = model.validate_auxiliary_choice_newton_evaluator(
        evaluator,
        parameter_vectors,
        directions=2,
        epsilon=1.0e-5,
        compare_legacy_gradient=True,
    )
    for report in reports:
        assert abs(report["objective_difference"]) < 1.0e-8, report
        assert report["legacy_gradient_max_difference"] < 2.0e-8, report
        assert np.max(report["hessp_relative_errors"]) < 2.0e-6, report

    parameters = parameter_vectors[-1]
    full_hessian = evaluator.hess(parameters)
    symmetry_error = np.max(np.abs(full_hessian - full_hessian.T))
    direction = rng.normal(size=model.total_n_multi)
    direction /= np.linalg.norm(direction)
    full_product = full_hessian @ direction
    direct_product = evaluator.hessp(parameters, direction)
    product_scale = max(1.0, np.max(np.abs(direct_product)))
    product_error = np.max(np.abs(full_product - direct_product)) / product_scale
    assert symmetry_error < 1.0e-10, symmetry_error
    assert product_error < 2.0e-10, product_error
    return reports, symmetry_error, product_error


if __name__ == "__main__":
    check_reports, hessian_symmetry_error, hessian_product_error = run_checks()
    for index, report in enumerate(check_reports):
        print(f"parameterization {index}: {report}")
    print("full Hessian symmetry error:", hessian_symmetry_error)
    print("full Hessian/Hv product relative error:", hessian_product_error)
