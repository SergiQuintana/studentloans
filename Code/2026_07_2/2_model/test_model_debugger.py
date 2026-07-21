"""Fast tests for debug classification and production-path isolation."""

import ast
import hashlib
from pathlib import Path

import numpy as np

from model_debug_checks import ccp_problems, finite_problems, first_bad_index, vjt_problems


HERE = Path(__file__).resolve().parent


PRODUCTION_AST_HASHES = {
    ("model_predict_ccps.py", "get_all_ccps"):
        "bbef00aaf16110196e927576a109b513ab056190140d85afa4ad495e72e430db",
    ("model_solution_em.py", "get_expected_conditional"):
        "6f4a253f5a3ffd21ceb6fa50d021996a5b26144884e56daa903b006c8cad6d21",
    ("model_solution_em.py", "get_all_choices"):
        "26a2e63d8ed6d4c03de99b8b3d543df829c5a070539bebd00bbba4b2ffd7d828",
    ("model_solution_em.py", "loop_rows"):
        "a2f19b953ed9b30b6e81efe4f12a578174ea668abcbaee6ed1c94f8c69051564",
    ("model_solution_em.py", "loop_over_states"):
        "00ecaa15c9a5d436eafa0c1545661f77d743fb84f6a9c9821c29db4e8b01f96e",
    ("model_solution_em.py", "get_all_evt"):
        "76ceef6be30371149f42d2e4e2e7893977a5997129e8be9ae38f9ab655ae0e63",
}


def _function_hash(filename, function_name):
    tree = ast.parse((HERE / filename).read_text(encoding="utf-8-sig"))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    normalized = ast.dump(node, include_attributes=False).encode()
    return hashlib.sha256(normalized).hexdigest()


def test_production_function_bodies_are_unchanged():
    for key, expected in PRODUCTION_AST_HASHES.items():
        assert _function_hash(*key) == expected


def test_vjt_allows_negative_infinity_but_not_nan_or_positive_infinity():
    valid = np.array([[0.0, -np.inf], [-2.0, -3.0]])
    assert vjt_problems(valid) == []
    reasons = {item["reason"] for item in vjt_problems([[np.nan, np.inf]])}
    assert reasons == {"vjt_nan", "vjt_positive_inf", "no_finite_choice"}


def test_vjt_rejects_rows_without_any_finite_choice():
    problems = vjt_problems(np.array([[0.0, -np.inf], [-np.inf, -np.inf]]))
    assert problems == [{"reason": "no_finite_choice", "index": (1,)}]


def test_ccp_requires_finite_strict_probability():
    assert ccp_problems(np.array([0.1, 1.0])) == []
    reasons = {item["reason"] for item in ccp_problems([0.0, np.nan, 1.1])}
    assert reasons == {
        "ccp_nonfinite", "ccp_not_strictly_positive", "ccp_above_one"
    }


def test_finite_check_labels_the_array():
    assert finite_problems([1.0], "evt") == []
    assert finite_problems([np.inf], "evt") == [
        {"reason": "evt_nonfinite", "index": (0,)}
    ]


def test_scalar_problem_is_detected():
    assert first_bad_index(np.asarray(True)) == ()
    assert finite_problems(np.asarray(np.nan), "risk_aversion") == [
        {"reason": "risk_aversion_nonfinite", "index": ()}
    ]
