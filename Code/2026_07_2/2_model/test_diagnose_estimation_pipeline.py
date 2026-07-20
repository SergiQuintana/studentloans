import importlib
import sys
import types
import unittest

import numpy as np


def import_pipeline_with_lightweight_model_stubs():
    """Import pure diagnostic helpers without loading the full Numba model."""
    stubs = {
        "diagnose_likelihood_inputs": types.SimpleNamespace(),
        "model_predict_ccps": types.SimpleNamespace(
            INITIAL_CCP_PERIODS=tuple(range(9, 0, -1)),
            MONEY_SCALE=1000.0,
        ),
        "model_solution_em": types.SimpleNamespace(
            invariant_states=np.zeros((64, 4), dtype=int),
        ),
        "config": types.SimpleNamespace(
            EST=lambda *parts: "/tmp/" + "/".join(parts),
            LIK=lambda *parts: "/tmp/" + "/".join(parts),
            OUT=lambda *parts: "/tmp/" + "/".join(parts),
        ),
        "latent_types": types.SimpleNamespace(
            TYPE_IDS=tuple(range(1, 17)),
            TYPE_NAMES=tuple(f"type_{index}" for index in range(1, 17)),
            load_em_posteriors=lambda path: np.ones((1, 16)) / 16.0,
            type_components=lambda type_id: (0, 0, 0, 0),
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        sys.modules.pop("diagnose_estimation_pipeline", None)
        return importlib.import_module("diagnose_estimation_pipeline")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


pipeline = import_pipeline_with_lightweight_model_stubs()


class EstimationPipelineDiagnosticTests(unittest.TestCase):
    def test_integer_selection_accepts_ranges_and_preserves_order(self):
        selected = pipeline.parse_integer_selection(
            "3,1-2,3", range(5), "indices"
        )
        self.assertEqual(selected, (3, 1, 2))

    def test_initial_failure_classifies_underflow_and_stale_mismatch(self):
        g = np.zeros(2)
        expected_consumption = np.zeros(2)
        fresh_vjt = np.asarray([[0.0, -800.0], [0.0, -1.0]])
        log_denom = np.asarray([0.0, 0.1])
        home_log_ccp = np.asarray([-800.0, -1.1])
        fresh_ccp = np.exp(home_log_ccp)
        stored_ccp = fresh_ccp.copy()
        stored_ccp[1] = np.nan
        mismatch = np.asarray([False, True])

        reason = pipeline._failure_reason_initial(
            g,
            expected_consumption,
            fresh_vjt,
            log_denom,
            home_log_ccp,
            fresh_ccp,
            stored_ccp,
            mismatch,
        )

        self.assertIn("fresh_probability_underflow", reason)
        self.assertIn("stored_ccp_nonfinite", reason)
        self.assertIn("stored_fresh_mismatch", reason)
        self.assertNotIn("fresh_vjt_nonfinite", reason)

    def test_failure_order_follows_backward_recursion(self):
        failures = [
            {"period": 6, "type_id": 1, "x1_index": 0, "x2_index": 0, "debt_index": 0},
            {"period": 9, "type_id": 2, "x1_index": 0, "x2_index": 0, "debt_index": 0},
            {"period": 9, "type_id": 1, "x1_index": 1, "x2_index": 0, "debt_index": 0},
        ]
        failures.sort(key=pipeline.initial_failure_sort_key)
        self.assertEqual(
            [(row["period"], row["type_id"]) for row in failures],
            [(9, 1), (9, 2), (6, 1)],
        )


if __name__ == "__main__":
    unittest.main()
