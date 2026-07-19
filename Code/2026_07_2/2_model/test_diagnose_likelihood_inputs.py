import unittest

import numpy as np

from diagnose_likelihood_inputs import analyze_likelihood_cell


class LikelihoodInputDiagnosticTests(unittest.TestCase):
    def test_distinguishes_underflow_infeasibility_and_invalid_values(self):
        vjt = np.asarray(
            [
                [-800.0, -2.0, 0.0],
                [-np.inf, -2.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, np.nan, 0.0],
            ]
        )
        g = np.zeros_like(vjt)
        choices = np.asarray([0, 0, 1, 1])
        weights = np.full(4, 0.25)

        summary, arrays = analyze_likelihood_cell(vjt, g, choices, weights)

        self.assertEqual(summary["numerical_underflow"], 1)
        self.assertEqual(summary["chosen_infeasible"], 1)
        self.assertEqual(summary["chosen_invalid"], 1)
        self.assertEqual(summary["positive_q_with_zero_legacy_probability"], 2)
        self.assertTrue(arrays["numerical_underflow"][0])
        self.assertTrue(np.isfinite(arrays["stable_log_probability"][0]))
        self.assertEqual(arrays["legacy_probability"][0], 0.0)
        self.assertTrue(arrays["chosen_infeasible"][1])
        self.assertTrue(arrays["chosen_invalid"][3])


if __name__ == "__main__":
    unittest.main()
