import unittest

import numpy as np
import pandas as pd

import tables_types as tt


def _posterior(high_probabilities):
    q = np.zeros((len(high_probabilities), 16), dtype=float)
    q[:, 0] = 1.0 - np.asarray(high_probabilities)
    q[:, 1] = np.asarray(high_probabilities)
    return q


class LoanTypeDescriptiveTests(unittest.TestCase):
    def test_posterior_summary_detects_separation_and_ambiguity(self):
        high = np.array([0.0, 0.5, 0.9, 1.0])
        q = _posterior(high)
        panel = pd.DataFrame(
            {"_em_row": np.arange(4), "PUBID": np.arange(101, 105)}
        )
        summary, histogram, individual = tt._loan_type_posterior_tables(
            panel, q, np.tile([0, 1], 8)
        )

        row = summary.iloc[0]
        self.assertAlmostEqual(row["posterior_expected_L1_share"], 0.6)
        self.assertAlmostEqual(
            row["share_L1_posterior_between_0_40_and_0_60"], 0.25
        )
        self.assertAlmostEqual(row["share_modal_probability_above_0_90"], 0.75)
        self.assertEqual(int(histogram["individuals"].sum()), 4)
        self.assertEqual(individual.loc[1, "modal_loan_type"], "L1")
        self.assertAlmostEqual(individual.loc[1, "normalized_binary_entropy"], 1.0)

    def test_persistence_uses_consecutive_enrolled_transitions(self):
        panel = pd.DataFrame(
            {
                "_em_row": np.repeat([0, 1], 3),
                "PUBID": np.repeat([10, 20], 3),
                "period": np.tile([1, 2, 3], 2),
                "educ": 2,
                "auxiliary_loan_flow": [100.0, 200.0, 0.0, 0.0, 300.0, 400.0],
            }
        )
        q = _posterior([0.0, 1.0])
        groups = tt._loan_type_groups(q, np.tile([0, 1], 8))
        people, transitions = tt._loan_persistence_by_type(panel, groups)

        low = transitions.loc[transitions["loan_type"].eq("L0")].iloc[0]
        high = transitions.loc[transitions["loan_type"].eq("L1")].iloc[0]
        self.assertAlmostEqual(
            low["continuation_probability_P_borrow_t_given_borrow_tminus1"], 0.5
        )
        self.assertAlmostEqual(
            high["continuation_probability_P_borrow_t_given_borrow_tminus1"], 1.0
        )
        self.assertAlmostEqual(
            high["entry_probability_P_borrow_t_given_no_borrow_tminus1"], 1.0
        )
        self.assertTrue(np.isnan(low["entry_probability_P_borrow_t_given_no_borrow_tminus1"]))
        self.assertTrue(np.allclose(people["share_borrowed_in_multiple_years"], 1.0))


if __name__ == "__main__":
    unittest.main()
