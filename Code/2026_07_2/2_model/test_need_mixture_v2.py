"""Tests for the need_mixture_v2 moment specification.

Guards three contracts: (1) need_mixture_v1 output is unchanged everywhere the
two specifications share code; (2) the reduced graduation block is exactly the
share-indebted and mean-positive subset of v1; (3) the pooled loan-type block
implements the posterior-weighting methodology — observed outcomes weighted by
the marginal posterior P(loan type), simulated outcomes hard-assigned to the
sampled type — with the debt-status restriction sets and the documented
ordering, weights, and scale floors.
"""

import unittest

import numpy as np

import model_fitloans_dynamic as m
from latent_types import N_TYPES, TYPE_LOAN


def _synthetic_panel_contexts(seed=0):
    """Two tiny cells x two periods with every key the block builders read."""
    rng = np.random.default_rng(seed)
    contexts = []
    individual_offset = 0
    for cell_index, education in enumerate((2, 2)):
        sample_by_period = {}
        for period in (1, 2):
            n = 12
            individual_index = individual_offset + np.arange(n, dtype=np.int64)
            q = rng.dirichlet(np.ones(N_TYPES), size=n)
            pack = {
                "x1": np.column_stack((
                    rng.integers(1, 5, size=n), rng.integers(1, 5, size=n),
                )).astype(np.int64),
                "individual_index": individual_index,
                "period": period,
                "parinc": rng.integers(1, 5, size=n).astype(np.int64),
                "loan_flow": np.where(
                    rng.random(n) < 0.5, rng.uniform(500.0, 9000.0, size=n), 0.0
                ),
                "debt": np.where(
                    rng.random(n) < 0.5, rng.uniform(1000.0, 20000.0, size=n), 0.0
                ),
                "q": q,
                "sampled_loan_type": rng.integers(0, 2, size=n).astype(np.int64),
            }
            sample_by_period[period] = pack
        contexts.append({
            "education": education,
            "sample_by_period": sample_by_period,
        })
        individual_offset += 100
    return contexts


def _column_arrays(contexts, key):
    return np.concatenate([
        np.asarray(pack[key], dtype=np.float64)
        for context in contexts
        for pack in context["sample_by_period"].values()
    ])


class SpecMembershipTests(unittest.TestCase):
    def test_both_specifications_are_registered(self):
        self.assertIn(
            m.NEED_MIXTURE_MOMENT_SPEC,
            m.MULTICELL_PARENTAL_INCOME_MOMENT_SPECS,
        )
        self.assertIn(
            m.NEED_MIXTURE_V2_MOMENT_SPEC,
            m.MULTICELL_PARENTAL_INCOME_MOMENT_SPECS,
        )
        self.assertEqual(
            m.NEED_MIXTURE_MOMENT_SPECS,
            (m.NEED_MIXTURE_MOMENT_SPEC, m.NEED_MIXTURE_V2_MOMENT_SPEC),
        )

    def test_cell_weight_pattern_is_shared_between_v1_and_v2(self):
        np.testing.assert_array_equal(
            m.parental_income_moment_weight_pattern(
                m.NEED_MIXTURE_MOMENT_SPEC, 4.0
            ),
            m.parental_income_moment_weight_pattern(
                m.NEED_MIXTURE_V2_MOMENT_SPEC, 4.0
            ),
        )


class GraduationBlockTests(unittest.TestCase):
    def _panel_and_flow(self):
        contexts = _synthetic_panel_contexts()
        panel = m.build_graduation_panel(contexts)
        return panel, panel["observed_flow"]

    def test_v1_output_is_unchanged_by_the_new_keyword(self):
        panel, flow = self._panel_and_flow()
        np.testing.assert_array_equal(
            m.graduation_block_moments(panel, flow),
            m.graduation_block_moments(
                panel, flow, moment_spec=m.NEED_MIXTURE_MOMENT_SPEC
            ),
        )

    def test_v2_is_the_share_and_mean_subset_of_v1(self):
        panel, flow = self._panel_and_flow()
        v1 = m.graduation_block_moments(panel, flow)
        v2 = m.graduation_block_moments(
            panel, flow, moment_spec=m.NEED_MIXTURE_V2_MOMENT_SPEC
        )
        self.assertEqual(v1.size, 18)
        self.assertEqual(v2.size, 8)
        np.testing.assert_array_equal(v2[0::2], v1[0:16:4])
        np.testing.assert_array_equal(v2[1::2], v1[1:16:4])

    def test_v2_loss_uses_the_reduced_weights_and_floors(self):
        data = np.asarray([0.5, 15000.0, 0.6, 18000.0,
                           0.7, 20000.0, 0.01, 100.0])
        simulated = data + np.asarray([0.1, 1000.0, -0.05, -500.0,
                                       0.0, 0.0, 0.01, 50.0])
        loss, residuals = m.graduation_block_loss_and_residuals(
            simulated, data, moment_spec=m.NEED_MIXTURE_V2_MOMENT_SPEC
        )
        weights = np.tile(m.GRADUATION_V2_MOMENT_WEIGHTS, 4)
        floors = np.tile(m.GRADUATION_V2_SCALE_FLOORS, 4)
        scale = np.maximum(np.abs(data), floors)
        expected = np.sum(weights * ((simulated - data) / scale) ** 2)
        self.assertAlmostEqual(loss, expected, places=12)
        self.assertAlmostEqual(loss, float(np.sum(residuals ** 2)), places=12)
        # The near-zero share and the tiny dollar mean must hit their floors.
        self.assertEqual(scale[6], m.GRADUATION_V2_SCALE_FLOORS[0])
        self.assertEqual(scale[7], m.GRADUATION_V2_SCALE_FLOORS[1])

    def test_v2_loss_rejects_the_v1_moment_count(self):
        with self.assertRaises(ValueError):
            m.graduation_block_loss_and_residuals(
                np.zeros(18), np.zeros(18),
                moment_spec=m.NEED_MIXTURE_V2_MOMENT_SPEC,
            )

    def test_unknown_specification_is_rejected(self):
        panel, flow = self._panel_and_flow()
        with self.assertRaises(ValueError):
            m.graduation_block_moments(panel, flow, moment_spec="fast_stock")


class LoanTypeBlockTests(unittest.TestCase):
    def test_block_columns_follow_the_pooled_order(self):
        contexts = _synthetic_panel_contexts()
        block = m.build_loan_type_block(contexts)
        np.testing.assert_array_equal(
            block["begin_debt"], _column_arrays(contexts, "debt")
        )
        np.testing.assert_array_equal(
            block["observed_flow"], _column_arrays(contexts, "loan_flow")
        )
        # Weight rows: posterior loan-margin masses on the data side, sampled
        # indicators on the simulated side; both columns sum to one.
        high = np.concatenate([
            np.asarray(pack["q"], dtype=np.float64)[
                :, np.flatnonzero(TYPE_LOAN == 1)
            ].sum(axis=1)
            for context in contexts
            for pack in context["sample_by_period"].values()
        ])
        np.testing.assert_allclose(block["data_weights"][1], high, atol=1e-12)
        np.testing.assert_allclose(
            block["data_weights"].sum(axis=0),
            np.ones(high.size), atol=1e-12,
        )
        sampled = _column_arrays(contexts, "sampled_loan_type")
        np.testing.assert_array_equal(block["sim_weights"][1], sampled)
        np.testing.assert_array_equal(block["sim_weights"][0], 1.0 - sampled)

    def test_posterior_weighted_data_moments_match_manual_computation(self):
        contexts = _synthetic_panel_contexts()
        block = m.build_loan_type_block(contexts)
        moments = m.loan_type_block_moments(
            block["begin_debt"], block["observed_flow"], block["data_weights"]
        )
        begin_debt = block["begin_debt"]
        flow = block["observed_flow"]
        positive = flow > 0.0
        expected = []
        for restriction in (begin_debt <= 0.0, begin_debt > 0.0):
            for level in (0, 1):
                w = block["data_weights"][level][restriction]
                expected.append(
                    np.sum(w * positive[restriction]) / np.sum(w)
                )
        for level in (0, 1):
            w = block["data_weights"][level][positive]
            expected.append(np.sum(w * flow[positive]) / np.sum(w))
        np.testing.assert_allclose(moments, expected, atol=1e-12)

    def test_hard_assignment_moments_equal_subgroup_statistics(self):
        contexts = _synthetic_panel_contexts()
        block = m.build_loan_type_block(contexts)
        rng = np.random.default_rng(7)
        flow = np.where(
            rng.random(block["begin_debt"].size) < 0.6,
            rng.uniform(200.0, 8000.0, size=block["begin_debt"].size), 0.0,
        )
        moments = m.loan_type_block_moments(
            block["begin_debt"], flow, block["sim_weights"]
        )
        sampled = block["sim_weights"][1].astype(bool)
        entry = block["begin_debt"] <= 0.0
        positive = flow > 0.0
        self.assertAlmostEqual(
            moments[0], float(np.mean(positive[entry & ~sampled])), places=12
        )
        self.assertAlmostEqual(
            moments[1], float(np.mean(positive[entry & sampled])), places=12
        )
        self.assertAlmostEqual(
            moments[2], float(np.mean(positive[~entry & ~sampled])), places=12
        )
        self.assertAlmostEqual(
            moments[3], float(np.mean(positive[~entry & sampled])), places=12
        )
        self.assertAlmostEqual(
            moments[4], float(np.mean(flow[positive & ~sampled])), places=9
        )
        self.assertAlmostEqual(
            moments[5], float(np.mean(flow[positive & sampled])), places=9
        )

    def test_empty_groups_receive_the_eps_floor(self):
        begin_debt = np.zeros(4)          # nobody holds debt: continuation empty
        flow = np.zeros(4)                # nobody borrows: mean flow empty
        weights = np.stack((np.ones(4), np.zeros(4)))  # no high-type mass
        moments = m.loan_type_block_moments(begin_debt, flow, weights, eps=0.01)
        np.testing.assert_array_equal(
            moments, [0.0, 0.01, 0.01, 0.01, 0.01, 0.01]
        )

    def test_misaligned_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            m.loan_type_block_moments(
                np.zeros(4), np.zeros(4), np.ones((2, 5))
            )
        with self.assertRaises(ValueError):
            m.loan_type_block_moments(
                np.zeros(4), np.zeros(5), np.ones((2, 4))
            )

    def test_loss_matches_residuals_weights_and_floors(self):
        data = np.asarray([0.05, 0.56, 0.84, 0.85, 4000.0, 5000.0])
        simulated = np.asarray([0.10, 0.50, 0.80, 0.90, 4400.0, 4800.0])
        loss, residuals = m.loan_type_block_loss_and_residuals(simulated, data)
        weights = np.asarray(m.LOAN_TYPE_BLOCK_MOMENT_WEIGHTS)
        floors = np.asarray(m.LOAN_TYPE_BLOCK_SCALE_FLOORS)
        scale = np.maximum(np.abs(data), floors)
        expected = np.sum(weights * ((simulated - data) / scale) ** 2)
        self.assertAlmostEqual(loss, expected, places=12)
        self.assertAlmostEqual(loss, float(np.sum(residuals ** 2)), places=12)

    def test_loss_rejects_the_wrong_moment_count(self):
        with self.assertRaises(ValueError):
            m.loan_type_block_loss_and_residuals(np.zeros(8), np.zeros(8))


if __name__ == "__main__":
    unittest.main()
