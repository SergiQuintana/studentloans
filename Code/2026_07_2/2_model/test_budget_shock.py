import unittest

import numpy as np

import budget_shock as bs


class BudgetShockSupportTests(unittest.TestCase):
    def test_grouped_education_cells(self):
        self.assertEqual(
            bs.BUDGET_EDUCATION_CELLS,
            (
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                (3, 1), (3, 2),
            ),
        )
        self.assertEqual(bs.budget_program_year(1, 7), 3)
        self.assertEqual(bs.budget_program_year(2, 7), 5)
        self.assertEqual(bs.budget_program_year(3, 7), 2)
        self.assertEqual(len(bs.BUDGET_EDUCATION_CELLS), 10)
        self.assertEqual(
            len(bs.BUDGET_EDUCATION_CELLS)
            * bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL
            + bs.N_RISK_PARAMETERS + bs.N_DEBT_PENALTY_PARAMETERS,
            68,
        )

    def test_state_mapping_does_not_modify_x2(self):
        state = np.zeros((3, 10), dtype=np.int64)
        state[:, 1] = np.array([0, 1, 6])
        state[:, 2] = np.array([0, 3, 7])
        state[:, 3] = np.array([0, 1, 5])
        original = state.copy()
        np.testing.assert_array_equal(
            bs.budget_education_cell_from_state(state, 1),
            np.array([101, 102, 103]),
        )
        np.testing.assert_array_equal(
            bs.budget_education_cell_from_state(state, 2),
            np.array([201, 204, 205]),
        )
        np.testing.assert_array_equal(
            bs.budget_education_cell_from_state(state, 3),
            np.array([301, 302, 302]),
        )
        np.testing.assert_array_equal(state, original)

    def test_explicit_debt_penalty_multiplier(self):
        beta = 0.98
        periods = np.arange(1, 10)
        observed = bs.explicit_debt_penalty_multiplier(
            periods, beta=beta, terminal_period=10
        )
        expected = np.array([
            sum(beta ** s for s in range(10 - period))
            for period in periods
        ])
        np.testing.assert_allclose(observed, expected)
        self.assertAlmostEqual(observed[0], sum(beta ** s for s in range(9)))
        self.assertAlmostEqual(observed[-1], 1.0)

    def test_grouped_bundle_maps_late_year_to_top_support(self):
        cells = [code for education, year in bs.BUDGET_EDUCATION_CELLS
                 for code in [bs.budget_education_cell_code(education, year)]]
        blocks = []
        for index in range(len(cells)):
            blocks.extend([1000.0 + index] * 4 + [50.0 + index, 0.0])
        vector = np.asarray(blocks + [1.0] * 4 + [-2.0] * 4)
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        state = np.zeros(10, dtype=np.int64)
        state[2] = 8
        mean = bs.conditional_mean(
            spec, np.array([1, 1, 0, 0]), education=2, state=state,
            pre_choice_resources=0.0,
        )
        four_year_top_index = cells.index(205)
        self.assertEqual(float(mean), 1000.0 + four_year_top_index)


if __name__ == "__main__":
    unittest.main()
