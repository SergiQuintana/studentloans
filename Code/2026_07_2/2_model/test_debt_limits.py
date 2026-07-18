import unittest

import numpy as np

from debt_limits import (
    CONSUMPTION_FLOOR,
    INTEREST_RATE,
    get_annual_cap_by_stage,
    get_debt_region_bounds,
    get_lifetime_cap_by_stage,
    get_simulation_bounds_indices,
    precompute_smm_bounds_indices,
)


class DebtLimitTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.array(
            [0.0, 5000.0, 9000.0, 15000.0, 25000.0, 70000.0,
             80000.0, 100000.0, 160000.0]
        )

    def test_shared_constants_and_stage_caps(self):
        self.assertEqual(INTEREST_RATE, 0.05)
        self.assertEqual(CONSUMPTION_FLOOR, 2000.0)
        self.assertEqual(get_annual_cap_by_stage(1, 0, 0), 8391.0)
        self.assertEqual(get_annual_cap_by_stage(1, 1, 0), 9309.0)
        self.assertEqual(get_annual_cap_by_stage(2, 0, 2), 12581.0)
        self.assertEqual(get_annual_cap_by_stage(3, 0, 0), 23222.0)
        self.assertEqual(get_lifetime_cap_by_stage(1), 70786.0)
        self.assertEqual(get_lifetime_cap_by_stage(3), 150000.0)

    def test_bellman_and_forward_strict_bounds_agree(self):
        state = np.zeros(10, dtype=np.int64)
        choice = np.array([12, 1, 0], dtype=np.int64)
        lo_all, hi_all, _ = get_debt_region_bounds(self.grid, state, choice)

        current_indices = np.arange(len(self.grid), dtype=np.int64)
        states = np.repeat(state[None, :], len(self.grid), axis=0)
        choices = np.repeat(choice[None, :], len(self.grid), axis=0)
        lo_forward, hi_forward, _ = get_simulation_bounds_indices(
            current_indices, states, choices, self.grid
        )
        np.testing.assert_array_equal(lo_forward, lo_all)
        np.testing.assert_array_equal(hi_forward, hi_all)

    def test_smm_retains_nearest_grid_mapping(self):
        previous_debt = np.array([0.0])
        states = np.zeros((1, 10), dtype=np.int64)
        choices = np.array([[12, 1, 0]], dtype=np.int64)
        lo_idx, hi_idx = precompute_smm_bounds_indices(
            previous_debt, states, choices, self.grid
        )
        self.assertEqual(int(lo_idx[0]), 0)
        # 8,391 is closer to 9,000 than 5,000, matching the existing SMM.
        self.assertEqual(int(hi_idx[0]), 2)


if __name__ == "__main__":
    unittest.main()
