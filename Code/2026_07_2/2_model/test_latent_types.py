import unittest

import numpy as np

from latent_types import (
    N_TYPES,
    TYPE_COMPONENTS,
    TYPE_IDS,
    TYPE_LOAN,
    TYPE_NAMES,
    draw_type_ids,
    sgt_index,
    type_components,
    validate_q,
    validate_saved_layout,
)


class LatentTypeLayoutTests(unittest.TestCase):
    def test_sgtl_ordering(self):
        self.assertEqual(N_TYPES, 16)
        self.assertEqual(TYPE_IDS, tuple(range(1, 17)))
        self.assertEqual(TYPE_NAMES[0], "S0G0T0L0")
        self.assertEqual(TYPE_NAMES[1], "S0G0T0L1")
        self.assertEqual(TYPE_NAMES[-1], "S1G1T1L1")
        self.assertEqual(type_components(10), (1, 0, 0, 1))

    def test_adjacent_loan_types_share_sgt_cell(self):
        for low_type_id in range(1, 17, 2):
            high_type_id = low_type_id + 1
            self.assertEqual(sgt_index(low_type_id), sgt_index(high_type_id))
            self.assertEqual(
                type_components(low_type_id)[:3],
                type_components(high_type_id)[:3],
            )
            self.assertEqual(TYPE_LOAN[low_type_id - 1], 0)
            self.assertEqual(TYPE_LOAN[high_type_id - 1], 1)

    def test_posterior_and_saved_layout_validation(self):
        q = np.full((3, N_TYPES), 1.0 / N_TYPES)
        np.testing.assert_array_equal(validate_q(q), q)
        self.assertTrue(
            validate_saved_layout(
                TYPE_NAMES,
                TYPE_COMPONENTS[:, 0],
                TYPE_COMPONENTS[:, 1],
                TYPE_COMPONENTS[:, 2],
                TYPE_COMPONENTS[:, 3],
            )
        )

    def test_categorical_type_draws_are_one_based(self):
        q = np.zeros((3, N_TYPES))
        q[0, 0] = 1.0
        q[1, 5] = 1.0
        q[2, -1] = 1.0
        np.testing.assert_array_equal(
            draw_type_ids(q, np.array([0.25, 0.50, 0.75])),
            np.array([1, 6, 16]),
        )


if __name__ == "__main__":
    unittest.main()
