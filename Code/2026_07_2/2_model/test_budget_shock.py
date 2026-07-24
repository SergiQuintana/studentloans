import os
import tempfile
import unittest

import numpy as np

import budget_shock as bs


def _multicell_cells():
    return [bs.budget_education_cell_code(education, year)
            for education, year in bs.BUDGET_EDUCATION_CELLS]


def _multicell_vector(cells):
    blocks = []
    for index in range(len(cells)):
        blocks.extend([1000.0 + index] * 4 + [50.0 + index, 0.0])
    return np.asarray(blocks + [1.0] * 4 + [-2.0] * 4)


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


class NewBorrowingCostTests(unittest.TestCase):
    KAPPA = np.array([-500.0, -250.0, -75.0])

    def _spec(self, extended):
        cells = _multicell_cells()
        vector = _multicell_vector(cells)
        if extended:
            vector = np.concatenate((vector, self.KAPPA))
        return bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )

    def test_vector_sizes(self):
        self.assertEqual(bs.N_NEW_BORROWING_PARAMETERS, 3)
        self.assertEqual(bs.estimation_vector_size_multicell(10), 68)
        self.assertEqual(
            bs.estimation_vector_size_multicell(10, include_new_borrowing=True), 71
        )

    def test_legacy_68_vector_unpacks_to_zero_kappas(self):
        spec = self._spec(extended=False)
        np.testing.assert_array_equal(
            spec["new_borrow_cost_entry_by_loan_type"], np.zeros(2)
        )
        self.assertEqual(spec["new_borrow_cost_continuation"], 0.0)
        self.assertEqual(
            spec["new_borrowing_cost_timing"], bs.NEW_BORROWING_COST_TIMING
        )

    def test_extended_71_vector_maps_trailing_kappas(self):
        spec = self._spec(extended=True)
        np.testing.assert_array_equal(
            spec["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )
        self.assertEqual(spec["new_borrow_cost_continuation"], self.KAPPA[2])
        # The kappa block must not disturb the existing parameter mapping.
        legacy = self._spec(extended=False)
        np.testing.assert_array_equal(spec["mu_blocks"], legacy["mu_blocks"])
        np.testing.assert_array_equal(spec["sigma_e"], legacy["sigma_e"])
        np.testing.assert_array_equal(spec["risk_aversion"], legacy["risk_aversion"])
        np.testing.assert_array_equal(
            spec["debt_pen_parinc"], legacy["debt_pen_parinc"]
        )
        np.testing.assert_array_equal(
            spec["budget_resource_slope"], legacy["budget_resource_slope"]
        )

    def test_accessor_scalar_and_vectorized(self):
        spec = self._spec(extended=True)
        self.assertEqual(bs.new_borrowing_cost(spec, True), -75.0)
        self.assertEqual(bs.new_borrowing_cost(spec, False, loan_type=0), -500.0)
        self.assertEqual(bs.new_borrowing_cost(spec, False, loan_type=1), -250.0)
        # loan_type=None uses the loan-type-0 entry cost.
        self.assertEqual(bs.new_borrowing_cost(spec, False), -500.0)
        np.testing.assert_array_equal(
            bs.new_borrowing_cost(
                spec,
                np.array([True, False, False, True]),
                loan_type=np.array([0, 0, 1, 1]),
            ),
            np.array([-75.0, -500.0, -250.0, -75.0]),
        )
        with self.assertRaises(ValueError):
            bs.new_borrowing_cost(spec, False, loan_type=2)

    def test_accessor_raw_parameters_are_copies(self):
        spec = self._spec(extended=True)
        entry, continuation = bs.new_borrowing_cost_parameters(spec)
        np.testing.assert_array_equal(entry, self.KAPPA[:2])
        self.assertEqual(continuation, self.KAPPA[2])
        entry[0] = 0.0
        np.testing.assert_array_equal(
            spec["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )

    def test_validate_defaults_missing_keys_on_legacy_dict(self):
        spec = self._spec(extended=False)
        for key in (
            "new_borrow_cost_entry_by_loan_type",
            "new_borrow_cost_continuation",
            "new_borrowing_cost_timing",
        ):
            del spec[key]
        validated = bs.validate(spec)
        np.testing.assert_array_equal(
            validated["new_borrow_cost_entry_by_loan_type"], np.zeros(2)
        )
        self.assertEqual(validated["new_borrow_cost_continuation"], 0.0)
        self.assertEqual(
            validated["new_borrowing_cost_timing"], bs.NEW_BORROWING_COST_TIMING
        )
        self.assertEqual(bs.new_borrowing_cost(validated, True), 0.0)
        self.assertEqual(bs.new_borrowing_cost(validated, False, loan_type=1), 0.0)

    def test_save_load_round_trip_preserves_kappa_keys(self):
        spec = self._spec(extended=True)
        original_est = bs.EST
        with tempfile.TemporaryDirectory() as directory:
            bs.EST = lambda *parts: os.path.join(directory, *map(str, parts))
            try:
                bs.save(spec, raw_vector=np.zeros(71))
                loaded = bs.load()
            finally:
                bs.EST = original_est
        np.testing.assert_array_equal(
            loaded["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )
        self.assertEqual(loaded["new_borrow_cost_continuation"], self.KAPPA[2])
        self.assertEqual(
            loaded["new_borrowing_cost_timing"], bs.NEW_BORROWING_COST_TIMING
        )
        np.testing.assert_array_equal(loaded["mu_blocks"], spec["mu_blocks"])

    def test_extended_vector_keeps_grouped_state_lookup(self):
        cells = _multicell_cells()
        vector = np.concatenate((_multicell_vector(cells), self.KAPPA))
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        state = np.zeros(10, dtype=np.int64)
        state[2] = 8
        mean = bs.conditional_mean(
            spec, np.array([1, 1, 0, 0]), education=2, state=state,
            pre_choice_resources=0.0,
        )
        self.assertEqual(float(mean), 1000.0 + cells.index(205))


class LoanTypeDebtPenaltyShiftTests(unittest.TestCase):
    KAPPA = np.array([-500.0, -250.0, -75.0])
    SHIFT = -0.8

    def _spec(self, with_kappa=False, with_shift=False):
        cells = _multicell_cells()
        vector = _multicell_vector(cells)
        if with_kappa:
            vector = np.concatenate((vector, self.KAPPA))
        if with_shift:
            vector = np.concatenate((vector, [self.SHIFT]))
        return bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )

    def test_vector_sizes(self):
        self.assertEqual(bs.N_LOAN_TYPE_DEBT_PENALTY_PARAMETERS, 1)
        self.assertEqual(bs.DEBT_PENALTY_SHIFT_LOAN_TYPE, 0)
        self.assertEqual(bs.estimation_vector_size_multicell(10), 68)
        self.assertEqual(
            bs.estimation_vector_size_multicell(
                10, include_loan_type_debt_penalty=True
            ),
            69,
        )
        self.assertEqual(
            bs.estimation_vector_size_multicell(10, include_new_borrowing=True),
            71,
        )
        self.assertEqual(
            bs.estimation_vector_size_multicell(
                10, include_new_borrowing=True,
                include_loan_type_debt_penalty=True,
            ),
            72,
        )

    def test_all_four_layouts_unpack(self):
        legacy = self._spec()
        self.assertEqual(legacy["debt_penalty_loan_type_shift"], 0.0)
        shift_only = self._spec(with_shift=True)
        self.assertEqual(shift_only["debt_penalty_loan_type_shift"], self.SHIFT)
        np.testing.assert_array_equal(
            shift_only["new_borrow_cost_entry_by_loan_type"], np.zeros(2)
        )
        self.assertEqual(shift_only["new_borrow_cost_continuation"], 0.0)
        kappa_only = self._spec(with_kappa=True)
        self.assertEqual(kappa_only["debt_penalty_loan_type_shift"], 0.0)
        np.testing.assert_array_equal(
            kappa_only["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )
        both = self._spec(with_kappa=True, with_shift=True)
        self.assertEqual(both["debt_penalty_loan_type_shift"], self.SHIFT)
        np.testing.assert_array_equal(
            both["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )
        self.assertEqual(both["new_borrow_cost_continuation"], self.KAPPA[2])
        # The shift tail must not disturb the existing parameter mapping.
        for spec in (shift_only, kappa_only, both):
            np.testing.assert_array_equal(spec["mu_blocks"], legacy["mu_blocks"])
            np.testing.assert_array_equal(spec["sigma_e"], legacy["sigma_e"])
            np.testing.assert_array_equal(
                spec["risk_aversion"], legacy["risk_aversion"]
            )
            np.testing.assert_array_equal(
                spec["debt_pen_parinc"], legacy["debt_pen_parinc"]
            )
            np.testing.assert_array_equal(
                spec["budget_resource_slope"], legacy["budget_resource_slope"]
            )

    def test_invalid_sizes_rejected(self):
        cells = _multicell_cells()
        vector = _multicell_vector(cells)
        for bad_extra in (2, 5):
            with self.assertRaises(ValueError):
                bs.unpack_parental_income_multicell_estimation_vector(
                    np.concatenate((vector, np.zeros(bad_extra))),
                    cells, index_kind="education_cell",
                )

    def test_accessor_applies_shift_to_low_type_only(self):
        spec = self._spec(with_shift=True)
        x1 = np.column_stack((np.arange(1, 5), np.ones(4)))
        base = bs.debt_penalty(spec, x1)
        # loan_type=None applies no shift and reproduces debt_penalty exactly.
        np.testing.assert_array_equal(
            bs.debt_penalty_by_loan_type(spec, x1), base
        )
        loan_type = np.array([0, 1, 0, 1])
        effective = bs.debt_penalty_by_loan_type(spec, x1, loan_type)
        np.testing.assert_array_equal(
            effective, base + self.SHIFT * (loan_type == 0)
        )
        with self.assertRaises(ValueError):
            bs.debt_penalty_by_loan_type(spec, x1, np.array([0, 2, 0, 1]))

    def test_accessor_zero_shift_is_identity(self):
        spec = self._spec()
        x1 = np.column_stack((np.arange(1, 5), np.ones(4)))
        base = bs.debt_penalty(spec, x1)
        np.testing.assert_array_equal(
            bs.debt_penalty_by_loan_type(spec, x1, np.array([0, 1, 0, 1])),
            base,
        )

    def test_validate_defaults_missing_key_on_legacy_dict(self):
        spec = self._spec()
        del spec["debt_penalty_loan_type_shift"]
        validated = bs.validate(spec)
        self.assertEqual(validated["debt_penalty_loan_type_shift"], 0.0)
        spec["debt_penalty_loan_type_shift"] = np.nan
        with self.assertRaises(ValueError):
            bs.validate(spec)

    def test_save_load_round_trip_preserves_shift(self):
        spec = self._spec(with_kappa=True, with_shift=True)
        original_est = bs.EST
        with tempfile.TemporaryDirectory() as directory:
            bs.EST = lambda *parts: os.path.join(directory, *map(str, parts))
            try:
                bs.save(spec, raw_vector=np.zeros(72))
                loaded = bs.load()
            finally:
                bs.EST = original_est
        self.assertEqual(loaded["debt_penalty_loan_type_shift"], self.SHIFT)
        np.testing.assert_array_equal(
            loaded["new_borrow_cost_entry_by_loan_type"], self.KAPPA[:2]
        )
        np.testing.assert_array_equal(loaded["mu_blocks"], spec["mu_blocks"])


if __name__ == "__main__":
    unittest.main()
