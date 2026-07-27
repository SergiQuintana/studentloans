"""Tests for the two-component need mixture in the budget shock.

Guards the contract every consumer relies on: a bundle without the mixture
behaves exactly as before (bitwise), the optimizer-vector tail decodes to the
documented slots, the component probability is the stated logistic, and the
realized shock picks the component the uniform draw selects.
"""

import unittest

import numpy as np

import budget_shock as bs


def _cells():
    return np.asarray(
        [bs.budget_education_cell_code(education, year)
         for education, year in bs.BUDGET_EDUCATION_CELLS],
        dtype=np.int64,
    )


def _base_vector(cells, seed=0):
    """A well-formed base (no optional tail) multicell vector."""
    rng = np.random.default_rng(seed)
    size = bs.estimation_vector_size_multicell(cells.size)
    vector = rng.normal(size=size)
    blocks = vector[:bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL * cells.size]
    blocks = blocks.reshape(cells.size, bs.PARENTAL_INCOME_MULTICELL_PARAMETERS_PER_CELL)
    blocks[:, 4] = np.abs(blocks[:, 4]) + 1.0     # sigma must be positive
    blocks[:, 5] = 0.0    # zero resource slope: no pre-choice resources needed
    risk = vector.size - bs.N_RISK_PARAMETERS - bs.N_DEBT_PENALTY_PARAMETERS
    vector[risk:risk + bs.N_RISK_PARAMETERS] = 1.5
    return vector


# The fourth grouped cell is four-year program year one (code 201).
SUPPORT = dict(education=2, program_year=1)


MIXTURE_BLOCK = np.asarray([-2.0, 3.5, 1.5, 250.0, 2000.0])


class VectorLayoutTests(unittest.TestCase):
    def test_base_vector_has_no_mixture(self):
        cells = _cells()
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            _base_vector(cells), cells, index_kind="education_cell"
        )
        self.assertIsNone(spec["mixture_logits"])
        self.assertFalse(bs.mixture_enabled(spec))

    def test_mixture_block_decodes_to_documented_slots(self):
        cells = _cells()
        base = _base_vector(cells)
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            np.concatenate((base, MIXTURE_BLOCK)), cells, index_kind="education_cell"
        )
        self.assertTrue(bs.mixture_enabled(spec))
        np.testing.assert_array_equal(spec["mixture_logits"], MIXTURE_BLOCK[:3])
        self.assertEqual(spec["mixture_noneed_mean"], MIXTURE_BLOCK[3])
        self.assertEqual(spec["mixture_noneed_sigma"], MIXTURE_BLOCK[4])

    def test_shift_stays_last_after_the_mixture(self):
        cells = _cells()
        base = _base_vector(cells)
        vector = np.concatenate((base, MIXTURE_BLOCK, [-7.0]))
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        np.testing.assert_array_equal(spec["mixture_logits"], MIXTURE_BLOCK[:3])
        self.assertEqual(spec["debt_penalty_loan_type_shift"], -7.0)

    def test_kappa_and_mixture_together(self):
        cells = _cells()
        base = _base_vector(cells)
        kappa = np.asarray([-1.0, -2.0, -0.5])
        vector = np.concatenate((base, kappa, MIXTURE_BLOCK, [-7.0]))
        spec = bs.unpack_parental_income_multicell_estimation_vector(
            vector, cells, index_kind="education_cell"
        )
        np.testing.assert_array_equal(
            spec["new_borrow_cost_entry_by_loan_type"], kappa[:2]
        )
        self.assertEqual(spec["new_borrow_cost_continuation"], kappa[2])
        np.testing.assert_array_equal(spec["mixture_logits"], MIXTURE_BLOCK[:3])
        self.assertEqual(spec["debt_penalty_loan_type_shift"], -7.0)

    def test_vector_sizes(self):
        n = len(bs.BUDGET_EDUCATION_CELLS)
        base = bs.estimation_vector_size_multicell(n)
        self.assertEqual(
            bs.estimation_vector_size_multicell(n, include_need_mixture=True),
            base + bs.N_MIXTURE_PARAMETERS,
        )
        self.assertEqual(
            bs.estimation_vector_size_multicell(
                n, include_new_borrowing=True, include_need_mixture=True,
                include_loan_type_debt_penalty=True,
            ),
            base + bs.N_NEW_BORROWING_PARAMETERS + bs.N_MIXTURE_PARAMETERS + 1,
        )

    def test_ambiguous_tail_lengths_rejected(self):
        cells = _cells()
        base = _base_vector(cells)
        for bad_extra in (2, 7, 10):
            with self.assertRaises(ValueError):
                bs.unpack_parental_income_multicell_estimation_vector(
                    np.concatenate((base, np.zeros(bad_extra))),
                    cells, index_kind="education_cell",
                )


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        cells = _cells()
        self.spec = bs.validate(
            bs.unpack_parental_income_multicell_estimation_vector(
                np.concatenate((_base_vector(cells), MIXTURE_BLOCK)),
                cells, index_kind="education_cell",
            )
        )

    def test_matches_the_closed_form(self):
        a0, a_type, a_debt = MIXTURE_BLOCK[:3]
        for loan in (0, 1):
            for debt in (False, True):
                linear = a0 + a_type * loan + a_debt * debt
                self.assertAlmostEqual(
                    bs.mixture_probability(self.spec, loan_type=loan, has_debt=debt),
                    1.0 / (1.0 + np.exp(-linear)),
                    places=12,
                )

    def test_broadcasts_and_stays_in_the_unit_interval(self):
        loan = np.array([0, 1, 0, 1])
        debt = np.array([False, False, True, True])
        probability = bs.mixture_probability(self.spec, loan_type=loan, has_debt=debt)
        self.assertEqual(probability.shape, (4,))
        self.assertTrue(np.all((probability > 0.0) & (probability < 1.0)))

    def test_extreme_logits_do_not_overflow(self):
        spec = dict(self.spec)
        for a0 in (-800.0, 800.0):
            spec["mixture_logits"] = np.asarray([a0, 0.0, 0.0])
            with np.errstate(over="raise"):
                probability = bs.mixture_probability(spec, loan_type=0, has_debt=False)
            self.assertTrue(np.isfinite(probability))
            self.assertTrue(0.0 <= probability <= 1.0)

    def test_disabled_specification_raises(self):
        spec = dict(self.spec)
        spec["mixture_logits"] = None
        with self.assertRaises(ValueError):
            bs.mixture_probability(spec, loan_type=0, has_debt=False)


class RealizationTests(unittest.TestCase):
    def setUp(self):
        cells = _cells()
        self.cells = cells
        self.spec = bs.validate(
            bs.unpack_parental_income_multicell_estimation_vector(
                np.concatenate((_base_vector(cells), MIXTURE_BLOCK)),
                cells, index_kind="education_cell",
            )
        )
        self.x1 = np.column_stack((np.arange(1, 5), np.ones(4)))

    def test_components_match_the_single_normal_accessors(self):
        mean_need, sigma_need, mean_noneed, sigma_noneed = bs.mixture_components(
            self.spec, self.x1, None, **SUPPORT
        )
        np.testing.assert_array_equal(
            mean_need, bs.conditional_mean(self.spec, self.x1, None, **SUPPORT)
        )
        self.assertEqual(sigma_need, bs.conditional_sigma(self.spec, None, **SUPPORT))
        self.assertEqual(mean_noneed, MIXTURE_BLOCK[3])
        self.assertEqual(sigma_noneed, MIXTURE_BLOCK[4])

    def test_uniform_draw_selects_the_component(self):
        standard = np.full(4, 0.5)
        probability = bs.mixture_probability(self.spec, loan_type=0, has_debt=False)
        mean_need, sigma_need, mean_noneed, sigma_noneed = bs.mixture_components(
            self.spec, self.x1, None, **SUPPORT
        )
        below = bs.realization_mixture(
            self.spec, self.x1, None, standard,
            np.zeros(4), has_debt=False, loan_type=0, **SUPPORT,
        )
        above = bs.realization_mixture(
            self.spec, self.x1, None, standard,
            np.ones(4), has_debt=False, loan_type=0, **SUPPORT,
        )
        self.assertTrue(0.0 < probability < 1.0)
        np.testing.assert_allclose(below, mean_need + sigma_need * standard)
        np.testing.assert_allclose(above, mean_noneed + sigma_noneed * standard)

    def test_certain_need_reproduces_the_single_component_path(self):
        spec = dict(self.spec)
        spec["mixture_logits"] = np.asarray([50.0, 0.0, 0.0])   # p is one
        standard = np.linspace(-2.0, 2.0, 4)
        np.testing.assert_allclose(
            bs.realization_mixture(
                spec, self.x1, None, standard, np.full(4, 0.5),
                has_debt=False, loan_type=0, **SUPPORT,
            ),
            bs.realization(spec, self.x1, None, standard, loan_type=0, **SUPPORT),
        )

    def test_debt_state_shifts_the_component_probability(self):
        without = bs.mixture_probability(self.spec, loan_type=0, has_debt=False)
        with_debt = bs.mixture_probability(self.spec, loan_type=0, has_debt=True)
        self.assertGreater(with_debt, without)     # a_debt is positive here


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.cells = _cells()
        self.base = _base_vector(self.cells)

    def _spec(self, block):
        return bs.unpack_parental_income_multicell_estimation_vector(
            np.concatenate((self.base, block)), self.cells,
            index_kind="education_cell",
        )

    def test_legacy_bundle_normalizes_to_disabled(self):
        spec = bs.validate(
            bs.unpack_parental_income_multicell_estimation_vector(
                self.base, self.cells, index_kind="education_cell"
            )
        )
        self.assertIsNone(spec["mixture_logits"])
        self.assertFalse(bs.mixture_enabled(spec))

    def test_dictionary_without_the_key_is_accepted(self):
        spec = bs.validate(
            bs.unpack_parental_income_multicell_estimation_vector(
                self.base, self.cells, index_kind="education_cell"
            )
        )
        spec.pop("mixture_logits")
        self.assertFalse(bs.mixture_enabled(bs.validate(spec)))

    def test_non_positive_noneed_sigma_rejected(self):
        block = MIXTURE_BLOCK.copy()
        block[4] = 0.0
        with self.assertRaises(ValueError):
            bs.validate(self._spec(block))

    def test_non_finite_logits_rejected(self):
        spec = self._spec(MIXTURE_BLOCK)
        spec["mixture_logits"] = np.asarray([np.nan, 0.0, 0.0])
        with self.assertRaises(ValueError):
            bs.validate(spec)

    def test_wrong_logit_count_rejected(self):
        spec = self._spec(MIXTURE_BLOCK)
        spec["mixture_logits"] = np.asarray([0.0, 1.0])
        with self.assertRaises(ValueError):
            bs.validate(spec)

    def test_unsupported_timing_rejected(self):
        spec = self._spec(MIXTURE_BLOCK)
        spec["need_mixture_timing"] = "something_else"
        with self.assertRaises(ValueError):
            bs.validate(spec)


if __name__ == "__main__":
    unittest.main()
