import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import model_em_algorithm as model


class PrepareVjtFeasibleTests(unittest.TestCase):
    def test_grouped_loader_matches_individual_selection(self):
        period = 3
        type_ids = (1, 2)
        total_choices = np.asarray(
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 0, 0]],
            dtype=np.int64,
        )
        x1 = np.asarray(
            [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]],
            dtype=np.int64,
        )
        state = np.asarray(
            [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0], [1, 0]],
            dtype=np.int64,
        )
        debt = np.asarray([0, 1, 2, 0, 2, 1], dtype=np.int64)
        observed_choices = np.zeros((len(x1), 3), dtype=np.int64)

        def possible_choices(x2):
            if int(x2[0]) == 0:
                return total_choices[[0, 3]]
            return total_choices[[1, 2, 3]]

        def state_values(type_id, x1i, x2i):
            columns = len(possible_choices(x2i))
            base = 1000 * type_id + 100 * int(x1i[0]) + 10 * int(x2i[0])
            return base + np.arange(3 * columns, dtype=float).reshape(3, columns)

        expected = {}
        for type_id in type_ids:
            array = np.full((len(x1), len(total_choices)), -np.inf)
            for row, (x1i, x2i, debt_index) in enumerate(zip(x1, state, debt)):
                feasible = possible_choices(x2i)
                choice_columns = np.where(
                    (total_choices == feasible[:, None]).all(-1)
                )[1]
                array[row, choice_columns] = state_values(
                    type_id, x1i, x2i
                )[debt_index]
            expected[type_id] = array

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "vjt_nog" / str(period)).mkdir(parents=True)
            (output / "likelihood").mkdir()

            for type_id in type_ids:
                for x1i in np.unique(x1, axis=0):
                    payload = {}
                    selected = np.all(x1 == x1i, axis=1)
                    for x2i in np.unique(state[selected], axis=0):
                        payload[f"vjt_t{period}_[{x1i}]_{x2i}"] = state_values(
                            type_id, x1i, x2i
                        )
                    np.savez_compressed(
                        output
                        / "vjt_nog"
                        / str(period)
                        / f"vjt_t{period}_[{x1i}]_em{type_id}.npz",
                        **payload,
                    )

            original_load = np.load
            opened_bundles = []

            def tracking_load(file, *args, **kwargs):
                if str(file).endswith(".npz"):
                    opened_bundles.append(str(file))
                return original_load(file, *args, **kwargs)

            with (
                mock.patch.object(model, "pathout", str(output)),
                mock.patch.object(model, "TYPE_IDS", type_ids),
                mock.patch.object(
                    model,
                    "load_data_superfeasible",
                    return_value=(x1, state, debt, observed_choices),
                ),
                mock.patch.object(model, "get_total_choices", return_value=total_choices),
                mock.patch.object(model, "get_possible_choices", side_effect=possible_choices),
                mock.patch.object(model.np, "load", side_effect=tracking_load),
            ):
                model.prepare_vjt_feasible(period)

            self.assertEqual(len(opened_bundles), len(type_ids) * len(np.unique(x1, axis=0)))
            for type_id in type_ids:
                observed = original_load(
                    output
                    / "likelihood"
                    / f"vjt_super_t{period}_em{type_id}.npy"
                )
                np.testing.assert_array_equal(observed, expected[type_id])


if __name__ == "__main__":
    unittest.main()
