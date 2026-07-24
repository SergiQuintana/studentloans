# -*- coding: utf-8 -*-
"""Fast CCP-sequence builder (dense-matrix rewrite of model_getccp_sequence).

STATUS 2026-07-24: PROTOTYPE, NOT WIRED INTO PRODUCTION. The production
driver still calls model_getccp_sequence.get_ccp_sequence. This module is
exercised by test_ccp_sequence_fast_equivalence.py; promotion into
estimation_all_em.py happens only after that test passes on the server
(solver-change protocol; see Agents_Readme/Tasks/
ESTIMATION_SPEED_ANALYSIS_2026_07_24.md, points 3-4).

What it computes — the identical recursion as the original:

    evt_p(x2) = -log(ccp_home_p(x2)) + beta * evt_{p+1}(succ(x2))

with succ(x2) the home-production state transition and zero continuation
at p = 9. The arithmetic is performed with the same elementwise numpy
operations as the original (unary minus of np.log, then + beta * cont), so
outputs are BITWISE identical; the speed comes from bookkeeping only:

  * per-period integer successor indices, precomputed once and shared by
    all (invariant state, type) tasks, replace per-state dictionary keys
    that embedded numpy arrays formatted into strings;
  * the per-period CCP bundle is read into one dense (n_states, n_debt)
    matrix and the recursion is one vectorized line per period;
  * output can be written either in the production per-state format
    ("legacy", a drop-in for every existing consumer) or as one dense
    matrix per period ("dense", the point-4 format).

Dense-format convention (write_mode="dense"): file
``evt_ccp_dense/{p}/evt_ccp_dense_t{p}_{inv}_em{type}.npz`` holds
``evt``    -- float64 (n_states, n_debt); ROW k IS STATE k IN THE ROW ORDER
              OF ``states_t{p}.npy`` (the file get_x2(p) loads);
``states`` -- int64 copy of that state matrix, for self-description.

DenseSequenceReader replicates model_fitloans_dynamic's
``_continuation_from_bundle`` (graduation mixing included, same helper
functions, same elementwise arithmetic) on top of the dense format.
"""

import os
import numpy as np

import model_solution_em as ms
from model_solution_em import (
    get_x2,
    move_state_grad,
    get_x1_new,
    probability_graduation,
    save_npz_here,
)
from latent_types import type_index
from config import DIR

path_out = DIR["MODEL_OUTPUT"]

T = 10
beta = 0.98
_HOME = np.array([0, 0, 0])

# ---------------------------------------------------------------------------
# One-time per-process statics (shared across all tasks and types)
# ---------------------------------------------------------------------------

_STATICS = None


def build_sequence_statics(force=False):
    """States, key strings, row lookups, and successor indices per period.

    Everything here depends only on the state files, never on the task, so
    one build serves all 1,024 production tasks of a process.
    """
    global _STATICS
    if _STATICS is not None and not force:
        return _STATICS

    states = {}
    x2_strings = {}
    row_index = {}
    for p in range(1, T):
        sp = np.ascontiguousarray(get_x2(p).astype("int"))
        states[p] = sp
        # Identical text to the original f-string keys (str of an int array).
        x2_strings[p] = [str(sp[k]) for k in range(sp.shape[0])]
        row_index[p] = {sp[k].tobytes(): k for k in range(sp.shape[0])}

    succ = {}
    for p in range(1, T - 1):
        sp = states[p]
        nxt = row_index[p + 1]
        s = np.empty(sp.shape[0], dtype=np.int64)
        for k in range(sp.shape[0]):
            x2n = np.ascontiguousarray(
                np.asarray(move_state_grad(sp[k], _HOME, p), dtype=np.int64)
            )
            s[k] = nxt[x2n.tobytes()]
        succ[p] = s

    _STATICS = {
        "states": states,
        "x2_strings": x2_strings,
        "row_index": row_index,
        "succ": succ,
    }
    return _STATICS


# ---------------------------------------------------------------------------
# The fast builder
# ---------------------------------------------------------------------------

def _default_legacy_writer(rel_path, names, arrays):
    save_npz_here(rel_path, names, arrays, compressed=True)


def _write_to_root(root):
    def writer(rel_path, names, arrays):
        full = os.path.join(root, *rel_path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        np.savez_compressed(full, **{n: a for n, a in zip(names, arrays)})
    return writer


def get_ccp_sequence_fast(
    i,
    x1,
    b,
    em_type,
    write_mode="legacy",
    out_root=None,
    ccp_root=None,
    return_dense=False,
    verbose=True,
):
    """Drop-in computation of one CCP-sequence task, vectorized.

    Parameters mirror model_getccp_sequence.get_ccp_sequence(i, x1, b,
    em_type); ``b`` is accepted and unused exactly as in the original.

    write_mode : "legacy" (production per-state compressed npz, byte-for-
        byte the same keys and arrays as the original), "dense" (one
        matrix per period, see module docstring), "both", or "none".
    out_root : optional directory root that replaces Model/Output as the
        output anchor (used by the equivalence test so Model/ is never
        touched). None -> production locations.
    ccp_root : optional directory root for READING the input CCP bundles
        (default: production Model/Output/ccp).
    return_dense : also return {period: dense evt matrix}.
    """
    type_index(em_type)
    statics = build_sequence_statics()
    states = statics["states"]
    x2_strings = statics["x2_strings"]
    succ = statics["succ"]

    inv = x1[i, :]
    if verbose:
        print(f"Individual {inv.astype('int')} (fast)")

    read_root = ccp_root if ccp_root is not None else path_out
    if write_mode in ("legacy", "both") and out_root is None:
        legacy_writer = _default_legacy_writer
    else:
        legacy_writer = _write_to_root(out_root) if out_root else None
    dense_root = (
        os.path.join(out_root, "evt_ccp_dense")
        if out_root
        else os.path.join(path_out, "evt_ccp_dense")
    )

    dense_by_period = {}
    evt_next = None
    for period in range(T - 1, 0, -1):
        n = states[period].shape[0]
        keys = x2_strings[period]

        with np.load(
            f"{read_root}/ccp/{period}/ccp_t{period}_[{inv}]_em{em_type}.npz"
        ) as bundle:
            first = bundle[f"ccp_t{period}_[{inv}]_{keys[0]}"]
            ccp = np.empty((n,) + np.shape(first), dtype=np.float64)
            ccp[0] = first
            for k in range(1, n):
                ccp[k] = bundle[f"ccp_t{period}_[{inv}]_{keys[k]}"]

        # Same elementwise arithmetic as get_ccp_continuation: the original
        # computes -log(ccp) + beta*cont with cont = 0 (period 9) or the
        # successor row; broadcasting does not change IEEE results.
        if period == T - 1:
            evt = -np.log(ccp) + beta * 0.0
        else:
            evt = -np.log(ccp) + beta * evt_next[succ[period]]
        evt_next = evt
        dense_by_period[period] = evt

        if write_mode in ("legacy", "both"):
            names = [
                f"evt_ccp_sequence_t{period}_{inv}_{keys[k]}"
                for k in range(n)
            ]
            legacy_writer(
                f"evt_ccp/{period}/evt_ccp_sequence_t{period}_{inv}_em{em_type}.npz",
                names,
                [evt[k] for k in range(n)],
            )
        if write_mode in ("dense", "both"):
            full = os.path.join(
                dense_root, str(period),
                f"evt_ccp_dense_t{period}_{inv}_em{em_type}.npz",
            )
            os.makedirs(os.path.dirname(full), exist_ok=True)
            np.savez_compressed(full, evt=evt, states=states[period])

    if return_dense:
        return dense_by_period
    return None


# ---------------------------------------------------------------------------
# Dense-format consumer (mirror of model_fitloans_dynamic's bundle reader)
# ---------------------------------------------------------------------------

class DenseSequenceReader:
    """Continuation lookups on the dense format.

    Replicates ``_continuation_from_bundle`` in model_fitloans_dynamic:
    the consumer at model period ``period`` reads the t = period+1 bundle,
    moves the state with the actual choice, and mixes the graduation and
    no-graduation rows with the graduation probability. Uses the very same
    helper functions (move_state_grad, get_x1_new, probability_graduation)
    and the same elementwise arithmetic, so results are bitwise identical.
    """

    def __init__(self, period, x1i, em_type, root=None):
        self.period = int(period)
        self.x1i = np.asarray(x1i, dtype=np.int64)
        base = root if root is not None else os.path.join(path_out, "evt_ccp_dense")
        path = os.path.join(
            base, str(self.period + 1),
            f"evt_ccp_dense_t{self.period + 1}_{self.x1i}_em{em_type}.npz",
        )
        with np.load(path) as bundle:
            self.evt = bundle["evt"]
        statics = build_sequence_statics()
        self._rows = statics["row_index"][self.period + 1]

    def _row(self, x2):
        key = np.ascontiguousarray(
            np.asarray(x2, dtype=np.int64)
        ).tobytes()
        return self.evt[self._rows[key]]

    def continuation(self, x2i, ji):
        x2i = np.asarray(x2i, dtype=np.int64)
        graduation_possible = (
            ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0))
            | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0))
            | (ji[1] == 3)
        )
        notgrad_x2 = move_state_grad(x2i, ji, self.period)
        evt_nograd = np.asarray(self._row(notgrad_x2), dtype=np.float64)
        if not graduation_possible:
            return evt_nograd
        grad_x2 = move_state_grad(x2i, ji, self.period, grad=1)
        evt_grad = np.asarray(self._row(grad_x2), dtype=np.float64)
        p_grad = probability_graduation(get_x1_new(self.x1i), x2i, ji)
        return p_grad * evt_grad + (1.0 - p_grad) * evt_nograd
