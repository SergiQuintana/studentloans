# -*- coding: utf-8 -*-
"""Fast, exact reimplementation of the Bellman solve (Phase 1 of the speed plan).

Drop-in alternative to ``model_solution_em.get_all_evt``: same signature, same
artifacts (paths, npz names, shapes), same numbers. Select it in
``estimation_all_em.py`` with ``USE_FAST_SOLVER = True`` only after
``test_fast_solver_equivalence.py`` passes on the server.

The economic algorithm is untouched. All arithmetic reuses the numeric kernels
of ``model_solution_em`` (wage/income functions, fused debt search, power
utility, graduation logits, budget-shock quadrature), evaluated with the same
inputs in the same order, so results should agree to the last bit. What changes
is only repeated Python work, per Agents_Readme/Tasks/SPEED_PLAN_FINAL.md:

- continuation values: dense ``evt[state_index, debt_index]`` arrays with
  precomputed integer successor indices, replacing per-choice string-key
  dictionary lookups in ``VT``/``evolve_continuation``;
- Gauss-Hermite nodes/weights: cached once per process (identical values --
  this is caching, NOT a quadrature reduction);
- debt snapping: sorted-grid lookup replacing the full squared-distance
  matrix, reproducing ``np.argmin`` tie behavior exactly;
- per-period static objects (feasible choices, successors, g() designs,
  ``x2_new`` rows) built once per process instead of once per state visit;
- per-task invariants (``x1_new``, ``sigma_u``, debt penalty, ``fin_help``,
  graduation probabilities, study-wage index) hoisted out of inner loops.

Known quirk replicated on purpose (flagged to the researcher 2026-07-23): in
``get_expected_conditional`` the pre-choice-resources wage index is
``wage0(x1_new, x2)`` with the RAW state vector, while the resources built in
``get_consumption_resources`` use ``wage0(x1_new, x2_new)``. The fast path
reproduces both calls verbatim so results match the production solver.
"""

import os
import time

import numpy as np
import scipy.special

import budget_shock as bs
import model_solution_em as ms
from numba import njit
from debt_limits import CONSUMPTION_FLOOR, get_debt_region_bounds
from fused_debt_search import _flow_utility

T = ms.T

# Phase 2 / 2b: reuse flow objects (budget-shock nodes, resources, incomes,
# debt transitions, utilities) across choices and states that share every
# input those objects consume. Cache keys are derived MECHANICALLY from the
# argument lists (see SPEED_PLAN_FINAL.md Phase 2b cache-key rule):
#   education payload (z_joint, resources): (j[1], j[2], x2[0:9])
#     - bs.realization uses x1/period (task/loop), education=j[1], the
#       program-year column (inside x2[0:9]), and pre_choice_resources,
#       which depends on fin_help(j1,j2), tuition(j1), the debt grid, and
#       the raw-x2 wage quirk wage0(x1_new, x2) -> columns 0-8.
#     - get_consumption_resources adds x2_new = f(x2[0], x2[8], x2[6]).
#   work/home payload (debt_position, u): (j, x2[{0,6,7,8}])
#     - get_debt_income uses x2[7] (years since school) and
#       x2_new = f(x2[0], x2[8], x2[6]); wage parameters via j.
#     - both caches are period-scoped (cleared each period) because incomes
#       and repayment depend on the period.
# States are processed SORTED by the education key, so only the current
# payload per (j1, j2) group is held in memory. Set False to fall back to
# the Phase-1 per-choice computation (numbers are identical either way; the
# only observable difference is the key order inside the saved npz bundles).
ENABLE_FLOW_CACHE = True

# Phase 2 grouped kernel: run the windowed debt search for ALL fields of one
# (education, labor) group in a single Numba call. The CRRA flow utilities
# u(resources + candidate_debt) are identical across fields (only the
# continuation added on top differs), so the kernel computes each one once per
# (shock, current debt) row and reuses it. Every field keeps its own search
# window, previous-argmax state, fallbacks, and tie-breaking, replicated
# branch-for-branch from fused_debt_search — so results are bitwise identical.
# Off by default until test_fast_solver_equivalence.py (run with --grouped 1)
# passes on the server; then flip this to True to promote.
ENABLE_GROUPED_KERNEL = False


# ---------------------------------------------------------------------------
# Quadrature caches (identical values, computed once per process)
# ---------------------------------------------------------------------------

_WAGE_QUAD_CACHE = {}
_NO_WORK_NODES = (np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64))
_BUDGET_STANDARD_CACHE = {}
_EDU_JOINT_CACHE = {}


def _wage_quadrature(j, deg=5):
    """Cached twin of ``ms.get_quadrature_wage`` (same expressions, same values)."""
    if j[2] == 0:
        return _NO_WORK_NODES
    sigma = float(ms.get_sigma(j, ms.sigmas))
    key = (deg, sigma)
    if key not in _WAGE_QUAD_CACHE:
        x, w = scipy.special.roots_hermite(deg, mu=False)
        w = w * 1 / np.sqrt(np.pi)
        y = np.sqrt(2) * sigma * x + ms.mu
        _WAGE_QUAD_CACHE[key] = (y, w)
    return _WAGE_QUAD_CACHE[key]


def _budget_standard(deg_budget=5):
    """Cached standardized budget nodes, as built in ``get_expected_conditional``."""
    if deg_budget not in _BUDGET_STANDARD_CACHE:
        standard_nodes, wz = np.polynomial.hermite.hermgauss(deg_budget)
        standard_nodes = np.sqrt(2.0) * standard_nodes
        wz = wz / np.sqrt(np.pi)
        _BUDGET_STANDARD_CACHE[deg_budget] = (standard_nodes, wz)
    return _BUDGET_STANDARD_CACHE[deg_budget]


def _edu_joint_nodes(j, nb, deg=5, deg_budget=5):
    """Cached joint wage x budget node layout for one education choice.

    Returns (e_joint, z_standard_joint, w_joint, w_vis) exactly as assembled in
    ``get_expected_conditional`` (tile/repeat/kron in the same order).
    """
    sigma = None if j[2] == 0 else float(ms.get_sigma(j, ms.sigmas))
    key = (deg, deg_budget, sigma, nb)
    if key not in _EDU_JOINT_CACHE:
        e_nodes, we = _wage_quadrature(j, deg)
        standard_nodes, wz = _budget_standard(deg_budget)
        e_joint = np.tile(e_nodes, len(standard_nodes))
        z_standard_joint = np.repeat(standard_nodes, len(e_nodes))
        w_joint = np.kron(wz, we)
        w_vis = np.repeat(w_joint, nb)
        _EDU_JOINT_CACHE[key] = (e_joint, z_standard_joint, w_joint, w_vis)
    return _EDU_JOINT_CACHE[key]


# ---------------------------------------------------------------------------
# Exact debt snapping (replaces the full squared-distance matrix)
# ---------------------------------------------------------------------------

def snap_debt_indices(debt_grid, values):
    """Nearest grid index for each value; identical to ``map_debt_position``.

    ``np.argmin`` over squared distances returns the FIRST minimizer, i.e. the
    lower grid point on an exact midpoint tie. The sorted-grid version below
    reproduces that: ties go to the lower index.
    """
    n = debt_grid.shape[0]
    pos = np.searchsorted(debt_grid, values)
    lo = np.clip(pos - 1, 0, n - 1)
    hi = np.clip(pos, 0, n - 1)
    take_lo = (values - debt_grid[lo]) <= (debt_grid[hi] - values)
    return np.where(take_lo, lo, hi)


# ---------------------------------------------------------------------------
# Grouped education debt search (Phase 2 kernel)
# ---------------------------------------------------------------------------

@njit
def _grouped_education_search(sigma_u, next_debt_grid, resources,
                              continuations, lo_idx, hi_idx, cap_start,
                              payoffs):
    """Multi-field twin of ``get_maximum_loop_modified_resources_maxdebt``.

    ``continuations`` is (n_fields, n_debt); ``payoffs`` (n_fields, n_rows) is
    filled in place. Fields share ``resources`` and the borrowing bounds
    (which depend only on the education level and years enrolled), so the CRRA
    value ``u = _flow_utility(sigma_u, resource + next_debt_grid[k])`` is
    computed once per (shock, current-debt) row and candidate ``k``, then
    reused by every field. Each field keeps its own previous-argmax state and
    follows the original kernel's control flow branch for branch, so the
    selected debt choices and payoffs are bitwise identical to running the
    single-field kernel once per field.
    """
    nf = continuations.shape[0]
    ncont = continuations.shape[1]
    quadrature = int(resources.shape[0] / ncont)

    u_val = np.empty(ncont)
    u_ok = np.empty(ncont, dtype=np.bool_)
    amax = np.empty(nf, dtype=np.int64)

    for shock_index in range(quadrature):
        for f in range(nf):
            amax[f] = 0

        for it in range(ncont):
            row_idx = it + ncont * shock_index
            resource = resources[row_idx]
            for k in range(ncont):
                u_ok[k] = False
            row_feasible = -1  # shared lazily across fields (same bounds)

            if it >= cap_start:
                idx_use = lo_idx[it]
                if not u_ok[idx_use]:
                    u_val[idx_use] = _flow_utility(
                        sigma_u, resource + next_debt_grid[idx_use]
                    )
                    u_ok[idx_use] = True
                for f in range(nf):
                    payoffs[f, row_idx] = (
                        u_val[idx_use] + continuations[f, idx_use]
                    )
                continue

            lo = lo_idx[it]
            hi = hi_idx[it]

            for f in range(nf):
                if it == 0:
                    if row_feasible < 0:
                        row_feasible = 0
                        for candidate in range(lo, hi + 1):
                            if (resource + next_debt_grid[candidate]
                                    >= CONSUMPTION_FLOOR):
                                row_feasible += 1

                    if row_feasible == 0:
                        idx_use = hi
                        if not u_ok[idx_use]:
                            u_val[idx_use] = _flow_utility(
                                sigma_u, resource + next_debt_grid[idx_use]
                            )
                            u_ok[idx_use] = True
                        payoffs[f, row_idx] = (
                            u_val[idx_use] + continuations[f, idx_use]
                        )
                        amax[f] = idx_use
                        continue

                    firstbound = hi + 1 - row_feasible
                    best_index = firstbound
                    if not u_ok[firstbound]:
                        u_val[firstbound] = _flow_utility(
                            sigma_u, resource + next_debt_grid[firstbound]
                        )
                        u_ok[firstbound] = True
                    best_value = (
                        u_val[firstbound] + continuations[f, firstbound]
                    )
                    for candidate in range(firstbound + 1, hi + 1):
                        if not u_ok[candidate]:
                            u_val[candidate] = _flow_utility(
                                sigma_u, resource + next_debt_grid[candidate]
                            )
                            u_ok[candidate] = True
                        value = u_val[candidate] + continuations[f, candidate]
                        if value > best_value:
                            best_value = value
                            best_index = candidate

                    amax[f] = best_index
                    payoffs[f, row_idx] = best_value
                    continue

                # Local window around the previous argmax, clipped as in the
                # original kernel.
                bound_left = max(amax[f] - 10, lo)
                bound_left = max(bound_left, it)
                bound_right = min(bound_left + 20, hi + 1)

                if bound_right <= bound_left:
                    bound_left = max(lo, it)
                    bound_right = hi + 1

                if resource + next_debt_grid[bound_left] < CONSUMPTION_FLOOR:
                    if row_feasible < 0:
                        row_feasible = 0
                        for candidate in range(lo, hi + 1):
                            if (resource + next_debt_grid[candidate]
                                    >= CONSUMPTION_FLOOR):
                                row_feasible += 1

                    if row_feasible == 0:
                        idx_use = hi
                        if not u_ok[idx_use]:
                            u_val[idx_use] = _flow_utility(
                                sigma_u, resource + next_debt_grid[idx_use]
                            )
                            u_ok[idx_use] = True
                        payoffs[f, row_idx] = (
                            u_val[idx_use] + continuations[f, idx_use]
                        )
                        amax[f] = idx_use
                        continue

                    bound_left = hi + 1 - row_feasible
                    bound_left = max(bound_left, lo)
                    bound_left = max(bound_left, it)
                    bound_right = hi + 1

                best_index = bound_left
                if not u_ok[bound_left]:
                    u_val[bound_left] = _flow_utility(
                        sigma_u, resource + next_debt_grid[bound_left]
                    )
                    u_ok[bound_left] = True
                best_value = u_val[bound_left] + continuations[f, bound_left]
                for candidate in range(bound_left + 1, bound_right):
                    if not u_ok[candidate]:
                        u_val[candidate] = _flow_utility(
                            sigma_u, resource + next_debt_grid[candidate]
                        )
                        u_ok[candidate] = True
                    value = u_val[candidate] + continuations[f, candidate]
                    if value > best_value:
                        best_value = value
                        best_index = candidate

                # A maximum at the window's right boundary does not update the
                # center used for the next current-debt state (legacy rule).
                if best_index != bound_right - 1:
                    amax_old = amax[f]
                    amax[f] = best_index

                    if (amax[f] - amax_old) > 9:
                        if row_feasible < 0:
                            row_feasible = 0
                            for candidate in range(lo, hi + 1):
                                if (resource + next_debt_grid[candidate]
                                        >= CONSUMPTION_FLOOR):
                                    row_feasible += 1

                        if row_feasible == 0:
                            idx_use = hi
                            if not u_ok[idx_use]:
                                u_val[idx_use] = _flow_utility(
                                    sigma_u,
                                    resource + next_debt_grid[idx_use],
                                )
                                u_ok[idx_use] = True
                            payoffs[f, row_idx] = (
                                u_val[idx_use] + continuations[f, idx_use]
                            )
                            amax[f] = idx_use
                            continue

                        firstbound = hi + 1 - row_feasible
                        firstbound = max(firstbound, lo)
                        firstbound = max(firstbound, it)

                        best_index = firstbound
                        if not u_ok[firstbound]:
                            u_val[firstbound] = _flow_utility(
                                sigma_u, resource + next_debt_grid[firstbound]
                            )
                            u_ok[firstbound] = True
                        best_value = (
                            u_val[firstbound] + continuations[f, firstbound]
                        )
                        for candidate in range(firstbound + 1, hi + 1):
                            if not u_ok[candidate]:
                                u_val[candidate] = _flow_utility(
                                    sigma_u,
                                    resource + next_debt_grid[candidate],
                                )
                                u_ok[candidate] = True
                            value = (
                                u_val[candidate] + continuations[f, candidate]
                            )
                            if value > best_value:
                                best_value = value
                                best_index = candidate
                        amax[f] = best_index

                payoffs[f, row_idx] = best_value


# ---------------------------------------------------------------------------
# Per-period static structure (parameter-free; shared by all tasks/iterations)
# ---------------------------------------------------------------------------

class PeriodStatics:
    """Everything about a period's states that does not depend on parameters."""

    __slots__ = (
        "period", "n_states", "x2_raw", "x2_int", "x2_new", "x2_str",
        "Jx", "g_idx", "x_change", "x_educ",
        "grad_flag", "succ_nograd", "succ_grad",
        "edu_key", "work_key", "solve_order",
        "edu_groups", "noneduc_idx",
    )


_PERIOD_STATICS = {}


def _state_index_map(states_int):
    return {states_int[k].tobytes(): k for k in range(states_int.shape[0])}


def get_period_statics(period):
    """Build (once per process) the static structure for one period.

    For period T only the state arrays/names are needed; for earlier periods we
    also precompute feasible choices, the g() design pieces, the graduation
    flag of ``VT``'s expectation branch, and integer successor indices into the
    NEXT period's state ordering (the row order of ``states_t{t+1}.npy``, which
    is exactly the order the fast solver stores EVT in).
    """
    if period in _PERIOD_STATICS:
        return _PERIOD_STATICS[period]

    st = PeriodStatics()
    st.period = period
    x2_raw = ms.get_x2(period)
    st.x2_raw = x2_raw
    st.n_states = x2_raw.shape[0]
    st.x2_int = np.ascontiguousarray(x2_raw.astype(np.int64))
    # x2_new is built from the RAW row, exactly as loop_rows does before the
    # int conversion (get_x2_new converts internally).
    st.x2_new = [ms.get_x2_new(x2_raw[s]) for s in range(st.n_states)]
    st.x2_str = [f"{x2_raw[s].astype(int)}" for s in range(st.n_states)]

    # Flow-cache keys (Phase 2b) and the education-key-sorted solve order.
    st.edu_key = [st.x2_int[s, :9].tobytes() for s in range(st.n_states)]
    st.work_key = [st.x2_int[s, [0, 6, 7, 8]].tobytes()
                   for s in range(st.n_states)]
    st.solve_order = sorted(range(st.n_states), key=st.edu_key.__getitem__)

    if period == T:
        st.Jx = st.g_idx = st.x_change = st.x_educ = None
        st.grad_flag = st.succ_nograd = st.succ_grad = None
        st.edu_groups = st.noneduc_idx = None
        _PERIOD_STATICS[period] = st
        return st

    next_int = np.ascontiguousarray(ms.get_x2(period + 1).astype(np.int64))
    next_index = _state_index_map(next_int)
    total_choices = ms.get_total_choices()

    st.Jx, st.g_idx, st.x_change, st.x_educ = [], [], [], []
    st.grad_flag, st.succ_nograd, st.succ_grad = [], [], []
    st.edu_groups, st.noneduc_idx = [], []
    for s in range(st.n_states):
        x2 = st.x2_int[s]
        Jx = ms.get_possible_choices(x2)
        nJ = Jx.shape[0]
        # Choice-row indices into get_total_choices(), as in get_all_g.
        idx = np.where((total_choices == Jx[:, None]).all(-1))[1]
        flags = np.zeros(nJ, dtype=bool)
        succ_ng = np.empty(nJ, dtype=np.int64)
        succ_g = np.full(nJ, -1, dtype=np.int64)
        for c in range(nJ):
            j = Jx[c]
            # VT's graduation-expectation condition, verbatim.
            flags[c] = (
                ((x2[1] >= 1) & (j[1] == 1) & (x2[4] == 0))
                | ((x2[2] >= 3) & (j[1] == 2) & (x2[5] == 0))
                | (j[1] == 3)
            )
            z_ng = np.asarray(ms.move_state_grad(x2, j, period), dtype=np.int64)
            try:
                succ_ng[c] = next_index[z_ng.tobytes()]
            except KeyError as exc:
                raise KeyError(
                    f"period {period}: no-grad successor {z_ng} of state {x2} "
                    f"(choice {j}) not found in states_t{period + 1}"
                ) from exc
            if flags[c]:
                z_g = np.asarray(
                    ms.move_state_grad(x2, j, period, grad=1), dtype=np.int64
                )
                try:
                    succ_g[c] = next_index[z_g.tobytes()]
                except KeyError as exc:
                    raise KeyError(
                        f"period {period}: grad successor {z_g} of state {x2} "
                        f"(choice {j}) not found in states_t{period + 1}"
                    ) from exc
        st.Jx.append(Jx)
        st.g_idx.append(idx)
        st.x_change.append(ms.get_x_change(x2, period))
        st.x_educ.append(ms.get_x_educ(x2, period))
        st.grad_flag.append(flags)
        st.succ_nograd.append(succ_ng)
        st.succ_grad.append(succ_g)
        # Choice indices grouped for the grouped kernel: work/home choices,
        # and education choices bundled by (education level, labor supply).
        groups = {}
        noneduc = []
        for c in range(nJ):
            if Jx[c][1] == 0:
                noneduc.append(c)
            else:
                groups.setdefault(
                    (int(Jx[c][1]), int(Jx[c][2])), []
                ).append(c)
        st.noneduc_idx.append(np.array(noneduc, dtype=np.int64))
        st.edu_groups.append(
            [np.array(v, dtype=np.int64) for v in groups.values()]
        )

    _PERIOD_STATICS[period] = st
    return st


def build_all_period_statics():
    """Prebuild statics for all periods (call in the parent before forking a
    pool on Linux so workers inherit them copy-on-write)."""
    for period in range(1, T + 1):
        get_period_statics(period)


# ---------------------------------------------------------------------------
# Deterministic utility g(), with the per-state designs precomputed
# ---------------------------------------------------------------------------

def _get_all_g_fast(utility_parameters, inv, x1_new, st, s, period):
    """Line-for-line twin of ``ms.get_all_g`` using precomputed idx/x_change/
    x_educ; the cheap per-state pieces still call the original helpers."""
    param_g = utility_parameters[0]
    param_g_work = utility_parameters[1]
    param_g_last = utility_parameters[2]
    param_g_educ = utility_parameters[3]
    param_g_period = utility_parameters[4]
    param_g_period_work = utility_parameters[5]
    param_g_first = utility_parameters[6]
    param_g_first_2 = param_g_first[0]
    param_g_first_4 = param_g_first[1]
    param_g_first_grad = param_g_first[2]
    param_g_exp = utility_parameters[7]
    param_g_type = utility_parameters[8]

    idx = st.g_idx[s]
    x2 = st.x2_int[s]

    g_x1 = x1_new @ param_g[idx, :].T
    g_work = param_g_work[0, idx]
    g_change = st.x_change[s] @ param_g_last[idx, :].T
    g_educ = st.x_educ[s] @ param_g_educ[idx, :].T
    g_period = param_g_period[period - 1, idx]
    g_period_work = param_g_period_work[period - 1, idx]

    x_afqt = ms.get_x_afqt_first(inv, x2, period, 1)
    g_first2 = x_afqt[0] * param_g_first_2[idx, 0].T
    x_afqt = ms.get_x_afqt_first(inv, x2, period, 2)
    g_first4 = x_afqt @ param_g_first_4[idx, :].T
    x_afqt = ms.get_x_afqt_first(inv, x2, period, 3)
    g_firstgrad = x_afqt * param_g_first_grad[idx, :].T

    x_exp = ms.get_x_exp(inv, x2)
    g_exp = x_exp @ param_g_exp[idx, :].T
    g_type = param_g_type[idx, 0]

    g = (
        g_x1 + g_work + g_change + g_educ + g_period + g_period_work
        + g_first2 + g_first4 + g_firstgrad + g_exp + g_type
    )
    return g


# ---------------------------------------------------------------------------
# One state's Bellman step (mirrors get_all_choices + get_expected_conditional
# + get_conditional, with dense-EVT continuation lookups)
# ---------------------------------------------------------------------------

def _noneduc_value(j, s, c, st, period, x1_new, x2, x2_new, sigma_u, debt_pen,
                   b, evt_next, conterfactual, task_cache, mask_current_debt):
    """Integrated value of one work/home choice (one column of all_vjt)."""
    nb = b.shape[0]
    e_nodes, we = _wage_quadrature(j)
    work_key = (j.tobytes(), st.work_key[s])
    payload = task_cache["work_flow"].get(work_key) \
        if ENABLE_FLOW_CACHE else None
    if payload is None:
        if j[2] != 0:
            param_wage_j = ms.get_params_wage(j)
            debtnew, income = ms.get_debt_income(
                x1_new, x2_new, x2, period, j, b, e_nodes,
                conterfactual, param_wage_j,
            )
        else:
            debtnew, income = ms.get_debt_income_home(
                x1_new, x2_new, x2, period, j, b, e_nodes, conterfactual
            )
        debt_position = snap_debt_indices(b, debtnew)
        u = ms.get_power_utility(sigma_u, income)
        if ENABLE_FLOW_CACHE:
            task_cache["work_flow"][work_key] = (debt_position, u)
    else:
        debt_position, u = payload
    evt_row = evt_next[st.succ_nograd[s][c]]
    continuation = ms.beta * evt_row[debt_position]
    vjt = u + continuation
    vjt += debt_pen * np.tile(mask_current_debt, e_nodes.shape[0])
    w_vis = np.repeat(we, nb)
    v = (vjt * w_vis).reshape((len(e_nodes), nb)).T
    return np.sum(v, axis=1)


def _edu_payload(j, s, st, period, inv, x1_new, x2, x2_new, b,
                 financial_parameters, task_cache, e_joint, z_standard_joint):
    """(z_joint, resources) for one (education, labor) group at one state.

    Field-independent: within a state, all fields of one group share it, and
    across states it depends only on x2[0:9] (see cache-key rule above).
    """
    group = (int(j[1]), int(j[2]))
    entry = task_cache["edu_flow"].get(group) if ENABLE_FLOW_CACHE else None
    if entry is not None and entry[0] == st.edu_key[s]:
        return entry[1], entry[2]

    if group not in task_cache["fin_help"]:
        task_cache["fin_help"][group] = float(
            ms.fin_help(x1_new, j, financial_parameters)
        )
    h0 = task_cache["fin_help"][group]

    w0_key = st.edu_key[s]
    if w0_key not in task_cache["wage0"]:
        # Quirk replicated from get_expected_conditional:
        # RAW x2 here (columns 0-8), not x2_new.
        task_cache["wage0"][w0_key] = float(
            np.asarray(ms.wage0(x1_new, x2)).reshape(-1)[0]
        )
    wage_index = task_cache["wage0"][w0_key]

    real_wage = (
        np.exp(wage_index + e_joint) * (j[2] / 2.0) * 52.0 * 40.0
    )
    pre_choice_resources = (
        h0 + real_wage[:, None] - ms.tuition(j)
        - (1.0 + ms.r) * b[None, :]
    )
    z_joint = bs.realization(
        ms.budget_params,
        inv,
        period,
        z_standard_joint[:, None],
        education=int(j[1]),
        state=x2,
        pre_choice_resources=pre_choice_resources,
    ).reshape(-1)

    resources = ms.get_consumption_resources(
        x1_new, x2_new, b, e_joint, j, financial_parameters, z=z_joint
    )
    if ENABLE_FLOW_CACHE:
        task_cache["edu_flow"][group] = (st.edu_key[s], z_joint, resources)
    return z_joint, resources


def _edu_continuation(j, s, c, st, x1_new, x2, evt_next, debt_pen,
                      mask_candidate_debt, task_cache):
    """(nb, 1) continuation of one schooling alternative: graduation-risk
    mixing plus the candidate-debt penalty, verbatim from get_conditional."""
    if st.grad_flag[s][c]:
        pg_key = (j.tobytes(), st.edu_key[s])
        if pg_key not in task_cache["pgrad"]:
            task_cache["pgrad"][pg_key] = ms.probability_graduation(
                x1_new, x2, j
            )
        p_grad = task_cache["pgrad"][pg_key]
        evt_grad = evt_next[st.succ_grad[s][c]][:, None]
        evt_nograd = evt_next[st.succ_nograd[s][c]][:, None]
        continuation = ms.beta * (
            p_grad * evt_grad + (1 - p_grad) * evt_nograd
        )
    else:
        continuation = ms.beta * evt_next[st.succ_nograd[s][c]][:, None]
    continuation[:, 0] += debt_pen * mask_candidate_debt
    return continuation


def _solve_state(
    s, st, period, inv, x1_key, x1_new, sigma_u, debt_pen, b, b1, evt_next,
    ccp_real, models_npz, solution_mode, conterfactual, maxdebt,
    financial_parameters, utility_parameters, task_cache,
):
    x2 = st.x2_int[s]
    x2_new = st.x2_new[s]
    Jx = st.Jx[s]
    nb = b.shape[0]
    all_vjt = np.zeros((nb, Jx.shape[0]))
    mask_current_debt = (b > 0).astype(np.float64)
    mask_candidate_debt = (b1 > 0).astype(np.float64)

    use_grouped = (
        ENABLE_GROUPED_KERNEL and ms.USE_FUSED_CONSUMPTION_SEARCH and maxdebt
    )
    if use_grouped:
        # ---- work/home choices, one at a time (as before) ----
        for c in st.noneduc_idx[s]:
            all_vjt[:, c] = _noneduc_value(
                Jx[c], s, c, st, period, x1_new, x2, x2_new, sigma_u,
                debt_pen, b, evt_next, conterfactual, task_cache,
                mask_current_debt,
            )
        # ---- schooling: one kernel call per (education, labor) group ----
        for idxs in st.edu_groups[s]:
            j0 = Jx[idxs[0]]
            e_joint, z_standard_joint, w_joint, w_vis = \
                _edu_joint_nodes(j0, nb)
            z_joint, resources = _edu_payload(
                j0, s, st, period, inv, x1_new, x2, x2_new, b,
                financial_parameters, task_cache, e_joint, z_standard_joint,
            )
            nf = idxs.shape[0]
            continuations = np.empty((nf, nb))
            for fi in range(nf):
                continuations[fi, :] = _edu_continuation(
                    Jx[idxs[fi]], s, idxs[fi], st, x1_new, x2, evt_next,
                    debt_pen, mask_candidate_debt, task_cache,
                )[:, 0]
            # Bounds depend only on education level and years enrolled --
            # identical for every field in the group.
            lo_idx, hi_idx, cap_start = get_debt_region_bounds(b, x2, j0)
            payoffs = np.zeros((nf, resources.shape[0]))
            _grouped_education_search(
                sigma_u, b1, resources, continuations,
                lo_idx, hi_idx, cap_start, payoffs,
            )
            for fi in range(nf):
                v = (payoffs[fi] * w_vis).reshape((len(w_joint), nb)).T
                all_vjt[:, idxs[fi]] = np.sum(v, axis=1)
    else:
        for c in range(Jx.shape[0]):
            j = Jx[c]
            if j[1] == 0:
                all_vjt[:, c] = _noneduc_value(
                    j, s, c, st, period, x1_new, x2, x2_new, sigma_u,
                    debt_pen, b, evt_next, conterfactual, task_cache,
                    mask_current_debt,
                )
            else:
                e_joint, z_standard_joint, w_joint, w_vis = \
                    _edu_joint_nodes(j, nb)
                z_joint, resources = _edu_payload(
                    j, s, st, period, inv, x1_new, x2, x2_new, b,
                    financial_parameters, task_cache, e_joint,
                    z_standard_joint,
                )
                continuation = _edu_continuation(
                    j, s, c, st, x1_new, x2, evt_next, debt_pen,
                    mask_candidate_debt, task_cache,
                )
                if ms.USE_FUSED_CONSUMPTION_SEARCH and maxdebt:
                    max_vjt = ms.get_maximum_loop_modified_resources_maxdebt(
                        sigma_u, b, b1, resources, continuation, j, x2
                    )
                else:
                    c_mat = resources[..., np.newaxis] + b1
                    max_vjt = ms.get_maximum(
                        sigma_u, c_mat, continuation, inv, b, j, x2, maxdebt
                    )
                v = (max_vjt * w_vis).reshape((len(w_joint), nb)).T
                all_vjt[:, c] = np.sum(v, axis=1)

    # ---- expectation / CCP step, verbatim from get_all_choices ----
    base = -1
    if solution_mode == 0:
        if ccp_real == 0:
            ccp = models_npz[f"ccp_t{period}_{x1_key}_{st.x2_str[s]}"]
        else:
            g = _get_all_g_fast(utility_parameters, inv, x1_new, st, s, period)
            all_vjt_temp = all_vjt + g
            log_ccp = (
                all_vjt_temp[:, base]
                - scipy.special.logsumexp(all_vjt_temp, axis=1)
            )
            ccp = np.exp(log_ccp)
        vjt_ccp = all_vjt[:, base]
        if ccp_real == 1:
            evt_vec = vjt_ccp - log_ccp + ms.gamma
        else:
            evt_vec = vjt_ccp - np.log(ccp) + ms.gamma
    elif solution_mode == 1:
        g = _get_all_g_fast(utility_parameters, inv, x1_new, st, s, period)
        all_vjt = all_vjt + g
        evt_vec = np.log(np.exp(all_vjt).sum(axis=1)) + ms.gamma
        ccp = None

    return all_vjt, evt_vec, ccp


# ---------------------------------------------------------------------------
# Drop-in task solver
# ---------------------------------------------------------------------------

def get_all_evt_fast(i, x1, b, b1, ccp_real, utility_parameters, models,
                     solution_mode, conterfactual, em_type, maxdebt):
    """Drop-in fast twin of ``ms.get_all_evt`` (same signature, artifacts)."""
    financial_parameters = ms.get_type_financial_parameters(em_type)
    sigma_u = float(bs.risk_aversion(ms.budget_params, x1[i, :]))

    inv = x1[i, :][..., None].T
    x1_key = f"{inv}"
    x1_new = ms.get_x1_new(inv[0])
    debt_pen = float(x1_new @ ms.debt_pen_vec)
    task_cache = {
        "fin_help": {}, "wage0": {}, "pgrad": {},   # task-wide
        "work_flow": {}, "edu_flow": {},            # cleared every period
    }

    task_started = time.perf_counter()
    total_states = len(x1)
    total_tasks = ms.N_TYPES * total_states
    task_number = (em_type - 1) * total_states + i + 1
    if ccp_real == 1:
        ccp_mode = "updated"
    elif ccp_real == 0:
        ccp_mode = "initial"
    else:
        ccp_mode = "supplied"

    nb = b.shape[0]
    evt_next = None  # dense (n_states_{period+1}, nb) EVT of the next period

    if conterfactual == 1:
        terminal_conter = np.load(
            f"{ms.pathcontfinal}/continuation_conter_s{inv[0, 2]}_eth{inv[0, 3]}_sigma{sigma_u}.npz"
        )

    for period in range(T, 0, -1):
        period_started = time.perf_counter()
        st = get_period_statics(period)
        task_cache["work_flow"].clear()
        task_cache["edu_flow"].clear()
        evt_arr = np.empty((st.n_states, nb))
        names_vjt, names_exp, names_ccp = [], [], []
        result_vjt, result_exp, results_ccp = [], [], []

        if period < T:
            # Same load (and failure mode) as the original solver; consumed
            # only when ccp_real == 0.
            models_npz = np.load(
                f"{ms.pathout}/ccp/{period}/ccp_t{period}_[{x1[i, :]}]_em{em_type}.npz"
            )
        else:
            models_npz = None

        # Sorted iteration keeps at most one live education payload per
        # (education, labor) group; npz bundles are keyed, so order is
        # irrelevant to every consumer.
        state_iter = st.solve_order if (period < T and ENABLE_FLOW_CACHE) \
            else range(st.n_states)
        for s in state_iter:
            name_suffix = f"{inv}_{st.x2_str[s]}"
            if period == T:
                x2 = st.x2_int[s]
                if conterfactual == 1:
                    vt = ms.get_terminal_pandas(
                        terminal_conter, inv, x2, sigma_u, conterfactual
                    )[..., None]
                else:
                    vt = ms.terminal_from_interp(inv, x2, sigma_u, b)[..., None]
                names_exp.append(f"evt_t{period}_{name_suffix}")
                result_exp.append(vt)
                evt_arr[s] = vt[:, 0]
            else:
                all_vjt, evt_vec, ccp = _solve_state(
                    s, st, period, inv, x1_key, x1_new, sigma_u, debt_pen,
                    b, b1, evt_next, ccp_real, models_npz, solution_mode,
                    conterfactual, maxdebt, financial_parameters,
                    utility_parameters, task_cache,
                )
                names_exp.append(f"evt_t{period}_{name_suffix}")
                names_vjt.append(f"vjt_t{period}_{name_suffix}")
                names_ccp.append(f"ccp_t{period}_{name_suffix}")
                result_exp.append(evt_vec[..., np.newaxis])
                result_vjt.append(all_vjt)
                results_ccp.append(ccp)
                evt_arr[s] = evt_vec

        ms.persist_outputs_for_period(
            period=period,
            x1i=inv,
            em_type=em_type,
            solution_mode=solution_mode,
            conterfactual=conterfactual,
            maxdebt=maxdebt,
            save_evt=1,
            names_vjt=names_vjt,
            result_vjt=result_vjt,
            names_exp=names_exp,
            result_exp=result_exp,
            save_fn=ms.save_npz_here,
        )
        if ccp_real == 1 and period < T:
            ms.save_npz_here(
                f"ccp/{period}/ccp_t{period}_{inv}_em{em_type}.npz",
                names_ccp,
                results_ccp,
                compressed=True,
            )

        evt_next = evt_arr
        print(
            f"[FastBellman | pid={os.getpid()} | task={task_number}/{total_tasks} "
            f"| type={em_type}/{ms.N_TYPES}:{ms.TYPE_NAMES[em_type - 1]} "
            f"| state={i + 1}/{total_states} | CCP={ccp_mode}] "
            f"completed period {period}/{T} | period={time.perf_counter() - period_started:.2f}s "
            f"| task={time.perf_counter() - task_started:.2f}s",
            flush=True,
        )
