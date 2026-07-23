# -*- coding: utf-8 -*-
"""Equivalence test: model_solution_fast vs model_solution_em.

Two layers:

1. UNIT CHECKS (run anywhere the module imports, including locally):
       python test_fast_solver_equivalence.py --units
   Verifies, against the original implementations: debt snapping (incl. tie
   and boundary behavior), quadrature caches, per-period static structure
   (feasible choices, successors, g() designs), and — when utility parameters
   are available — the fast g() against ``get_all_g``.

2. FULL-TASK EQUIVALENCE (run ON THE SERVER; needs auxiliary_em_results.npz,
   budget estimates, initial CCPs, and the terminal interp cache):
       python test_fast_solver_equivalence.py --em-types 2 --states 0 --ccp-real 0
       python test_fast_solver_equivalence.py --em-types 2,9 --states 0,63 --ccp-real 1
   For each (type, invariant state) it runs the ORIGINAL solver, snapshots
   every artifact (VJT/EVT/CCP for all periods) into memory, runs the FAST
   solver (which overwrites the same files), and compares entry by entry.
   Default tolerance is 0.0 (bitwise); use --tol to relax for inspection.

Once this passes for a representative set of tasks (recommended: both an
early and a late invariant state, several types, ccp-real 0 AND 1), flip
``USE_FAST_SOLVER = True`` in estimation_all_em.py.

Note: with --ccp-real 1 the solvers write ccp/{t} files but never read them
back within the run (they are consumed only at iteration 0), so the test does
not contaminate its own inputs. After a PASSING run the on-disk artifacts are
bitwise identical to the reference. After a FAILING run they are the fast
solver's — pass --restore to re-run the reference solver at the end.
"""

import argparse
import sys

import numpy as np

import model_solution_em as ms
import model_solution_fast as msf


# ---------------------------------------------------------------------------
# Unit checks
# ---------------------------------------------------------------------------

def unit_check_snapping(rng):
    grid = ms.debt_range
    vals = np.concatenate([
        rng.uniform(-1000.0, 120000.0, 5000),
        grid,                                   # exact hits
        (grid[:-1] + grid[1:]) / 2.0,           # midpoints (tie -> lower index)
        np.array([0.0, grid[-1] + 1e6]),        # boundaries
    ])
    ref = ms.map_debt_position(grid, vals)
    fast = msf.snap_debt_indices(grid, vals)
    assert np.array_equal(ref, fast), "debt snapping differs from map_debt_position"
    print(f"  snapping: OK ({vals.size} values, incl. ties and boundaries)")


def unit_check_quadrature():
    test_choices = [
        np.array([0, 0, 0]),    # home
        np.array([1, 0, 1]),    # work part-time, occ 1
        np.array([9, 0, 2]),    # work full-time, high occ index
        np.array([12, 1, 2]),   # two-year, full-time work
        np.array([3, 2, 0]),    # four-year field, no work
        np.array([13, 3, 1]),   # grad school, part-time
    ]
    for j in test_choices:
        y_ref, w_ref = ms.get_quadrature_wage(5, ms.mu, j)
        y_fast, w_fast = msf._wage_quadrature(j)
        assert np.array_equal(y_ref, y_fast) and np.array_equal(w_ref, w_fast), \
            f"wage quadrature differs for j={j}"
    # Joint education layout vs the expressions in get_expected_conditional.
    nb = ms.debt_range.shape[0]
    for j in [np.array([12, 1, 2]), np.array([3, 2, 0]), np.array([13, 3, 1])]:
        e_nodes, we = ms.get_quadrature_wage(5, ms.mu, j)
        standard_nodes, wz = np.polynomial.hermite.hermgauss(5)
        standard_nodes = np.sqrt(2.0) * standard_nodes
        wz = wz / np.sqrt(np.pi)
        e_ref = np.tile(e_nodes, len(standard_nodes))
        z_ref = np.repeat(standard_nodes, len(e_nodes))
        w_ref = np.kron(wz, we)
        e_f, z_f, w_f, w_vis = msf._edu_joint_nodes(j, nb)
        assert np.array_equal(e_ref, e_f) and np.array_equal(z_ref, z_f) \
            and np.array_equal(w_ref, w_f) \
            and np.array_equal(np.repeat(w_ref, nb), w_vis), \
            f"joint education nodes differ for j={j}"
    print("  quadrature caches: OK")


def unit_check_statics(rng, periods=(2, 5, 9), sample=200):
    for period in periods:
        st = msf.get_period_statics(period)
        nxt = ms.get_x2(period + 1).astype(np.int64)
        idx_pick = rng.choice(st.n_states, size=min(sample, st.n_states),
                              replace=False)
        total_choices = ms.get_total_choices()
        for s in idx_pick:
            x2 = st.x2_int[s]
            Jx_ref = ms.get_possible_choices(x2)
            assert np.array_equal(Jx_ref, st.Jx[s]), f"Jx differs t={period} s={s}"
            g_idx_ref = np.where((total_choices == Jx_ref[:, None]).all(-1))[1]
            assert np.array_equal(g_idx_ref, st.g_idx[s]), \
                f"g idx differs t={period} s={s}"
            assert np.array_equal(ms.get_x_change(x2, period), st.x_change[s])
            assert np.array_equal(ms.get_x_educ(x2, period), st.x_educ[s])
            assert st.x2_str[s] == f"{st.x2_raw[s].astype(int)}"
            for c in range(Jx_ref.shape[0]):
                j = Jx_ref[c]
                z_ng = np.asarray(ms.move_state_grad(x2, j, period),
                                  dtype=np.int64)
                assert np.array_equal(nxt[st.succ_nograd[s][c]], z_ng), \
                    f"no-grad successor differs t={period} s={s} c={c}"
                flag_ref = bool(
                    ((x2[1] >= 1) & (j[1] == 1) & (x2[4] == 0))
                    | ((x2[2] >= 3) & (j[1] == 2) & (x2[5] == 0))
                    | (j[1] == 3)
                )
                assert flag_ref == bool(st.grad_flag[s][c])
                if flag_ref:
                    z_g = np.asarray(
                        ms.move_state_grad(x2, j, period, grad=1),
                        dtype=np.int64,
                    )
                    assert np.array_equal(nxt[st.succ_grad[s][c]], z_g), \
                        f"grad successor differs t={period} s={s} c={c}"
        print(f"  statics t={period}: OK ({len(idx_pick)} states checked)")


def unit_check_g(rng, em_type=2, periods=(2, 5, 9), sample=50):
    try:
        utility_parameters = ms.load_param_g(em_type, real=0)
    except Exception as exc:  # noqa: BLE001 - inputs may be server-only
        print(f"  g(): SKIPPED (utility parameters unavailable here: {exc})")
        return
    x1 = ms.invariant_states
    for period in periods:
        st = msf.get_period_statics(period)
        idx_pick = rng.choice(st.n_states, size=min(sample, st.n_states),
                              replace=False)
        for i in (0, len(x1) - 1):
            inv = x1[i, :][..., None].T
            x1_new = ms.get_x1_new(inv[0])
            for s in idx_pick:
                g_ref = ms.get_all_g(utility_parameters, inv, x1_new,
                                     st.x2_int[s], st.Jx[s], period)
                g_fast = msf._get_all_g_fast(utility_parameters, inv, x1_new,
                                             st, s, period)
                assert np.array_equal(np.asarray(g_ref), np.asarray(g_fast)), \
                    f"g differs t={period} s={s} inv={i}"
    print("  g(): OK")


def unit_check_flow_keys(rng, periods=(5, 9), pairs=40):
    """Key sufficiency: states sharing a flow key must produce identical flow
    objects. Checks the work/home payload directly (runnable locally); the
    education payload is certified end-to-end by the server bitwise test."""
    x1 = ms.invariant_states
    inv = x1[0, :][..., None].T
    x1_new = ms.get_x1_new(inv[0])
    b = ms.debt_range
    test_js = [np.array([1, 0, 2]), np.array([2, 0, 1]), np.array([0, 0, 0])]
    for period in periods:
        st = msf.get_period_statics(period)
        assert sorted(st.solve_order) == list(range(st.n_states)), \
            "solve_order is not a permutation"
        groups = {}
        for s in range(st.n_states):
            groups.setdefault(st.work_key[s], []).append(s)
        multi = [g for g in groups.values() if len(g) > 1]
        checked = 0
        for g in multi:
            if checked >= pairs:
                break
            s_a, s_b = g[0], g[rng.integers(1, len(g))]
            for j in test_js:
                e_nodes, we = msf._wage_quadrature(j)
                out = []
                for s in (s_a, s_b):
                    if j[2] != 0:
                        pw = ms.get_params_wage(j)
                        out.append(ms.get_debt_income(
                            x1_new, st.x2_new[s], st.x2_int[s], period, j,
                            b, e_nodes, 0, pw))
                    else:
                        out.append(ms.get_debt_income_home(
                            x1_new, st.x2_new[s], st.x2_int[s], period, j,
                            b, e_nodes, 0))
                assert np.array_equal(out[0][0], out[1][0]) \
                    and np.array_equal(out[0][1], out[1][1]), \
                    f"work flow key insufficient: t={period} states {s_a},{s_b} j={j}"
            checked += 1
        # Education wage quirk: same edu key -> same raw-x2 wage index.
        egroups = {}
        for s in range(st.n_states):
            egroups.setdefault(st.edu_key[s], []).append(s)
        for g in [g for g in egroups.values() if len(g) > 1][:pairs]:
            w = [float(np.asarray(ms.wage0(x1_new, st.x2_int[s])).reshape(-1)[0])
                 for s in (g[0], g[-1])]
            assert w[0] == w[1], f"edu flow key insufficient at t={period}"
        print(f"  flow keys t={period}: OK ({checked} work groups, "
              f"{len(multi)} shareable)")


def run_units():
    rng = np.random.default_rng(0)
    print("Unit checks (fast vs original implementations):")
    unit_check_snapping(rng)
    unit_check_quadrature()
    unit_check_statics(rng)
    unit_check_flow_keys(rng)
    unit_check_g(rng)
    print("All unit checks passed.")


# ---------------------------------------------------------------------------
# Full-task equivalence
# ---------------------------------------------------------------------------

def _artifact_paths(inv, em_type, ccp_real):
    """All artifact files one NPL-path task writes (solution_mode == 0)."""
    paths = []
    for t in range(1, ms.T + 1):
        if t < ms.T:
            paths.append(f"{ms.pathout}/vjt_nog/{t}/vjt_t{t}_{inv}_em{em_type}.npz")
        paths.append(f"{ms.pathout}/evt_nog/{t}/evt_t{t}_{inv}_em{em_type}.npz")
        if ccp_real == 1 and t < ms.T:
            paths.append(f"{ms.pathout}/ccp/{t}/ccp_t{t}_{inv}_em{em_type}.npz")
    return paths


def _load_artifacts(paths):
    out = {}
    for p in paths:
        with np.load(p) as z:
            out[p] = {k: z[k].copy() for k in z.files}
    return out


def _compare(ref, new, tol):
    n_arrays = 0
    n_exact = 0
    worst = 0.0
    worst_where = None
    for p in ref:
        assert set(ref[p]) == set(new[p]), f"key sets differ in {p}"
        for k in ref[p]:
            a, barr = ref[p][k], new[p][k]
            n_arrays += 1
            if a.shape != barr.shape:
                return False, f"SHAPE MISMATCH {p} :: {k}: {a.shape} vs {barr.shape}"
            if np.array_equal(a, barr):
                n_exact += 1
                continue
            d = float(np.nanmax(np.abs(a - barr)))
            if d > worst:
                worst, worst_where = d, f"{p} :: {k}"
    ok = worst <= tol
    msg = (f"{n_arrays} arrays compared, {n_exact} bitwise-equal, "
           f"max abs diff = {worst:.3e}"
           + (f" at {worst_where}" if worst_where else ""))
    return ok, msg


def _run_one_task(task):
    """Run reference + fast solver for one (type, state) task; compare & time."""
    em_type, i, ccp_real, solution_mode, params_mode, tol, restore = task
    import time

    x1 = ms.invariant_states
    b = ms.debt_range
    if params_mode == "npl":
        x0 = np.load(f"{ms.path_estimates}/param_g.npy")
        utility_parameters = ms.build_param_g(em_type, x0)
    else:
        utility_parameters = ms.load_param_g(em_type, real=0)

    inv = x1[i, :][..., None].T
    args = (i, x1, b, b, ccp_real, utility_parameters, 0,
            solution_mode, 0, em_type, True)

    t0 = time.perf_counter()
    ms.get_all_evt(*args)
    t_ref = time.perf_counter() - t0
    paths = _artifact_paths(inv, em_type, ccp_real)
    ref = _load_artifacts(paths)

    t0 = time.perf_counter()
    msf.get_all_evt_fast(*args)
    t_fast = time.perf_counter() - t0
    new = _load_artifacts(paths)

    ok, msg = _compare(ref, new, tol)
    if not ok and restore:
        print(f"[em{em_type} i={i}] restoring reference artifacts", flush=True)
        ms.get_all_evt(*args)
    return em_type, i, ok, msg, t_ref, t_fast


def _init_worker():
    ms.reload_budgetshock_params()


def run_full(em_types, states, ccp_real, solution_mode, params_mode, tol,
             restore, workers):
    import multiprocessing

    ms.reload_budgetshock_params()
    tasks = [
        (em_type, i, ccp_real, solution_mode, params_mode, tol, restore)
        for em_type in em_types
        for i in states
    ]
    print(f"Running {len(tasks)} task(s) with {workers} worker(s); each task "
          f"solves reference + fast sequentially and compares artifacts.")

    if workers > 1:
        # Prebuild the fast solver's statics so forked workers inherit them.
        msf.build_all_period_statics()
        with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(_run_one_task, tasks, chunksize=1)
    else:
        results = [_run_one_task(t) for t in tasks]

    print("\n===== SUMMARY =====")
    print(f"{'type':>5} {'state':>6} {'result':>7} {'ref [s]':>9} "
          f"{'fast [s]':>9} {'speedup':>8}   detail")
    overall_ok = True
    for em_type, i, ok, msg, t_ref, t_fast in results:
        overall_ok &= ok
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"{em_type:>5} {i:>6} {'PASS' if ok else 'FAIL':>7} "
              f"{t_ref:>9.1f} {t_fast:>9.1f} {speedup:>7.1f}x   {msg}")
    print("\n" + ("EQUIVALENCE TEST PASSED" if overall_ok
                  else "EQUIVALENCE TEST FAILED"))
    return overall_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", action="store_true",
                    help="run only the local unit checks")
    ap.add_argument("--em-types", default="2",
                    help="comma-separated joint types, e.g. 2,9")
    ap.add_argument("--states", default="0",
                    help="comma-separated invariant-state indices, e.g. 0,63")
    ap.add_argument("--ccp-real", type=int, default=0, choices=[0, 1])
    ap.add_argument("--solution-mode", type=int, default=0, choices=[0, 1])
    ap.add_argument("--params-mode", default="npl", choices=["npl", "solve"],
                    help="npl: build_param_g from saved param_g.npy (as the "
                         "NPL loop does); solve: load_param_g(real=0)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs difference allowed (default bitwise)")
    ap.add_argument("--restore", action="store_true",
                    help="re-run the reference solver after a failure so the "
                         "on-disk artifacts stay the production ones")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers over (type, state) tasks; each "
                         "task still runs its two solvers sequentially")
    args = ap.parse_args()

    if args.units:
        run_units()
        return

    em_types = [int(v) for v in args.em_types.split(",")]
    states = [int(v) for v in args.states.split(",")]
    ok = run_full(em_types, states, args.ccp_real, args.solution_mode,
                  args.params_mode, args.tol, args.restore, args.workers)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
