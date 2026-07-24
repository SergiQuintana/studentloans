# -*- coding: utf-8 -*-
"""Equivalence test: fast/dense CCP-sequence rewrite vs production code.

Covers points 3 and 4 of Agents_Readme/Tasks/
ESTIMATION_SPEED_ANALYSIS_2026_07_24.md. NOTHING under Model/ is written:
the original builder's saves are redirected to a temporary directory, the
fast builder writes to the same temporary tree, and part C round-trips
real bundles into the temp directory as well.

Three parts, each with a hard PASS/FAIL (process exit code 1 on any FAIL):

A. Builder equivalence (point 3). For every tested (invariant state,
   type) task, run the production ``get_ccp_sequence`` and the fast
   builder, then compare EVERY saved array of EVERY period: identical key
   sets, identical dtype, identical shape, identical BYTES
   (``a.tobytes() == b.tobytes()`` — stricter than np.array_equal: it
   distinguishes -0.0 from 0.0 and treats NaNs as equal only when their
   bit patterns are). The dense matrices are additionally compared
   row-by-row against the original per-state arrays.

B. Consumer equivalence (points 3+4). The production SMM reader
   ``model_fitloans_dynamic._continuation_from_bundle`` (graduation
   mixing included) is evaluated on the ORIGINAL bundles, and
   ``DenseSequenceReader`` on the dense files, over an exhaustive sweep of
   feasible choices for deterministically sampled states in every period
   1..8. Byte equality required. Skipped with a warning if
   model_fitloans_dynamic cannot be imported on this machine.

C. Storage round-trip on real artifacts (point 4). Real ccp / evt / vjt
   bundles from Model/Output are packed into one dense array (padded
   where per-state shapes differ, as in vjt), verified slice-by-slice
   against every original key (bytes), written as a dense npz, re-read,
   and verified again. Read/write timings and file sizes are reported.

D. Fused solver emission (--fused; server only, needs full solver inputs).
   Runs the fast solver with EMIT_CCP_SEQUENCE on (all writes redirected
   into the temp tree), then rebuilds the sequence from the CCP files that
   same solve wrote — with the standalone fast builder AND the original
   production builder — and requires all three to be byte-identical.
   This is the gate for USE_FAST_CCP_SEQUENCE in estimation_all_em.py.

Run locally (uses whichever em types exist under Model/Output/ccp):
    python test_ccp_sequence_fast_equivalence.py --tasks 2
Full server gate for the fast CCP-sequence pipeline:
    python test_ccp_sequence_fast_equivalence.py --tasks 20 --types all --fused
"""

import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

import model_solution_em as ms
import model_getccp_sequence as mgs
import model_getccp_sequence_fast as mgf
from latent_types import TYPE_IDS

T = 10
FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == b.tobytes()
    )


def report_mismatch(label, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        fail(f"{label}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        return
    with np.errstate(invalid="ignore"):
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        fail(f"{label}: bytes differ, max abs diff {np.nanmax(diff):.6e}")


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def available_types():
    found = []
    inv0 = ms.invariant_states[0, :]
    for em in TYPE_IDS:
        if all(
            os.path.exists(
                f"{mgs.path_out}/ccp/{p}/ccp_t{p}_[{inv0}]_em{em}.npz"
            )
            for p in range(1, T)
        ):
            found.append(em)
    return found


def complete_tasks(types):
    tasks = []
    for em in types:
        for i in range(ms.invariant_states.shape[0]):
            inv = ms.invariant_states[i, :]
            if all(
                os.path.exists(
                    f"{mgs.path_out}/ccp/{p}/ccp_t{p}_[{inv}]_em{em}.npz"
                )
                for p in range(1, T)
            ):
                tasks.append((i, em))
    return tasks


# ---------------------------------------------------------------------------
# Part A: builder equivalence
# ---------------------------------------------------------------------------

def redirected_writer(root):
    def writer(rel_path, names, arrays, compressed=True):
        full = os.path.join(root, *rel_path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        payload = {n: a for n, a in zip(names, arrays)}
        if compressed:
            np.savez_compressed(full, **payload)
        else:
            np.savez(full, **payload)
    return writer


def run_part_a(tasks, tmp):
    print("\n=== PART A: builder equivalence (original vs fast) ===")
    orig_root = os.path.join(tmp, "orig")
    fast_root = os.path.join(tmp, "fast")
    statics_started = time.perf_counter()
    mgf.build_sequence_statics()
    print(f"one-time statics: {time.perf_counter() - statics_started:.1f} s")

    arrays_compared = 0
    t_orig_total = t_fast_total = 0.0
    original_writer = mgs.save_npz_here
    try:
        mgs.save_npz_here = redirected_writer(orig_root)
        for i, em in tasks:
            inv = ms.invariant_states[i, :]
            label = f"task(i={i}, em={em})"

            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                mgs.get_ccp_sequence(
                    i, ms.invariant_states, ms.debt_range, em
                )
            t_orig = time.perf_counter() - started

            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                dense = mgf.get_ccp_sequence_fast(
                    i, ms.invariant_states, ms.debt_range, em,
                    write_mode="both", out_root=fast_root,
                    return_dense=True,
                )
            t_fast = time.perf_counter() - started
            t_orig_total += t_orig
            t_fast_total += t_fast

            statics = mgf.build_sequence_statics()
            for p in range(1, T):
                rel = f"evt_ccp/{p}/evt_ccp_sequence_t{p}_{inv}_em{em}.npz"
                with np.load(os.path.join(orig_root, *rel.split("/"))) as fo, \
                        np.load(os.path.join(fast_root, *rel.split("/"))) as ff:
                    keys_o = sorted(fo.files)
                    keys_f = sorted(ff.files)
                    if keys_o != keys_f:
                        fail(f"{label} p{p}: key sets differ "
                             f"({len(keys_o)} vs {len(keys_f)})")
                        continue
                    for key in keys_o:
                        a, b = fo[key], ff[key]
                        if not bitwise_equal(a, b):
                            report_mismatch(f"{label} p{p} {key} (legacy)", a, b)
                        arrays_compared += 1
                    # dense rows against the original per-state arrays
                    strs = statics["x2_strings"][p]
                    for k, s in enumerate(strs):
                        key = f"evt_ccp_sequence_t{p}_{inv}_{s}"
                        a = fo[key]
                        b = dense[p][k]
                        if not bitwise_equal(a, b):
                            report_mismatch(f"{label} p{p} row {k} (dense)", a, b)
                        arrays_compared += 1
            print(f"{label}: original {t_orig:6.1f} s | fast {t_fast:5.1f} s "
                  f"| speedup {t_orig / max(t_fast, 1e-9):5.1f}x")
    finally:
        mgs.save_npz_here = original_writer

    print(f"PART A: {arrays_compared:,} arrays compared "
          f"({len(tasks)} tasks); original {t_orig_total:.1f} s total, "
          f"fast {t_fast_total:.1f} s total")
    return orig_root, fast_root


# ---------------------------------------------------------------------------
# Part B: consumer equivalence
# ---------------------------------------------------------------------------

def run_part_b(tasks, orig_root, fast_root, states_per_period, seed):
    print("\n=== PART B: consumer equivalence "
          "(production bundle reader vs dense reader) ===")
    try:
        import model_fitloans_dynamic as mfd
    except Exception as exc:  # noqa: BLE001 - report and skip
        print(f"  SKIPPED: cannot import model_fitloans_dynamic here ({exc}).")
        print("  Run this part on the server, where all SMM inputs exist.")
        return

    statics = mgf.build_sequence_statics()
    rng = np.random.default_rng(seed)
    comparisons = 0
    grad_branch = 0
    matrix_rows = 0
    dense_root = os.path.join(fast_root, "evt_ccp_dense")

    # Redirect BOTH production paths at the temp trees for the duration:
    # the legacy bundle path and the dense root.
    bundle_path_before = mfd._ccp_bundle_path
    dense_root_before = mgf.DENSE_ROOT
    format_before = mfd.CCP_SEQUENCE_FORMAT

    def temp_bundle_path(period, x1i, em_type):
        return Path(os.path.join(
            orig_root, "evt_ccp", str(period + 1),
            f"evt_ccp_sequence_t{period + 1}_{x1i}_em{em_type}.npz",
        ))

    try:
        mfd._ccp_bundle_path = temp_bundle_path
        mgf.DENSE_ROOT = dense_root

        for i, em in tasks:
            inv = ms.invariant_states[i, :].astype(np.int64)
            for period in range(1, T - 1):
                sp = statics["states"][period]
                n = sp.shape[0]
                take = min(states_per_period, n)
                rows = rng.choice(n, size=take, replace=False)
                reader = mgf.DenseSequenceReader(
                    period, inv, em, root=dense_root
                )
                sampled_states, sampled_choices = [], []
                rel = (f"evt_ccp/{period + 1}/"
                       f"evt_ccp_sequence_t{period + 1}_{inv}_em{em}.npz")
                with np.load(os.path.join(orig_root, *rel.split("/"))) as bundle:
                    for k in rows:
                        x2i = sp[k]
                        for ji in ms.get_possible_choices(x2i):
                            old = mfd._continuation_from_bundle(
                                inv, x2i, ji, period, bundle
                            )
                            new = reader.continuation(x2i, ji)
                            if not bitwise_equal(old, new):
                                report_mismatch(
                                    f"consumer task(i={i},em={em}) p{period} "
                                    f"state{k} choice{np.asarray(ji).tolist()}",
                                    old, new,
                                )
                            comparisons += 1
                            sampled_states.append(x2i)
                            sampled_choices.append(np.asarray(ji))
                            if (
                                ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0))
                                | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0))
                                | (ji[1] == 3)
                            ):
                                grad_branch += 1

                # Same lookups through the PRODUCTION loader, both formats.
                x1_mat = np.tile(inv, (len(sampled_states), 1))
                state_mat = np.asarray(sampled_states)
                choice_mat = np.asarray(sampled_choices)
                mfd.CCP_SEQUENCE_FORMAT = "legacy"
                legacy_path_matrix = mfd.load_ccp_path(
                    x1_mat, state_mat, choice_mat, period, em
                )
                mfd.CCP_SEQUENCE_FORMAT = "dense"
                dense_path_matrix = mfd.load_ccp_path(
                    x1_mat, state_mat, choice_mat, period, em
                )
                if not bitwise_equal(legacy_path_matrix, dense_path_matrix):
                    report_mismatch(
                        f"load_ccp_path task(i={i},em={em}) p{period}",
                        legacy_path_matrix, dense_path_matrix,
                    )
                matrix_rows += state_mat.shape[0]
    finally:
        mfd._ccp_bundle_path = bundle_path_before
        mgf.DENSE_ROOT = dense_root_before
        mfd.CCP_SEQUENCE_FORMAT = format_before

    print(f"PART B: {comparisons:,} continuation lookups compared "
          f"({grad_branch:,} exercised the graduation-mixing branch); "
          f"production load_ccp_path legacy-vs-dense verified on "
          f"{matrix_rows:,} rows")


# ---------------------------------------------------------------------------
# Part C: dense storage round-trip on real artifact families
# ---------------------------------------------------------------------------

def family_files(family, n_files):
    out = Path(mgs.path_out)
    if family == "ccp":
        candidates = sorted((out / "ccp" / "9").glob("*.npz"))
    elif family in ("evt", "vjt"):
        # Period subfolders (evt/9/evt_t9_*.npz); fall back to a recursive
        # search so the test also finds flat or vjt_nog-style layouts.
        candidates = sorted((out / family / "9").glob(f"{family}_t9_*.npz"))
        if not candidates:
            candidates = sorted((out / family).rglob(f"{family}_t*.npz"))
    else:
        raise ValueError(family)
    if not candidates:
        return []
    step = max(1, len(candidates) // n_files)
    return candidates[::step][:n_files]


def run_part_c(tmp, families, n_files):
    print("\n=== PART C: dense storage round-trip on real bundles ===")
    for family in families:
        files = family_files(family, n_files)
        if not files:
            print(f"  {family}: no local files found, skipped")
            continue
        for src in files:
            label = f"{family}:{src.name}"
            started = time.perf_counter()
            with np.load(src) as bundle:
                keys = list(bundle.files)
                arrays = [np.asarray(bundle[k]) for k in keys]
            t_read_old = time.perf_counter() - started

            shapes = {a.shape for a in arrays}
            dtypes = {a.dtype for a in arrays}
            if len(dtypes) != 1:
                fail(f"{label}: mixed dtypes {dtypes}, cannot pack densely")
                continue
            dtype = arrays[0].dtype
            if len(shapes) == 1:
                dense = np.stack(arrays)
                rows = cols = None
            else:
                mats = [np.atleast_2d(a) for a in arrays]
                rows = np.array([m.shape[0] for m in mats], dtype=np.int64)
                cols = np.array([m.shape[1] for m in mats], dtype=np.int64)
                dense = np.zeros(
                    (len(mats), rows.max(), cols.max()), dtype=dtype
                )
                for k, m in enumerate(mats):
                    dense[k, : m.shape[0], : m.shape[1]] = m

            # slice-by-slice verification against every original key
            for k, a in enumerate(arrays):
                sl = (
                    dense[k]
                    if rows is None
                    else dense[k, : rows[k], : cols[k]].reshape(a.shape)
                )
                if not bitwise_equal(a, sl):
                    report_mismatch(f"{label} key {keys[k]} (pack)", a, sl)

            # Time re-writing the SAME data in the current per-state format
            # (to the temp folder), so old-write vs dense-write is measured
            # on this machine rather than assumed.
            legacy_path = os.path.join(tmp, f"legacy_{family}_{src.stem}.npz")
            started = time.perf_counter()
            np.savez_compressed(
                legacy_path, **{k: a for k, a in zip(keys, arrays)}
            )
            t_write_old = time.perf_counter() - started

            dense_path = os.path.join(tmp, f"dense_{family}_{src.stem}.npz")
            payload = {"data": dense, "keys": np.array(keys)}
            if rows is not None:
                payload["rows"] = rows
                payload["cols"] = cols
            started = time.perf_counter()
            np.savez_compressed(dense_path, **payload)
            t_write = time.perf_counter() - started

            started = time.perf_counter()
            with np.load(dense_path) as rb:
                dense2 = rb["data"][:]
                keys2 = [str(k) for k in rb["keys"]]
                rows2 = rb["rows"] if "rows" in rb.files else None
                cols2 = rb["cols"] if "cols" in rb.files else None
            t_read_new = time.perf_counter() - started

            if keys2 != keys:
                fail(f"{label}: key list not preserved in round-trip")
            for k, a in enumerate(arrays):
                sl = (
                    dense2[k]
                    if rows2 is None
                    else dense2[k, : rows2[k], : cols2[k]].reshape(a.shape)
                )
                if not bitwise_equal(a, sl):
                    report_mismatch(f"{label} key {keys[k]} (round-trip)", a, sl)

            print(
                f"  {label}: {len(keys):,} keys | read old "
                f"{t_read_old * 1000:7.0f} ms -> dense {t_read_new * 1000:6.0f} ms "
                f"| write old {t_write_old * 1000:6.0f} ms -> dense "
                f"{t_write * 1000:6.0f} ms | size "
                f"{src.stat().st_size / 1e6:6.1f} MB -> "
                f"{os.path.getsize(dense_path) / 1e6:6.1f} MB"
            )


# ---------------------------------------------------------------------------
# Part D: fused solver emission (server job — needs full solver inputs)
# ---------------------------------------------------------------------------

def run_part_d(tasks, tmp):
    """Fused-vs-standalone-vs-original sequence equivalence.

    For each task: run the FAST SOLVER with EMIT_CCP_SEQUENCE on (ccp_real=1,
    every write redirected into the temp tree), then rebuild the sequence
    from the CCP files that very solve wrote — once with the standalone fast
    builder and once with the ORIGINAL production builder — and require all
    three sequence sets to be byte-identical. Requires param_g, budget
    estimates, and the terminal cache, so this part runs on the server;
    it is skipped with a message when those inputs are missing.
    """
    print("\n=== PART D: fused solver emission "
          "(solver-emitted vs standalone vs original) ===")
    import model_solution_fast as msf
    import contextlib as _ctx

    solver_out = os.path.join(tmp, "fused_solver_writes")
    emitted_root = os.path.join(tmp, "fused_emitted")
    statics = mgf.build_sequence_statics()

    save_before = ms.save_npz_here
    mgs_save_before = mgs.save_npz_here
    mgs_path_before = mgs.path_out
    dense_root_before = mgf.DENSE_ROOT
    emit_before = msf.EMIT_CCP_SEQUENCE
    arrays_compared = 0
    try:
        ms.reload_budgetshock_params()
        x0 = np.load(f"{ms.path_estimates}/param_g.npy")
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED: solver inputs unavailable here ({exc}).")
        print("  Run this part on the server.")
        return
    try:
        for i, em in tasks:
            inv = ms.invariant_states[i, :]
            utility_parameters = ms.build_param_g(em, x0)
            args = (i, ms.invariant_states, ms.debt_range, ms.debt_range,
                    1, utility_parameters, 0, 0, 0, em, True)

            ms.save_npz_here = redirected_writer(solver_out)
            mgf.DENSE_ROOT = emitted_root
            msf.EMIT_CCP_SEQUENCE = True
            started = time.perf_counter()
            with _ctx.redirect_stdout(io.StringIO()):
                msf.get_all_evt_fast(*args)
            t_solve = time.perf_counter() - started
            msf.EMIT_CCP_SEQUENCE = emit_before
            ms.save_npz_here = save_before

            # Standalone fast builder on the CCPs this solve just wrote.
            with _ctx.redirect_stdout(io.StringIO()):
                standalone = mgf.get_ccp_sequence_fast(
                    i, ms.invariant_states, ms.debt_range, em,
                    write_mode="none", ccp_root=solver_out,
                    return_dense=True,
                )

            # Original production builder on the same CCPs.
            mgs.save_npz_here = redirected_writer(
                os.path.join(tmp, "fused_original")
            )
            mgs.path_out = solver_out
            with _ctx.redirect_stdout(io.StringIO()):
                mgs.get_ccp_sequence(
                    i, ms.invariant_states, ms.debt_range, em
                )
            mgs.save_npz_here = mgs_save_before
            mgs.path_out = mgs_path_before

            for p in range(1, T):
                with np.load(mgf.dense_sequence_path(
                    p, inv, em, root=emitted_root
                )) as bundle:
                    emitted = bundle["evt"][:]
                if not bitwise_equal(emitted, standalone[p]):
                    report_mismatch(
                        f"fused task(i={i},em={em}) p{p} "
                        "(emitted vs standalone)",
                        emitted, standalone[p],
                    )
                arrays_compared += 1
                rel = (f"evt_ccp/{p}/"
                       f"evt_ccp_sequence_t{p}_{inv}_em{em}.npz")
                with np.load(os.path.join(
                    tmp, "fused_original", *rel.split("/")
                )) as fo:
                    for k, s in enumerate(statics["x2_strings"][p]):
                        key = f"evt_ccp_sequence_t{p}_{inv}_{s}"
                        if not bitwise_equal(fo[key], emitted[k]):
                            report_mismatch(
                                f"fused task(i={i},em={em}) p{p} row {k} "
                                "(emitted vs original)",
                                fo[key], emitted[k],
                            )
                        arrays_compared += 1
            print(f"  task(i={i}, em={em}): fused solve {t_solve:6.1f} s; "
                  "emitted == standalone == original checked")
    finally:
        ms.save_npz_here = save_before
        mgs.save_npz_here = mgs_save_before
        mgs.path_out = mgs_path_before
        mgf.DENSE_ROOT = dense_root_before
        msf.EMIT_CCP_SEQUENCE = emit_before
    print(f"PART D: {arrays_compared:,} sequence arrays compared "
          f"({len(tasks)} fused solver tasks)")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", default="2",
                        help="number of (state, type) tasks, or 'all'")
    parser.add_argument("--types", default="auto",
                        help="'auto' (locally available), 'all', or e.g. '1,2'")
    parser.add_argument("--consumer-states", type=int, default=25,
                        help="sampled states per period in part B "
                             "(all feasible choices of each are tested)")
    parser.add_argument("--storage-files", type=int, default=2,
                        help="files per family in part C")
    parser.add_argument("--families", default="ccp,evt,vjt")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--skip-consumer", action="store_true")
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--fused", action="store_true",
                        help="run part D: fused solver emission (server; "
                             "runs full solver tasks, ~1 min each)")
    parser.add_argument("--fused-tasks", type=int, default=2)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if args.types == "auto":
        types = available_types()
    elif args.types == "all":
        types = list(TYPE_IDS)
    else:
        types = [int(t) for t in args.types.split(",")]
    if not types:
        print("No complete CCP inputs found for any type; nothing to test.")
        sys.exit(2)
    print(f"types under test: {types}")

    tasks = complete_tasks(types)
    if args.tasks != "all":
        want = int(args.tasks)
        step = max(1, len(tasks) // want)
        tasks = tasks[::step][:want]
    print(f"tasks under test: {tasks}")

    tmp = tempfile.mkdtemp(prefix="ccp_seq_equiv_")
    print(f"temporary output tree: {tmp}  (Model/ is never written)")
    try:
        orig_root, fast_root = run_part_a(tasks, tmp)
        if not args.skip_consumer:
            run_part_b(tasks, orig_root, fast_root,
                       args.consumer_states, args.seed)
        if not args.skip_storage:
            run_part_c(tmp, args.families.split(","), args.storage_files)
        if args.fused:
            run_part_d(tasks[: args.fused_tasks], tmp)
    finally:
        if args.keep_temp:
            print(f"temp kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} mismatches)")
        sys.exit(1)
    print("RESULT: PASS — all comparisons bitwise identical")


if __name__ == "__main__":
    main()
