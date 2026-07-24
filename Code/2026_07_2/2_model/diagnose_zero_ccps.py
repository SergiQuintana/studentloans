# -*- coding: utf-8 -*-
"""Diagnose exact-zero home-production CCPs (READ-ONLY, writes nothing).

Background (2026-07-24): running the CCP-sequence code with warnings
visible showed `divide by zero encountered in log` at
``-np.log(ccp_home)`` — some (state, debt) points in the solver's
ccp_t{p} files carry a home-production probability of EXACTLY 0.0, so the
continuation sequence gets +inf entries there. This script answers:

  1. How many exact zeros are there, and where (period / state type /
     debt region)? Are they underflow (tiny-but-positive neighbors) or
     structural?
  2. How far do the resulting +inf values propagate through the backward
     recursion evt_p = -log(ccp_p) + beta * evt_{p+1}(succ)?
  3. Do OBSERVED estimation states ever touch an inf continuation value —
     i.e., can this affect the budget SMM at all?

Usage (server; --types auto restricts to the em types whose ccp files
exist, "all" asserts all 16):
    python3 .../2_model/diagnose_zero_ccps.py --types all --max-report 20
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import model_solution_em as ms
import model_getccp_sequence as mgs
import model_getccp_sequence_fast as mgf
from latent_types import TYPE_IDS

T = 10
beta = 0.98


def available_types():
    found = []
    inv0 = ms.invariant_states[0, :]
    for em in TYPE_IDS:
        if all(
            os.path.exists(f"{mgs.path_out}/ccp/{p}/ccp_t{p}_[{inv0}]_em{em}.npz")
            for p in range(1, T)
        ):
            found.append(em)
    return found


def describe_state(x2):
    """Human-readable classification of a dynamic state."""
    tags = []
    if x2[7] == 0 and x2[9] == 0 and x2[1] == 0 and x2[2] == 0:
        tags.append("never-active")
    if x2[1] >= 1:
        tags.append(f"2yr={x2[1]}")
    if x2[2] >= 1:
        tags.append(f"4yr={x2[2]}")
    if x2[3] >= 1:
        tags.append(f"grad={x2[3]}")
    if x2[0] >= 1:
        tags.append(f"exp={x2[0]}")
    if x2[4] == 1 or x2[5] == 1 or x2[6] == 1:
        tags.append("degree")
    return ",".join(tags) if tags else "initial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="auto")
    parser.add_argument("--states", default="all",
                        help="'all' or number of invariant states to scan")
    parser.add_argument("--max-report", type=int, default=12)
    args = parser.parse_args()

    if args.types == "auto":
        types = available_types()
    elif args.types == "all":
        types = list(TYPE_IDS)
    else:
        types = [int(t) for t in args.types.split(",")]
    if not types:
        print("no complete ccp file sets found")
        sys.exit(2)

    n_inv = ms.invariant_states.shape[0]
    inv_ids = range(n_inv) if args.states == "all" else range(int(args.states))

    statics = mgf.build_sequence_statics()
    states = statics["states"]
    succ = statics["succ"]
    n_debt = ms.get_debt_range().size

    # ------------------------------------------------------------------
    # 1+2: scan zeros and propagate infs, per (invariant state, type)
    # ------------------------------------------------------------------
    zero_by_period = {p: 0 for p in range(1, T)}
    entries_by_period = {p: 0 for p in range(1, T)}
    tiny_by_period = {p: 0 for p in range(1, T)}   # 0 < ccp < 1e-300
    inf_by_period = {p: 0 for p in range(1, T)}
    zero_state_rows = {}   # (period, state row) -> count of zero debt points
    example_reports = []
    inf_state_rows = {p: set() for p in range(1, T)}

    for em in types:
        for i in inv_ids:
            inv = ms.invariant_states[i, :]
            evt_next = None
            for p in range(T - 1, 0, -1):
                path = (f"{mgs.path_out}/ccp/{p}/"
                        f"ccp_t{p}_[{inv}]_em{em}.npz")
                keys = statics["x2_strings"][p]
                with np.load(path) as bundle:
                    first = bundle[f"ccp_t{p}_[{inv}]_{keys[0]}"]
                    ccp = np.empty((len(keys),) + np.shape(first))
                    ccp[0] = first
                    for k in range(1, len(keys)):
                        ccp[k] = bundle[f"ccp_t{p}_[{inv}]_{keys[k]}"]

                zeros = ccp == 0.0
                tiny = (ccp > 0.0) & (ccp < 1e-300)
                zero_by_period[p] += int(zeros.sum())
                tiny_by_period[p] += int(tiny.sum())
                entries_by_period[p] += ccp.size

                if zeros.any():
                    rows = np.flatnonzero(zeros.any(axis=tuple(
                        range(1, ccp.ndim))))
                    for k in rows:
                        zero_state_rows[(p, k)] = (
                            zero_state_rows.get((p, k), 0)
                            + int(zeros[k].sum())
                        )
                    if len(example_reports) < args.max_report:
                        k = rows[0]
                        dz = np.flatnonzero(zeros[k].reshape(-1))
                        example_reports.append(
                            f"  em={em} inv={inv.astype(int)} p={p} "
                            f"state#{k} [{describe_state(states[p][k])}] "
                            f"zeros at debt idx {dz[:8].tolist()}"
                            f"{'...' if dz.size > 8 else ''} "
                            f"(min positive ccp this state: "
                            f"{ccp[k][ccp[k] > 0].min() if (ccp[k] > 0).any() else float('nan'):.3e})"
                        )

                with np.errstate(divide="ignore"):
                    if p == T - 1:
                        evt = -np.log(ccp) + beta * 0.0
                    else:
                        evt = -np.log(ccp) + beta * evt_next[succ[p]]
                evt_next = evt
                n_inf = int(np.isinf(evt).sum())
                inf_by_period[p] += n_inf
                if n_inf:
                    inf_rows = np.flatnonzero(
                        np.isinf(evt).any(axis=tuple(range(1, evt.ndim)))
                    )
                    inf_state_rows[p].update(inf_rows.tolist())

    print("=" * 70)
    print("1) EXACT ZEROS in home-production CCP files "
          f"(types {types}, {len(list(inv_ids))} invariant states)")
    print(f"{'period':>7} | {'entries':>13} | {'exact 0':>9} | "
          f"{'0<ccp<1e-300':>12} | share zero")
    for p in range(1, T):
        e, z, t = entries_by_period[p], zero_by_period[p], tiny_by_period[p]
        print(f"{p:>7} | {e:>13,} | {z:>9,} | {t:>12,} | "
              f"{z / max(e, 1):.2e}")
    print("\nfirst examples (state classification + debt location):")
    for line in example_reports:
        print(line)

    print("\n2) PROPAGATED +inf ENTRIES in the continuation sequence")
    for p in range(1, T):
        n_states_inf = len(inf_state_rows[p])
        print(f"  period {p}: {inf_by_period[p]:>10,} inf entries "
              f"across {n_states_inf:,} of {states[p].shape[0]:,} states")

    # ------------------------------------------------------------------
    # 3: do observed estimation states touch an inf?
    # ------------------------------------------------------------------
    print("\n3) OBSERVED-STATE CONTACT (exact SMM usage)")
    # The SMM's load_ccp_path takes an observed individual at period p,
    # moves the state with the OBSERVED choice (both graduation branches
    # when graduation is possible), and reads the period p+1 sequence.
    # Contact therefore means: a successor of an observed (state, choice)
    # pair lies in a period p+1 state whose sequence contains an inf.
    try:
        import model_em_algorithm as me
        from model_solution_em import move_state_grad as msg

        checked = 0
        touched_pairs = 0
        touched_by_period = {}
        current_state_zero = 0
        for p in range(1, T - 1):
            try:
                x1_obs, x2_obs, debt_obs, choices_obs = (
                    me.load_data_superfeasible(p, return_income=False)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  period {p}: cannot load observed data ({exc})")
                continue
            x2_obs = np.asarray(x2_obs, dtype=np.int64)
            choices_obs = np.asarray(choices_obs)
            rows_next = statics["row_index"][p + 1]
            inf_next = inf_state_rows[p + 1]
            rows_here = statics["row_index"][p]
            inf_here = inf_state_rows[p]
            for k in range(x2_obs.shape[0]):
                x2i = x2_obs[k]
                ji = choices_obs[k]
                checked += 1
                # secondary stat: the observed state ITSELF has inf entries
                row_here = rows_here.get(
                    np.ascontiguousarray(x2i).tobytes()
                )
                if row_here is not None and row_here in inf_here:
                    current_state_zero += 1
                successors = [np.asarray(msg(x2i, ji, p), dtype=np.int64)]
                if (
                    ((x2i[1] >= 1) & (ji[1] == 1) & (x2i[4] == 0))
                    | ((x2i[2] >= 3) & (ji[1] == 2) & (x2i[5] == 0))
                    | (ji[1] == 3)
                ):
                    successors.append(
                        np.asarray(msg(x2i, ji, p, grad=1), dtype=np.int64)
                    )
                hit = False
                for x2n in successors:
                    row = rows_next.get(
                        np.ascontiguousarray(x2n).tobytes()
                    )
                    if row is not None and row in inf_next:
                        hit = True
                if hit:
                    touched_pairs += 1
                    touched_by_period[p] = touched_by_period.get(p, 0) + 1
        print(f"  observed (state, choice) pairs checked        : {checked:,}")
        print(f"  pairs whose SMM successor has any inf entry   : "
              f"{touched_pairs:,}")
        print(f"  observed states whose OWN sequence has an inf : "
              f"{current_state_zero:,}")
        if touched_by_period:
            for p in sorted(touched_by_period):
                print(f"    period {p}: {touched_by_period[p]:,} pairs")
        if touched_pairs == 0 and current_state_zero == 0:
            print("  -> the zeros live only in states the estimation sample "
                  "never reaches; they cannot affect the SMM through the "
                  "continuation sequences.")
        else:
            print("  -> the zeros DO reach estimation states; the next "
                  "question is whether the inf debt points fall inside the "
                  "feasible debt windows the search visits.")
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIPPED (needs estimation data files): {exc}")

    print("\nDone. Read-only: nothing was written.")


if __name__ == "__main__":
    main()
