# -*- coding: utf-8 -*-
"""Full-grid audit of the exact-zero home-production CCPs (READ-ONLY).

Follow-up to diagnose_zero_ccps.py (whose server run on 2026-07-24 showed
~5.5% exact zeros in periods 1-7, degenerate 0/1 CCPs, infs propagating
into the ccp sequence, and contact with observed states at SOME debt
point). Sergi's requirement: check ALL state points, not only observed
individuals, and classify every possible reason the zeros exist —
p(home) = 0 must never be reachable by the estimation.

Four sections, each with a printed verdict:

1. CENSUS + CLASSIFICATION of every zero over the whole grid.
   For each (type, invariant state, period, dynamic state, debt point)
   with ccp_home == 0, classify the (state, debt) combination:
     Z1  no education history at all, debt > 0            -> impossible
     Z2  debt above a deliberately GENEROUS upper bound of what the
         borrowing caps allow given the state's education history
         (caps compounded at (1+r) for the maximal number of periods,
         so anything above it is certainly unreachable)    -> impossible
     Z3  debt within what the caps allow                  -> SUSPICIOUS
   Z3 is the bucket that must be explained; all Z3 patterns are printed
   (aggregated). Degenerate ccp_home == 1 entries are counted as well.

2. MECHANISM PROBE. For samples of Z1 and Z3 points, open the matching
   vjt bundle and report the choice-value pattern at that (state, debt)
   row: is home -inf while other choices are finite (the expected
   mechanism: with debt service and no labor income, home production
   cannot reach the consumption floor), or something else?

3. SMM CONTACT OVER THE FULL GRID. For EVERY period, dynamic state,
   feasible education choice, and CAP-REACHABLE current debt point,
   build the exact production-SMM debt window (nearest-grid convention
   of debt_limits.precompute_smm_bounds_indices) and check whether the
   successor state's ccp-sequence row (graduation and no-graduation
   branches) contains an inf INSIDE the window. Zero contacts = the
   budget shock estimation can never evaluate an inf continuation, for
   any individual whatsoever.

4. BELLMAN CONTACT OVER THE FULL GRID. Scan the solved evt files for
   non-finite entries (inf or nan) and run the same window logic with
   the Bellman conventions (get_debt_region_bounds; education windows
   plus the non-school rollover point) over every feasible choice and
   cap-reachable current debt. Zero contacts = the model solution's
   value at every reachable point is insulated from the degenerate
   corners of the grid.

Run on the server (writes nothing anywhere):
    python3 .../2_model/diagnose_zero_ccps_grid.py --types all --workers 30
Runtime is dominated by reading the ccp and evt files (~25 s per
(type, invariant state) task); with 30 workers the full 16-type scan is
roughly 15-20 minutes.
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
from debt_limits import (
    INTEREST_RATE,
    get_annual_cap_by_stage,
    get_lifetime_cap_by_stage,
    lower_bound_index,
    upper_bound_index,
    nearest_grid_index,
)

T = 10
beta = 0.98
r = INTEREST_RATE

# ---------------------------------------------------------------------------
# Shared statics (built once in the parent; workers inherit via fork)
# ---------------------------------------------------------------------------

STAT = {}


def build_grid_statics():
    """States, feasible choices, successor rows, cap bounds, debt windows."""
    seq_statics = mgf.build_sequence_statics()
    states = seq_statics["states"]
    row_index = seq_statics["row_index"]
    grid = ms.get_debt_range().astype(np.float64)
    n_debt = grid.size

    # Feasible choices and successor rows for every (period, state, choice).
    choices, succ_ng, succ_g, grad_possible = {}, {}, {}, {}
    for p in range(1, T - 1 + 1):
        ch_p, ng_p, g_p, gp_p = [], [], [], []
        nxt = row_index.get(p + 1)
        for s in range(states[p].shape[0]):
            x2 = states[p][s]
            Jx = np.asarray(ms.get_possible_choices(x2), dtype=np.int64)
            ng = np.full(Jx.shape[0], -1, dtype=np.int64)
            gg = np.full(Jx.shape[0], -1, dtype=np.int64)
            gp = np.zeros(Jx.shape[0], dtype=bool)
            if nxt is not None:
                for c in range(Jx.shape[0]):
                    j = Jx[c]
                    z = np.ascontiguousarray(np.asarray(
                        ms.move_state_grad(x2, j, p), dtype=np.int64))
                    ng[c] = nxt[z.tobytes()]
                    gp[c] = bool(
                        ((x2[1] >= 1) & (j[1] == 1) & (x2[4] == 0))
                        | ((x2[2] >= 3) & (j[1] == 2) & (x2[5] == 0))
                        | (j[1] == 3)
                    )
                    if gp[c]:
                        zg = np.ascontiguousarray(np.asarray(
                            ms.move_state_grad(x2, j, p, grad=1),
                            dtype=np.int64))
                        gg[c] = nxt[zg.tobytes()]
            ch_p.append(Jx)
            ng_p.append(ng)
            g_p.append(gg)
            gp_p.append(gp)
        choices[p], succ_ng[p], succ_g[p], grad_possible[p] = (
            ch_p, ng_p, g_p, gp_p
        )

    # Generous upper bound of reachable debt for every (period, state):
    # borrow every enrolled year at the LARGEST applicable annual cap,
    # compound everything at (1+r) for the maximal p-1 periods, and take
    # the lifetime cap (also compounded) as an alternative ceiling.
    # Anything above this is certainly unreachable; anything below it is
    # treated as potentially reachable (the conservative direction).
    max_debt = {}
    for p in range(1, T):
        sp = states[p]
        ub = np.zeros(sp.shape[0])
        for s in range(sp.shape[0]):
            twoy, foury, grad = int(sp[s][1]), int(sp[s][2]), int(sp[s][3])
            total = 0.0
            for y in range(twoy):
                total += get_annual_cap_by_stage(1, y, 0)
            for y in range(foury):
                total += get_annual_cap_by_stage(2, 0, y)
            for _ in range(grad):
                total += get_annual_cap_by_stage(3, 0, 0)
            if total <= 0.0:
                ub[s] = 0.0
                continue
            ltc = get_lifetime_cap_by_stage(3 if grad > 0 else 2)
            compounded = (1.0 + r) ** max(p - 1, 0)
            ub[s] = min(total, ltc) * compounded
        max_debt[p] = ub

    # legal_debt[p][s]: boolean over grid points (index 0 always legal).
    legal_debt = {
        p: (grid[None, :] <= max_debt[p][:, None] + 1e-9) | (grid[None, :] == 0.0)
        for p in range(1, T)
    }

    # Debt windows depend only on (annual cap, lifetime cap) — a handful of
    # distinct pairs — precompute lo/hi over the grid for each pair, in BOTH
    # conventions.
    smm_windows, bell_windows = {}, {}
    for cap in (8391.0, 9309.0, 12581.0, 23222.0):
        for ltc in (70786.0, 150000.0):
            lo_s = np.empty(n_debt, dtype=np.int64)
            hi_s = np.empty(n_debt, dtype=np.int64)
            lo_b = np.empty(n_debt, dtype=np.int64)
            hi_b = np.empty(n_debt, dtype=np.int64)
            for k in range(n_debt):
                accrued = (1.0 + r) * grid[k]
                maximum = min(accrued + cap, ltc)
                lo_s[k] = nearest_grid_index(grid, grid[k])
                hi_s[k] = nearest_grid_index(grid, maximum)
                if accrued >= ltc:
                    lo_b[k] = hi_b[k] = lower_bound_index(grid, accrued)
                else:
                    lo_b[k] = lower_bound_index(grid, accrued)
                    hi_b[k] = max(upper_bound_index(grid, maximum), lo_b[k])
            smm_windows[(cap, ltc)] = (lo_s, hi_s)
            bell_windows[(cap, ltc)] = (lo_b, hi_b)
    rollover = np.array(
        [lower_bound_index(grid, (1.0 + r) * grid[k]) for k in range(n_debt)],
        dtype=np.int64,
    )

    STAT.update(
        states=states, grid=grid, n_debt=n_debt, choices=choices,
        succ_ng=succ_ng, succ_g=succ_g, grad_possible=grad_possible,
        max_debt=max_debt, legal_debt=legal_debt,
        smm_windows=smm_windows, bell_windows=bell_windows,
        rollover=rollover, x2_strings=seq_statics["x2_strings"],
    )


def window_key(j, x2):
    cap = get_annual_cap_by_stage(int(j[1]), int(x2[1]), int(x2[2]))
    ltc = get_lifetime_cap_by_stage(int(j[1]))
    return (cap, ltc)


def any_in_window(prefix_row, lo, hi):
    """Vectorized: any flagged entry in [lo, hi] per element (prefix sums)."""
    return (prefix_row[hi + 1] - prefix_row[lo]) > 0


# ---------------------------------------------------------------------------
# Per-(type, invariant state) worker
# ---------------------------------------------------------------------------

def scan_task(task):
    em, i = task
    inv = ms.invariant_states[i, :]
    states = STAT["states"]
    n_debt = STAT["n_debt"]
    out = {
        "zeros": np.zeros((T, 3), dtype=np.int64),      # per period x class
        "ones": np.zeros(T, dtype=np.int64),
        "zero_at_nodebt": np.zeros(T, dtype=np.int64),
        "z3_patterns": {},        # (p, hist, debt idx) aggregated
        "z1_samples": [], "z3_samples": [],
        "smm_contacts": np.zeros(T, dtype=np.int64),
        "smm_checked": np.zeros(T, dtype=np.int64),
        "smm_samples": [],
        "bell_contacts": np.zeros(T, dtype=np.int64),
        "bell_checked": np.zeros(T, dtype=np.int64),
        "bell_nonfinite": np.zeros(T, dtype=np.int64),
        "bell_samples": [],
        "errors": [],
    }

    # ---- read ccp files; census + sequence recursion ----------------------
    zero_mask = {}
    seq_inf_prefix = {}
    seq_next = None
    for p in range(T - 1, 0, -1):
        keys = STAT["x2_strings"][p]
        try:
            with np.load(
                f"{mgs.path_out}/ccp/{p}/ccp_t{p}_[{inv}]_em{em}.npz"
            ) as bundle:
                first = bundle[f"ccp_t{p}_[{inv}]_{keys[0]}"]
                ccp = np.empty((len(keys),) + np.shape(first))
                ccp[0] = first
                for k in range(1, len(keys)):
                    ccp[k] = bundle[f"ccp_t{p}_[{inv}]_{keys[k]}"]
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"ccp p{p}: {exc}")
            return out
        ccp = ccp.reshape(len(keys), n_debt)

        zmask = ccp == 0.0
        zero_mask[p] = zmask
        out["ones"][p] = int((ccp == 1.0).sum())
        out["zero_at_nodebt"][p] = int(zmask[:, 0].sum())

        legal = STAT["legal_debt"][p]
        hist_all_zero = (
            (states[p][:, 1] == 0) & (states[p][:, 2] == 0)
            & (states[p][:, 3] == 0)
        )
        z1 = zmask & hist_all_zero[:, None] & (STAT["grid"][None, :] > 0)
        z2 = zmask & ~z1 & ~legal
        z3 = zmask & ~z1 & ~z2
        out["zeros"][p, 0] = int(z1.sum())
        out["zeros"][p, 1] = int(z2.sum())
        out["zeros"][p, 2] = int(z3.sum())
        for cls, mask, store in (
            ("z1", z1, out["z1_samples"]), ("z3", z3, out["z3_samples"]),
        ):
            rows = np.flatnonzero(mask.any(axis=1))
            for s in rows[:2]:
                d = int(np.flatnonzero(mask[s])[0])
                if len(store) < 3:
                    store.append((em, i, p, int(s), d))
        z3_rows = np.flatnonzero(z3.any(axis=1))
        for s in z3_rows:
            x2 = states[p][s]
            hist = (int(x2[1]), int(x2[2]), int(x2[3]),
                    int(x2[4]), int(x2[5]), int(x2[6]))
            dz = np.flatnonzero(z3[s])
            key = (p, hist, int(dz.min()), int(dz.max()))
            out["z3_patterns"][key] = (
                out["z3_patterns"].get(key, 0) + int(dz.size)
            )

        with np.errstate(divide="ignore"):
            if p == T - 1:
                seq = -np.log(ccp) + beta * 0.0
            else:
                # home successor: reuse the sequence statics of mgf
                home_succ = mgf.build_sequence_statics()["succ"][p]
                seq = -np.log(ccp) + beta * seq_next[home_succ]
        seq_next = seq
        infmask = ~np.isfinite(seq)
        seq_inf_prefix[p] = np.concatenate(
            [np.zeros((infmask.shape[0], 1), dtype=np.int64),
             np.cumsum(infmask, axis=1)], axis=1,
        )

    # ---- section 3: SMM windows over the full grid ------------------------
    for p in range(1, T - 1):        # SMM continuation at period 9 is zeros
        pref = seq_inf_prefix[p + 1]
        legal = STAT["legal_debt"][p]
        for s in range(states[p].shape[0]):
            b_ok = np.flatnonzero(legal[s])
            if b_ok.size == 0:
                continue
            x2 = states[p][s]
            Jx = STAT["choices"][p][s]
            for c in range(Jx.shape[0]):
                j = Jx[c]
                if j[1] == 0:
                    continue          # the SMM models education choices
                lo, hi = STAT["smm_windows"][window_key(j, x2)]
                hit = any_in_window(
                    pref[STAT["succ_ng"][p][s][c]], lo[b_ok], hi[b_ok]
                )
                if STAT["grad_possible"][p][s][c]:
                    hit |= any_in_window(
                        pref[STAT["succ_g"][p][s][c]], lo[b_ok], hi[b_ok]
                    )
                out["smm_checked"][p] += int(b_ok.size)
                n_hit = int(hit.sum())
                if n_hit:
                    out["smm_contacts"][p] += n_hit
                    if len(out["smm_samples"]) < 3:
                        b0 = int(b_ok[np.flatnonzero(hit)[0]])
                        out["smm_samples"].append(
                            (em, i, p, int(s), [int(v) for v in j],
                             b0, int(lo[b0]), int(hi[b0]))
                        )

    # ---- section 4: Bellman windows against non-finite evt ----------------
    evt_nonfinite_prefix = {}
    for p in range(2, T):            # continuations looked up at p >= 2
        keys = STAT["x2_strings"][p]
        try:
            with np.load(
                f"{mgs.path_out}/evt_nog/{p}/evt_t{p}_[{inv}]_em{em}.npz"
            ) as bundle:
                nf = np.empty((len(keys), n_debt), dtype=bool)
                for k in range(len(keys)):
                    nf[k] = ~np.isfinite(
                        np.asarray(
                            bundle[f"evt_t{p}_[{inv}]_{keys[k]}"]
                        ).reshape(n_debt)
                    )
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"evt p{p}: {exc}")
            evt_nonfinite_prefix[p] = None
            continue
        out["bell_nonfinite"][p] = int(nf.sum())
        evt_nonfinite_prefix[p] = np.concatenate(
            [np.zeros((nf.shape[0], 1), dtype=np.int64),
             np.cumsum(nf, axis=1)], axis=1,
        )

    for p in range(1, T - 1):
        pref = evt_nonfinite_prefix.get(p + 1)
        if pref is None:
            continue
        legal = STAT["legal_debt"][p]
        roll = STAT["rollover"]
        for s in range(states[p].shape[0]):
            b_ok = np.flatnonzero(legal[s])
            if b_ok.size == 0:
                continue
            x2 = states[p][s]
            Jx = STAT["choices"][p][s]
            for c in range(Jx.shape[0]):
                j = Jx[c]
                rows = [STAT["succ_ng"][p][s][c]]
                if STAT["grad_possible"][p][s][c]:
                    rows.append(STAT["succ_g"][p][s][c])
                if j[1] == 0:
                    idx = roll[b_ok]
                    hit = np.zeros(b_ok.size, dtype=bool)
                    for row in rows:
                        hit |= (pref[row][idx + 1] - pref[row][idx]) > 0
                else:
                    lo, hi = STAT["bell_windows"][window_key(j, x2)]
                    hit = np.zeros(b_ok.size, dtype=bool)
                    for row in rows:
                        hit |= any_in_window(pref[row], lo[b_ok], hi[b_ok])
                out["bell_checked"][p] += int(b_ok.size)
                n_hit = int(hit.sum())
                if n_hit:
                    out["bell_contacts"][p] += n_hit
                    if len(out["bell_samples"]) < 3:
                        b0 = int(b_ok[np.flatnonzero(hit)[0]])
                        out["bell_samples"].append(
                            (em, i, p, int(s), [int(v) for v in j], b0)
                        )
    return out


# ---------------------------------------------------------------------------
# Section 2: mechanism probe (parent, serial, few files)
# ---------------------------------------------------------------------------

def probe_mechanism(samples, label, max_probe):
    print(f"\n  probe of {label} points (vjt choice-value pattern):")
    shown = 0
    for em, i, p, s, d in samples:
        if shown >= max_probe:
            break
        inv = ms.invariant_states[i, :]
        x2 = STAT["states"][p][s]
        key = f"vjt_t{p}_[{inv}]_{STAT['x2_strings'][p][s]}"
        path = f"{mgs.path_out}/vjt_nog/{p}/vjt_t{p}_[{inv}]_em{em}.npz"
        try:
            with np.load(path) as bundle:
                vjt_row = np.asarray(bundle[key])[d, :]
        except Exception as exc:  # noqa: BLE001
            print(f"    (cannot read {path}: {exc})")
            continue
        Jx = np.asarray(ms.get_possible_choices(x2), dtype=np.int64)
        home_col = int(np.flatnonzero((Jx == 0).all(axis=1))[0])
        n_inf = int(np.sum(np.isneginf(vjt_row)))
        print(
            f"    em={em} inv={inv.astype(int)} p={p} state#{s} "
            f"hist(2y,4y,gr)={tuple(int(v) for v in x2[1:4])} "
            f"debt_idx={d} (${STAT['grid'][d]:,.0f}): "
            f"v(home)={vjt_row[home_col]:.3g}, "
            f"{n_inf}/{vjt_row.size} choices at -inf, "
            f"max other v={np.max(np.delete(vjt_row, home_col)):.3g}"
        )
        shown += 1


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="auto")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--max-report", type=int, default=25)
    parser.add_argument("--probe-samples", type=int, default=8)
    args = parser.parse_args()

    if args.types == "auto":
        types = []
        inv0 = ms.invariant_states[0, :]
        for em in TYPE_IDS:
            if all(os.path.exists(
                f"{mgs.path_out}/ccp/{p}/ccp_t{p}_[{inv0}]_em{em}.npz"
            ) for p in range(1, T)):
                types.append(em)
    elif args.types == "all":
        types = list(TYPE_IDS)
    else:
        types = [int(t) for t in args.types.split(",")]
    if not types:
        print("no ccp files found")
        sys.exit(2)

    print("building grid statics (states, successors, caps, windows)...")
    build_grid_statics()
    tasks = [(em, i) for em in types
             for i in range(ms.invariant_states.shape[0])]
    print(f"scanning {len(tasks)} (type, invariant state) tasks "
          f"with {args.workers} workers...")

    if args.workers > 1 and os.name != "nt":
        import multiprocessing
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(scan_task, tasks, chunksize=1)
    else:
        results = [scan_task(t) for t in tasks]

    # ---- aggregate --------------------------------------------------------
    zeros = sum(res["zeros"] for res in results)
    ones = sum(res["ones"] for res in results)
    zero_nodebt = sum(res["zero_at_nodebt"] for res in results)
    smm_contacts = sum(res["smm_contacts"] for res in results)
    smm_checked = sum(res["smm_checked"] for res in results)
    bell_contacts = sum(res["bell_contacts"] for res in results)
    bell_checked = sum(res["bell_checked"] for res in results)
    bell_nonfinite = sum(res["bell_nonfinite"] for res in results)
    z3_patterns = {}
    z1_samples, z3_samples, smm_samples, bell_samples = [], [], [], []
    errors = []
    for res in results:
        for key, n in res["z3_patterns"].items():
            z3_patterns[key] = z3_patterns.get(key, 0) + n
        z1_samples += res["z1_samples"]
        z3_samples += res["z3_samples"]
        smm_samples += res["smm_samples"]
        bell_samples += res["bell_samples"]
        errors += res["errors"]

    print("\n" + "=" * 72)
    print("1) ZERO CENSUS, CLASSIFIED "
          "(Z1 never-enrolled+debt | Z2 above caps | Z3 SUSPICIOUS)")
    print(f"{'period':>7} | {'Z1 impossible':>14} | {'Z2 impossible':>14} | "
          f"{'Z3 suspicious':>14} | {'ccp==1':>12} | zero@no-debt")
    for p in range(1, T):
        print(f"{p:>7} | {zeros[p, 0]:>14,} | {zeros[p, 1]:>14,} | "
              f"{zeros[p, 2]:>14,} | {ones[p]:>12,} | {zero_nodebt[p]:>10,}")
    total = zeros.sum(axis=0)
    print(f"{'TOTAL':>7} | {total[0]:>14,} | {total[1]:>14,} | "
          f"{total[2]:>14,}")
    if z3_patterns:
        print(f"\n  Z3 patterns (period, history(2y,4y,gr,deg2,deg4,deggr), "
              f"debt-idx range) — top {args.max_report} by count:")
        ranked = sorted(z3_patterns.items(), key=lambda kv: -kv[1])
        for (p, hist, dlo, dhi), n in ranked[: args.max_report]:
            print(f"    p={p} hist={hist} debt_idx {dlo}-{dhi}: {n:,} zeros")
    else:
        print("\n  NO Z3 zeros: every zero sits at a certainly "
              "unreachable (state, debt) point.")

    print("\n2) MECHANISM (value pattern at sampled zero points)")
    probe_mechanism(z1_samples, "Z1 (never-enrolled + debt)",
                    args.probe_samples)
    if z3_samples:
        probe_mechanism(z3_samples, "Z3 (SUSPICIOUS)", args.probe_samples)

    print("\n3) SMM CONTACT, FULL GRID "
          "(inf inside a production debt window, any state x cap-reachable "
          "debt x education choice)")
    for p in range(1, T - 1):
        print(f"  period {p}: {smm_contacts[p]:>12,} contacts "
              f"of {smm_checked[p]:>14,} checked windows")
    if smm_contacts.sum() == 0:
        print("  -> VERDICT: the budget shock estimation can NEVER evaluate "
              "an inf continuation, for any individual at any reachable "
              "debt level.")
    else:
        print("  -> VERDICT: infs CAN enter production debt windows. "
              "First examples (em, inv, period, state, choice, debt idx, "
              "window):")
        for row in smm_samples[:6]:
            print(f"     {row}")

    print("\n4) BELLMAN CONTACT, FULL GRID "
          "(non-finite evt inside a Bellman debt window)")
    print(f"  non-finite evt entries by period: "
          f"{ {p: int(bell_nonfinite[p]) for p in range(2, T) if bell_nonfinite[p]} }")
    for p in range(1, T - 1):
        print(f"  period {p}: {bell_contacts[p]:>12,} contacts "
              f"of {bell_checked[p]:>14,} checked windows")
    if bell_contacts.sum() == 0:
        print("  -> VERDICT: the model solution never looks up a "
              "non-finite continuation from any reachable "
              "(state, debt, choice).")
    else:
        print("  -> VERDICT: non-finite continuations ARE reachable in the "
              "Bellman. First examples (em, inv, period, state, choice, "
              "debt idx):")
        for row in bell_samples[:6]:
            print(f"     {row}")

    if errors:
        print(f"\nfile errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
    print("\nDone. Read-only: nothing was written.")


if __name__ == "__main__":
    main()
