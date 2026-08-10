# -*- coding: utf-8 -*-
"""SIGMA GRID: Aguirregabiria-Mira estimation + baseline simulation per sigma.

Sergi's request (2026-08-08): run the full estimation_all_em pipeline for a
grid of FIXED risk-aversion values, at least 20 NPL iterations each, keep the
small permanent outputs per sigma, and let the heavy solution matrices be
overwritten by the next sigma.

For each sigma in SIGMAS, sequentially:

  1. Restore the budget-shock warm start captured once at driver start, so
     every sigma begins from the SAME vector (independent runs, no chaining
     across sigmas).
  2. Run estimation_all_em.py as a subprocess with
         SIGMA_GRID_FIXED_RISK_AVERSION = <sigma>
         SIGMA_GRID_NPL_ITERATIONS     = <NPL_ITERATIONS>
     (environment hooks added 2026-08-08; without these variables that
     script behaves exactly as before).
  3. Run run_sigma_grid_simulate.py: re-solve the estimated model in
     simulation mode and simulate SIMULATION_SAMPLES cohorts — one replica
     of every initial individual per cohort — storing the FULL simulated
     dataset (states, choices, welfare, graduation probabilities, types)
     AND the taste epsilons (e_cohort*_period*.npy), which the
     counterfactual pipeline reloads for common-random-number analysis.
  4. Archive the permanent outputs to Model/Estimates/sigma_grid/sigma_<s>/:
       estimates/   estimates_it*, se_it*, likelihood_it*, param_g.npy,
                    budgetshock_* bundle files, this sigma's run logs
       simulation/  the six simulation trees, MOVED (so they are empty for
                    the next sigma)
  5. Write DONE.txt. The heavy matrices (vjt/evt trees, dense CCP
     sequences) are NOT archived: the next sigma overwrites them in place.
     The shared auxiliary trees (ccp_initial, evt_ccp_dense_initial) are
     never touched — they do not depend on sigma.

A sigma that fails (e.g. -inf likelihood at an extreme sigma) is logged,
marked with FAILED.txt, and the driver continues with the next sigma.
Re-running the driver SKIPS sigmas whose folder already contains DONE.txt,
so the grid can be resumed after an interruption.

Run on the server (from the repo root, inside tmux):

    python3 studentloans/Code/2026_07_2/2_model/run_sigma_grid.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from config import OUT, path_estimates  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SIGMAS = [0.2, 0.4, 0.8, 1.1, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
NPL_ITERATIONS = 20            # Aguirregabiria-Mira iterations per sigma
SIMULATION_SAMPLES = 100       # simulated replicas per initial individual
SIMULATION_WORKERS = 50        # concurrent cohort simulations (memory guard)

GRID_ROOT = Path(path_estimates) / "sigma_grid"
SNAPSHOT_DIR = GRID_ROOT / "_warm_start_snapshot"
PRE_GRID_BACKUP = GRID_ROOT / "_pre_grid_simulation_backup"
LOGS_DIR = Path(path_estimates) / "logs"

# Small files restored before every sigma so each run starts identically.
WARM_START_FILES = ("budgetshock_bestx.npy", "budgetshock_params.npy")

# Small estimate files copied into the archive after a successful run.
ESTIMATE_PATTERNS = (
    "estimates_it*_sigma_est.npy",
    "se_it*_sigma_est.npy",
    "likelihood_it*_sigma_est.npy",
    "param_g.npy",
    "budgetshock*.npy",
)

# Simulation output trees written by model_simulation_em.simulate_choices.
SIMULATION_TREES = ("choice", "state", "epsilon", "welfare", "grad_prob", "types")


def _log(message):
    print(f"[sigma-grid] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
          flush=True)


def _snapshot_warm_start():
    """Capture the pre-grid budget-shock vector once; reuse on later resumes."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in WARM_START_FILES:
        source = Path(path_estimates) / name
        target = SNAPSHOT_DIR / name
        if target.exists():
            continue  # resume: keep the original snapshot
        if source.exists():
            shutil.copy2(source, target)
            _log(f"warm-start snapshot: {name}")
        else:
            _log(f"WARNING: {name} not found; sigma runs will start from "
                 "whatever the estimation's own restart logic finds")


def _restore_warm_start():
    for name in WARM_START_FILES:
        source = SNAPSHOT_DIR / name
        if source.exists():
            shutil.copy2(source, Path(path_estimates) / name)


def _backup_pre_grid_simulation():
    """Move any pre-existing simulation output out of the way, once."""
    for tree in SIMULATION_TREES:
        tree_dir = Path(OUT(tree))
        tree_dir.mkdir(parents=True, exist_ok=True)
        entries = list(tree_dir.iterdir())
        if not entries:
            continue
        backup = PRE_GRID_BACKUP / tree
        backup.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            shutil.move(str(entry), str(backup / entry.name))
        _log(f"pre-grid simulation files moved: {tree} ({len(entries)} entries)")


def _clear_simulation_trees():
    for tree in SIMULATION_TREES:
        tree_dir = Path(OUT(tree))
        tree_dir.mkdir(parents=True, exist_ok=True)
        for entry in tree_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def _archive_estimates(sigma_dir):
    target = sigma_dir / "estimates"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ESTIMATE_PATTERNS:
        for source in Path(path_estimates).glob(pattern):
            shutil.copy2(source, target / source.name)
            copied += 1
    _log(f"archived {copied} estimate files -> {target}")


def _archive_logs(sigma_dir, started_at):
    """Copy every estimation log created after this sigma started."""
    target = sigma_dir / "estimates"
    target.mkdir(parents=True, exist_ok=True)
    if not LOGS_DIR.exists():
        return
    for source in LOGS_DIR.glob("estimation_all_em_*.log"):
        if source.stat().st_mtime >= started_at:
            shutil.copy2(source, target / source.name)


def _archive_simulation(sigma_dir):
    """MOVE the simulation trees into the archive (empties them for the
    next sigma)."""
    target_root = sigma_dir / "simulation"
    moved = 0
    for tree in SIMULATION_TREES:
        tree_dir = Path(OUT(tree))
        target = target_root / tree
        target.mkdir(parents=True, exist_ok=True)
        if not tree_dir.exists():
            continue
        for entry in tree_dir.iterdir():
            shutil.move(str(entry), str(target / entry.name))
            moved += 1
    _log(f"archived {moved} simulation files -> {target_root}")


def _run_subprocess(label, command, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    _log(f"starting {label}: {' '.join(command)}"
         + (f" | env {extra_env}" if extra_env else ""))
    started = time.perf_counter()
    return_code = subprocess.call(command, cwd=str(THIS_DIR), env=env)
    elapsed = (time.perf_counter() - started) / 3600.0
    _log(f"finished {label}: return code {return_code} "
         f"after {elapsed:.2f} hours")
    return return_code


def main():
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    _log(f"SIGMA GRID: {SIGMAS}")
    _log(f"NPL iterations per sigma = {NPL_ITERATIONS}; "
         f"simulation samples = {SIMULATION_SAMPLES} "
         f"(workers {SIMULATION_WORKERS})")
    _log(f"archive root = {GRID_ROOT}")

    _snapshot_warm_start()
    _backup_pre_grid_simulation()

    summary = []
    for sigma in SIGMAS:
        tag = f"sigma_{sigma:.2f}"
        sigma_dir = GRID_ROOT / tag
        if (sigma_dir / "DONE.txt").exists():
            _log(f"{tag}: already DONE, skipping")
            summary.append((sigma, "done (previous run)"))
            continue
        sigma_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        _log(f"===== {tag} =====")

        _restore_warm_start()
        _clear_simulation_trees()

        return_code = _run_subprocess(
            f"{tag} estimation ({NPL_ITERATIONS} NPL iterations)",
            [sys.executable, str(THIS_DIR / "estimation_all_em.py")],
            extra_env={
                "SIGMA_GRID_FIXED_RISK_AVERSION": str(sigma),
                "SIGMA_GRID_NPL_ITERATIONS": str(NPL_ITERATIONS),
            },
        )
        if return_code != 0:
            (sigma_dir / "FAILED.txt").write_text(
                f"estimation failed with return code {return_code} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            _archive_logs(sigma_dir, started_at)
            _log(f"{tag}: ESTIMATION FAILED (rc {return_code}) — "
                 "moving to the next sigma")
            summary.append((sigma, f"FAILED estimation (rc {return_code})"))
            continue

        return_code = _run_subprocess(
            f"{tag} simulation ({SIMULATION_SAMPLES} cohorts)",
            [sys.executable, str(THIS_DIR / "run_sigma_grid_simulate.py"),
             "--samples", str(SIMULATION_SAMPLES),
             "--workers", str(SIMULATION_WORKERS)],
        )
        simulation_ok = return_code == 0
        if not simulation_ok:
            (sigma_dir / "SIMULATION_FAILED.txt").write_text(
                f"simulation failed with return code {return_code} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            _log(f"{tag}: SIMULATION FAILED (rc {return_code}) — archiving "
                 "the estimates anyway")

        _archive_estimates(sigma_dir)
        _archive_logs(sigma_dir, started_at)
        _archive_simulation(sigma_dir)

        (sigma_dir / "MANIFEST.txt").write_text(
            f"sigma (fixed risk aversion) = {sigma}\n"
            f"NPL iterations              = {NPL_ITERATIONS}\n"
            f"simulation samples          = {SIMULATION_SAMPLES}\n"
            f"simulation ok               = {simulation_ok}\n"
            f"started                     = "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}\n"
            f"finished                    = {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        if simulation_ok:
            (sigma_dir / "DONE.txt").write_text("ok\n")
            summary.append((sigma, "done"))
        else:
            summary.append((sigma, "estimation ok, SIMULATION FAILED"))
        _log(f"{tag}: complete after "
             f"{(time.time() - started_at) / 3600.0:.2f} hours")

    _log("===== GRID SUMMARY =====")
    for sigma, status in summary:
        _log(f"  sigma {sigma:.2f}: {status}")


if __name__ == "__main__":
    main()
