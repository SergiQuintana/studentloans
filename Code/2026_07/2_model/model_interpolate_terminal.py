# -*- coding: utf-8 -*-
"""
Build and validate interpolation objects for continuation values.

For each state (sex, race, lastschool, educ, major), this script:
1) Loads continuation values (evt) over a grid of (sigma_u, debt).
2) Builds a 2D linear interpolator evt_hat(sigma_u, debt).
3) Optionally validates interpolation quality on a subset of states by:
   - Holding out one sigma value and predicting evt across all debts.
   - Holding out one debt value and predicting evt across all sigmas.
   - Computing R^2 for each test and exporting summary tables.

The state space for (educ, major) is generated from the model coding rules:
- educ in {0,1}  -> major = 0
- educ == 2      -> major = 12
- educ in {3,4}  -> major in {1..fields} excluding 3

NEW (Cache helpers)
-------------------
Added disk cache utilities so you can:
- build_interp_cache(...): build interpolators and STORE them on disk (returns None)
- load_interp_cache(...): load and RETURN the stored interpolators
- cache_exists(...): check if cache exists

Cache location:
    <pathout>/cache/interp_dict.joblib
and optional metadata:
    <pathout>/cache/interp_dict_meta.joblib

@author: S.Quintana
"""

import os
import re
import numpy as np
import pandas as pd
import joblib


# ---------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------
# File discovery and state-space construction
# ---------------------------------------------------------------------
def find_sigma_files(pathcont, sex_filters=None, race_filters=None):
    """
    Locate all NPZ files matching:
      continuation_s{sex}_eth{race}_sigma{sigma}.npz

    Returns a list of dicts containing sex, race, sigma value, and full file path.
    """
    pat = re.compile(r"^continuation_s(?P<s>-?\d+)_eth(?P<eth>-?\d+)_sigma(?P<sigma>.+)\.npz$")
    out = []
    for fn in os.listdir(pathcont):
        if not fn.endswith(".npz"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        try:
            sigma_val = float(m.group("sigma"))
        except ValueError:
            continue

        sex = int(m.group("s"))
        race = int(m.group("eth"))

        if sex_filters is not None and sex not in sex_filters:
            continue
        if race_filters is not None and race not in race_filters:
            continue

        out.append({
            "sex": sex,
            "race": race,
            "sigma": float(sigma_val),
            "fullpath": os.path.join(pathcont, fn),
            "file": fn
        })

    if not out:
        raise FileNotFoundError(f"No continuation_*.npz files found in:\n{pathcont}")

    return out


def group_files_by_sex_race(file_list):
    """
    Group discovered sigma files by (sex, race), and sort each group by sigma.
    """
    grouped = {}
    for d in file_list:
        grouped.setdefault((d["sex"], d["race"]), []).append(d)
    for k in grouped:
        grouped[k].sort(key=lambda z: z["sigma"])
    return grouped


def infer_lastschool_horizon(npz_path):
    """
    Infer the number of lastschool periods by scanning keys like:
      con_last{lastschool}_educ{educ}_major{major}
    """
    pat = re.compile(r"^con_last(?P<last>-?\d+)_educ(?P<educ>-?\d+)_major(?P<major>-?\d+)$")
    max_last = None
    with np.load(npz_path) as z:
        for k in z.files:
            m = pat.match(k)
            if m:
                last = int(m.group("last"))
                max_last = last if max_last is None else max(max_last, last)

    if max_last is None:
        raise ValueError(
            f"Could not infer lastschool horizon: expected con_last*_educ*_major* keys in {os.path.basename(npz_path)}"
        )

    return max_last + 1


def generate_educ_major_pairs(fields=8):
    """
    Construct (educ, major) combinations according to model rules.
    """
    pairs = []
    for educ in range(0, 5):
        if educ < 2:
            pairs.append((educ, 0))
        elif educ == 2:
            pairs.append((educ, 12))
        else:
            for major in range(1, fields + 1):
                if major == 3:
                    continue
                pairs.append((educ, major))
    return pairs


# ---------------------------------------------------------------------
# Loading grids and building interpolators
# ---------------------------------------------------------------------
def load_vector_from_npz(npz_path, key):
    """
    Load a 1D array from a NPZ file under the given key.
    """
    with np.load(npz_path) as z:
        if key not in z.files:
            raise KeyError(key)
        return np.asarray(z[key], dtype=float).ravel()


def load_evt_grid(files_by_sex_race, debt_grid, sex, race, lastschool, educ, major):
    """
    For a fixed (sex, race, lastschool, educ, major), load evt across all sigma files.

    Returns:
      sigmas : (n_sigma,)
      debts  : (n_debt,)
      EVT    : (n_sigma, n_debt)
    """
    flist = files_by_sex_race.get((sex, race), [])
    if not flist:
        raise FileNotFoundError(f"No sigma files found for (sex={sex}, race={race}).")

    key = f"con_last{lastschool}_educ{educ}_major{major}"

    sigmas = np.array([f["sigma"] for f in flist], dtype=float)
    evt_rows = []
    for f in flist:
        evt_rows.append(load_vector_from_npz(f["fullpath"], key))

    EVT = np.vstack(evt_rows)
    debts = np.asarray(debt_grid, dtype=float).ravel()

    if EVT.shape[1] != debts.size:
        raise ValueError(
            f"Debt grid mismatch for state (sex={sex}, race={race}, last={lastschool}, educ={educ}, major={major}): "
            f"evt length={EVT.shape[1]} vs debt_grid length={debts.size}"
        )

    # Ensure strictly increasing axes for the interpolator
    if np.any(np.diff(sigmas) <= 0):
        order = np.argsort(sigmas)
        sigmas = sigmas[order]
        EVT = EVT[order, :]

    if np.any(np.diff(debts) <= 0):
        order = np.argsort(debts)
        debts = debts[order]
        EVT = EVT[:, order]

    return sigmas, debts, EVT


def build_grid_interpolator(sigmas, debts, EVT):
    """
    Create a 2D linear interpolator evt_hat(sigma, debt) on a regular grid.
    """
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator(
        (np.asarray(sigmas, float), np.asarray(debts, float)),
        np.asarray(EVT, float),
        method="linear",
        bounds_error=True
    )


def predict_curve_over_debt(interp, sigma_value, debt_points):
    """
    Evaluate evt_hat at a fixed sigma over many debts.
    """
    debt_points = np.asarray(debt_points, float)
    sigma_arr = np.full_like(debt_points, float(sigma_value))
    pts = np.column_stack([sigma_arr, debt_points])
    return interp(pts)


def predict_curve_over_sigma(interp, sigma_points, debt_value):
    """
    Evaluate evt_hat at a fixed debt over many sigmas.
    """
    sigma_points = np.asarray(sigma_points, float)
    debt_arr = np.full_like(sigma_points, float(debt_value))
    pts = np.column_stack([sigma_points, debt_arr])
    return interp(pts)


# ---------------------------------------------------------------------
# Public API: build interpolators dictionary
# ---------------------------------------------------------------------
def build_interpolator_dictionary(
    pathcont,
    debt_grid,
    fields=8,
    lastschool_horizon=None,
    sex_filters=None,
    race_filters=None,
    verbose=False
):
    """
    Build interpolation objects for all (sex, race, lastschool, educ, major).

    Returns:
      interp_dict : dict[(sex, race, lastschool, educ, major)] -> interpolator
      meta_dict   : dict[...] -> {"sigmas":..., "debts":...}
      missing     : list of (state, missing_key)
      context     : dict containing grouped file references and lastschool horizon (used for validation)
    """
    file_list = find_sigma_files(pathcont, sex_filters=sex_filters, race_filters=race_filters)
    files_by_sex_race = group_files_by_sex_race(file_list)

    if lastschool_horizon is None:
        any_group = next(iter(files_by_sex_race.keys()))
        first_file = files_by_sex_race[any_group][0]["fullpath"]
        lastschool_horizon = infer_lastschool_horizon(first_file)

    educ_major = generate_educ_major_pairs(fields=fields)

    interp_dict = {}
    meta_dict = {}
    missing = []

    for (sex, race), _flist in sorted(files_by_sex_race.items()):
        for lastschool in range(0, lastschool_horizon):
            for educ, major in educ_major:
                state = (sex, race, lastschool, educ, major)
                if verbose:
                    print(state)
                try:
                    sigmas, debts, EVT = load_evt_grid(
                        files_by_sex_race, debt_grid, sex, race, lastschool, educ, major
                    )
                    interp_dict[state] = build_grid_interpolator(sigmas, debts, EVT)
                    meta_dict[state] = {"sigmas": sigmas, "debts": debts}
                except KeyError as e:
                    missing.append((state, str(e)))

    context = {
        "files_by_sex_race": files_by_sex_race,
        "lastschool_horizon": lastschool_horizon,
        "fields": fields
    }

    return interp_dict, meta_dict, missing, context


# ---------------------------------------------------------------------
# NEW: Disk cache helpers
# ---------------------------------------------------------------------
DEFAULT_CACHE_NAME = "interp_dict.joblib"
DEFAULT_META_NAME  = "interp_dict_meta.joblib"


def _cache_dir(pathout: str) -> str:
    return os.path.join(pathout, "cache")


def _cache_paths(pathout: str,
                 cache_name: str = DEFAULT_CACHE_NAME,
                 meta_name: str = DEFAULT_META_NAME) -> tuple[str, str]:
    cdir = _cache_dir(pathout)
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, cache_name), os.path.join(cdir, meta_name)


def cache_exists(pathout: str, cache_name: str = DEFAULT_CACHE_NAME) -> bool:
    cache_path, _ = _cache_paths(pathout, cache_name=cache_name)
    return os.path.exists(cache_path)


def build_interp_cache(
    pathout,
    pathcont,
    debt_grid,
    fields=8,
    lastschool_horizon=None,
    sex_filters=None,
    race_filters=None,
    cache_name=DEFAULT_CACHE_NAME,
    meta_name=DEFAULT_META_NAME,
    compress=3,
    force_rebuild=False,
    verbose=True,
):
    """
    Build interpolators and STORE them on disk.
    Returns None (build-only).
    """
    cache_path, meta_path = _cache_paths(pathout, cache_name=cache_name, meta_name=meta_name)

    if os.path.exists(cache_path) and (not force_rebuild):
        if verbose:
            print(f"[interp_cache] Cache exists, skipping build: {cache_path}")
        return None

    if verbose:
        print("[interp_cache] Building interpolators (expensive)...")

    interp_dict, meta_dict, missing, context = build_interpolator_dictionary(
        pathcont=pathcont,
        debt_grid=debt_grid,
        fields=fields,
        lastschool_horizon=lastschool_horizon,
        sex_filters=sex_filters,
        race_filters=race_filters,
        verbose=verbose
    )

    if verbose:
        print(f"[interp_cache] Saving interp_dict -> {cache_path}")
    joblib.dump(interp_dict, cache_path, compress=compress)

    # Save metadata (optional but useful)
    try:
        joblib.dump(
            {"meta_dict": meta_dict, "missing": missing, "context": context,
             "pathcont": pathcont, "fields": fields},
            meta_path,
            compress=compress,
        )
        if verbose:
            print(f"[interp_cache] Saving meta -> {meta_path}")
    except Exception as e:
        if verbose:
            print(f"[interp_cache] Warning: could not save meta file: {e}")

    if verbose:
        print(f"[interp_cache] Done. interp_dict size={len(interp_dict)}  missing={len(missing)}")

    return None


def load_interp_cache(pathout, cache_name=DEFAULT_CACHE_NAME):
    """
    Load and RETURN interp_dict from disk cache.
    """
    cache_path = os.path.join(pathout, "cache", cache_name)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Interpolation cache not found: {cache_path}\n"
            "Run build_interp_cache(...) once to create it."
        )
    return joblib.load(cache_path)


# ---------------------------------------------------------------------
# Validation: evaluate interpolation accuracy on a subset of models
# ---------------------------------------------------------------------
def validate_interpolators(
    context,
    debt_grid,
    fields=8,
    evaluation_stride=10,
    heldout_debt_index=50,
    out_dir=r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Output\Tables",
    out_name="interp_validation_r2.csv"
):
    """
    Evaluate interpolation accuracy on a subset of state models.

    For each evaluated state:
      1) Hold out one sigma (chosen as the interior middle index), train an interpolator on remaining sigmas,
         predict evt across all debts at the held-out sigma, compute R^2 across debts.
      2) Hold out one debt (index heldout_debt_index if feasible, otherwise interior middle index), train an interpolator
         on remaining debts, predict evt across all sigmas at the held-out debt, compute R^2 across sigmas.

    Exports:
      - Summary CSV: mean/min/max R^2 for each test + number and share of evaluated models.
      - Per-state CSV: R^2 for each evaluated state.

    Returns:
      summary_df, per_state_df, summary_path, per_state_path
    """
    files_by_sex_race = context["files_by_sex_race"]
    lastschool_horizon = int(context["lastschool_horizon"])
    educ_major = generate_educ_major_pairs(fields=fields)

    # Construct full list of candidate states (some will be missing in the NPZs and skipped)
    all_states = []
    for (sex, race), _flist in sorted(files_by_sex_race.items()):
        for lastschool in range(0, lastschool_horizon):
            for educ, major in educ_major:
                all_states.append((sex, race, lastschool, educ, major))

    evaluation_stride = max(1, int(evaluation_stride))
    evaluated_states = all_states[::evaluation_stride]

    rows = []
    n_evaluated = 0

    for (sex, race, lastschool, educ, major) in evaluated_states:
        try:
            sigmas, debts, EVT = load_evt_grid(
                files_by_sex_race, debt_grid, sex, race, lastschool, educ, major
            )
        except KeyError:
            # This state does not exist in the NPZ key set
            continue

        ns, nd = EVT.shape
        if ns < 3 or nd < 3:
            # Not enough grid points to hold out an interior slice without extrapolation
            rows.append({
                "sex": sex, "race": race, "lastschool": lastschool, "educ": educ, "major": major,
                "r2_debt_given_sigma": np.nan,
                "r2_sigma_given_debt": np.nan,
                "nsigma": ns, "ndebt": nd
            })
            continue

        # Choose one sigma index (interior) for the hold-out test
        sigma_idx = ns // 2
        sigma_idx = max(1, min(sigma_idx, ns - 2))

        # Choose one debt index (interior) for the hold-out test
        if 0 <= heldout_debt_index < nd:
            debt_idx = int(heldout_debt_index)
        else:
            debt_idx = nd // 2
        debt_idx = max(1, min(debt_idx, nd - 2))

        sigma_holdout = float(sigmas[sigma_idx])
        debt_holdout = float(debts[debt_idx])

        # Test 1: hold out a sigma slice and predict the debt profile at that sigma
        mask_sigma = np.ones(ns, dtype=bool)
        mask_sigma[sigma_idx] = False
        interp_sigma = build_grid_interpolator(sigmas[mask_sigma], debts, EVT[mask_sigma, :])

        real_debt_curve = EVT[sigma_idx, :]
        pred_debt_curve = predict_curve_over_debt(interp_sigma, sigma_holdout, debts)
        r2_debt_given_sigma = r2_score(real_debt_curve, pred_debt_curve)

        # Test 2: hold out a debt slice and predict the sigma profile at that debt
        mask_debt = np.ones(nd, dtype=bool)
        mask_debt[debt_idx] = False
        interp_debt = build_grid_interpolator(sigmas, debts[mask_debt], EVT[:, mask_debt])

        real_sigma_curve = EVT[:, debt_idx]
        pred_sigma_curve = predict_curve_over_sigma(interp_debt, sigmas, debt_holdout)
        r2_sigma_given_debt = r2_score(real_sigma_curve, pred_sigma_curve)

        rows.append({
            "sex": sex, "race": race, "lastschool": lastschool, "educ": educ, "major": major,
            "sigma_holdout": sigma_holdout,
            "debt_holdout": debt_holdout,
            "r2_debt_given_sigma": r2_debt_given_sigma,
            "r2_sigma_given_debt": r2_sigma_given_debt,
            "nsigma": ns, "ndebt": nd
        })
        n_evaluated += 1

    per_state = pd.DataFrame(rows)

    n_total_candidates = len(all_states)
    share_evaluated = (n_evaluated / n_total_candidates) if n_total_candidates else np.nan

    summary = pd.DataFrame([{
        "mean_r2_debt_given_sigma": round(float(per_state["r2_debt_given_sigma"].mean(skipna=True)), 3),
        "min_r2_debt_given_sigma": round(float(per_state["r2_debt_given_sigma"].min(skipna=True)), 3),
        "max_r2_debt_given_sigma": round(float(per_state["r2_debt_given_sigma"].max(skipna=True)), 3),
        "mean_r2_sigma_given_debt": round(float(per_state["r2_sigma_given_debt"].mean(skipna=True)), 3),
        "min_r2_sigma_given_debt": round(float(per_state["r2_sigma_given_debt"].min(skipna=True)), 3),
        "max_r2_sigma_given_debt": round(float(per_state["r2_sigma_given_debt"].max(skipna=True)), 3),
        "n_models_total_candidates": int(n_total_candidates),
        "n_models_evaluated": int(n_evaluated),
        "share_models_evaluated": round(float(share_evaluated), 3) if np.isfinite(share_evaluated) else np.nan,
        "evaluation_stride": int(evaluation_stride),
        "heldout_debt_index": int(heldout_debt_index)
    }])

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, out_name)
    per_state_path = os.path.join(out_dir, out_name.replace(".csv", "_by_state.csv"))

    summary.to_csv(summary_path, index=False)

    per_state_out = per_state.copy()
    per_state_out["r2_debt_given_sigma"] = per_state_out["r2_debt_given_sigma"].round(3)
    per_state_out["r2_sigma_given_debt"] = per_state_out["r2_sigma_given_debt"].round(3)
    per_state_out.to_csv(per_state_path, index=False)

    return summary, per_state_out, summary_path, per_state_path


# ---------------------------------------------------------------------
# Script entry point (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")
    from config import DIR
    from model_solution_em import get_debt_range

    pathcont = DIR["MODEL_CONTINUATION_FINAL"]
    pathout  = DIR["MODEL_OUTPUT"]
    debt_grid = np.asarray(get_debt_range(), dtype=float).ravel()

    # 1) Build + save cache (returns None)
    build_interp_cache(
        pathout=pathout,
        pathcont=pathcont,
        debt_grid=debt_grid,
        fields=8,
        force_rebuild=False,
        verbose=True
    )

    # 2) Load cache (returns interp_dict)
    interp_dict = load_interp_cache(pathout=pathout)
    print("Loaded interpolators from cache:", len(interp_dict))

    # Optional validation (still uses the underlying NPZ grid)
    # If you want validation, build context in-memory:
    interp_dict2, meta_dict, missing, context = build_interpolator_dictionary(
        pathcont=pathcont,
        debt_grid=debt_grid,
        fields=8,
        lastschool_horizon=None,
        sex_filters=None,
        race_filters=None,
        verbose=False
    )

    summary, per_state, p_sum, p_state = validate_interpolators(
        context=context,
        debt_grid=debt_grid,
        fields=8,
        evaluation_stride=10,
        heldout_debt_index=50,
        out_dir=r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Output\Tables",
        out_name="interp_validation_r2.csv"
    )

    print("Saved summary table:", p_sum)
    print("Saved per-state table:", p_state)
    print(summary)