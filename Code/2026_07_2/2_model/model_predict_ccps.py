# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:13:32 2025

@author: S.Quintana
"""

import numpy as np
import multiprocessing 
import os
import time
from pathlib import Path
from scipy.special import logsumexp

from model_solution_em import (get_all_g,
                               save_npz_here,
                               get_possible_choices,
                               get_x1_new,
                               get_x2,
                               build_param_g)

from model_em_algorithm import (
    MONEY_SCALE,
    get_amount_educ,
    load_fixed_wage_parameters,
    predict_expected_wages,
)
import model_solution_em as ms
from financial_process import (
    expected_grants_vectorized,
    expected_transfers_vectorized,
    load_auxiliary_financial_process,
)
from latent_types import TYPE_IDS, type_components, type_index


from config import DIR

path_estimates  = DIR["MODEL_ESTIMATES"]
path_out = DIR["MODEL_OUTPUT"]

T = 10
INITIAL_CCP_PERIODS = tuple(range(T - 1, 0, -1))
fields=8
occupations = 8
n_param_g_x1 = 9*(1+fields+occupations+1)
n_param_g_work = 2 + 2*fields+occupations+2
n_param_g_change = 1+ fields  + occupations
n_param_g_educ = get_amount_educ()
n_param_g_period = np.maximum((1+fields+occupations)*(T-1-1),0) + np.maximum((T-5-1),0)
n_param_g_period_work = np.maximum((T-1-1)*3,0)  + np.maximum((T-5-1)*2,0)
n_param_g_first = 1 + 4 + 1
n_param_g_exp = (fields-1)*4*6
n_param_g_types = 2 # one effect on associate and one on 4y schools
total_n = n_param_g_x1 + n_param_g_change + n_param_g_work + n_param_g_educ + n_param_g_period + n_param_g_period_work + n_param_g_first + n_param_g_exp + n_param_g_types
total_n_multi = total_n
AUXILIARY_RESULTS_FILE = f"{path_estimates}/auxiliary_em_results.npz"


def get_expected_consumption(x1_new, x2, Jx, type_id, model_parameters):
    """Reproduce the auxiliary EM expected-consumption index for one state."""
    _, grant_type, transfer_type, _ = type_components(type_id)
    choices = np.asarray(Jx, dtype=np.int64)
    x1_design = np.asarray(x1_new, dtype=float).reshape(1, -1)
    state = np.asarray(x2, dtype=float).reshape(1, -1)
    nchoices = len(choices)

    expected_wage = predict_expected_wages(
        x1_design,
        state,
        choices,
        model_parameters["wage_parameters"],
    )[0]
    choice_x1 = np.repeat(x1_design, nchoices, axis=0)
    expected_grant = expected_grants_vectorized(
        choice_x1,
        choices[:, 1],
        choices[:, 2],
        model_parameters["financial_process"]["grant"],
        grant_type=grant_type,
    )
    expected_transfer = expected_transfers_vectorized(
        choice_x1,
        choices[:, 1],
        choices[:, 2],
        model_parameters["financial_process"]["transfer"],
        transfer_type=transfer_type,
    )
    tuition = np.select(
        [choices[:, 1] == 1, choices[:, 1] == 2, choices[:, 1] == 3],
        [4000.0, 8000.0, 14000.0],
        default=0.0,
    )
    expected_consumption = (
        expected_wage + expected_grant + expected_transfer - tuition
    )
    home = np.all(choices == 0, axis=1)
    expected_consumption[home] = 0.0
    return expected_consumption


def get_vjt_static(model_parameters, x1, x1_new, x2, Jx, period, b, type_id):
    """Auxiliary choice index by debt-grid point for one state and joint type."""
    utility_parameters = model_parameters["utility_parameters"]
    g = get_all_g(utility_parameters, x1, x1_new, x2, Jx, period)
    expected_consumption = get_expected_consumption(
        x1_new, x2, Jx, type_id, model_parameters
    )
    choices = np.asarray(Jx)
    nonhome = np.any(choices != 0, axis=1).astype(float)
    debt = np.asarray(b, dtype=float).reshape(-1, 1)
    return (
        g
        + model_parameters["consumption_coefficient"]
        * expected_consumption[None, :]
        / MONEY_SCALE
        + model_parameters["debt_coefficient"]
        * debt
        * nonhome[None, :]
        / MONEY_SCALE
    )
    
    


def get_all_choices(x1, x1_new, x2, b, period, model_parameters, type_id):
    
    # For each state x1 and x2 loop over all possible choices.
    
    Jx  = get_possible_choices(x2)
    # now loop over all the entrances in Jx
    # Create a matrix that will store the values
    
    all_vjt = get_vjt_static(
        model_parameters, x1, x1_new, x2, Jx, period, b, type_id
    )
    
    return all_vjt
 
def get_all_ccps(i, x1, b, model_parameters, type_id):

    """This function computes CCP for choosing home production for every period and
    state using the auxiliary model."""
    
    
    task_started = time.perf_counter()
    total_states = len(x1)
    total_tasks = len(TYPE_IDS) * total_states
    task_number = (type_id - 1) * total_states + i + 1
    # Period T is a terminal-value period in the structural model. Flow-utility
    # period arrays, the likelihood, and CCP continuation sequences all cover
    # periods 1, ..., T-1 only.
    for period in INITIAL_CCP_PERIODS:
        period_started = time.perf_counter()
        # Get set of states
        
        x2_set = get_x2(period)
        
        
        results_ccp = []
        names_ccp = []
        
        for x2 in x2_set:
            x2 = x2.astype("int")
            inv = x1[i,:]
            inv = inv[..., None].T
            # Generate x1
            x1_new = get_x1_new(inv[0])

            all_vjt = get_all_choices(
                inv, x1_new, x2, b, period, model_parameters, type_id
            )

            home = np.flatnonzero(np.all(get_possible_choices(x2) == 0, axis=1))
            if len(home) != 1:
                raise ValueError(
                    f"Expected one home-production alternative; found {len(home)}."
                )
            base = int(home[0])
            all_ccps = np.exp(all_vjt[:, base] - logsumexp(all_vjt, axis=1))
            results_ccp.append(all_ccps)
            names_ccp.append(f"ccp_t{period}_{inv}_{x2.astype(int)}")


        # save_npz_here already anchors relative paths below Model/Output.
        # Passing path_out here would duplicate the absolute root on Linux.
        save_npz_here(
            f"ccp/{period}/ccp_t{period}_{inv}_em{type_id}.npz",
            names_ccp,
            results_ccp,
            compressed=True,
        )
        print(
            f"[Initial CCP | pid={os.getpid()} | task={task_number}/{total_tasks} "
            f"| type={type_id}/{len(TYPE_IDS)} | state={i + 1}/{total_states}] "
            f"completed period {period}/{T} | period={time.perf_counter() - period_started:.2f}s "
            f"| task={time.perf_counter() - task_started:.2f}s",
            flush=True,
        )


def get_all_ccps_debug(i, x1, b, model_parameters, type_id, debug_config):
    """Checked initial-CCP solve used only by ``estimation_all_em_debugger``.

    This deliberately lives beside, rather than inside, :func:`get_all_ccps`.
    The production function therefore retains its exact code path and runtime.
    """
    from model_debug_checks import (
        DebugRecorder,
        ccp_problems,
        finite_problems,
        vjt_problems,
    )

    recorder = DebugRecorder(debug_config, "initial_ccp", type_id, i)
    inv_row = np.asarray(x1[i, :], dtype=int)
    inv = inv_row[None, :]
    x1_new = get_x1_new(inv_row)
    debt = np.asarray(b, dtype=float)

    for period in INITIAL_CCP_PERIODS:
        x2_set = get_x2(period)
        results_ccp = []
        names_ccp = []
        for x2_index, x2_value in enumerate(x2_set):
            x2 = np.asarray(x2_value, dtype=int)
            choices = get_possible_choices(x2)
            metadata = {
                "period": int(period),
                "x1": inv_row,
                "x2_index": int(x2_index),
                "x2": x2,
            }

            expected_consumption = get_expected_consumption(
                x1_new, x2, choices, type_id, model_parameters
            )
            recorder.check(
                "expected_consumption",
                expected_consumption,
                finite_problems(expected_consumption, "expected_consumption"),
                metadata,
            )
            g = get_all_g(
                model_parameters["utility_parameters"],
                inv,
                x1_new,
                x2,
                choices,
                period,
            )
            recorder.check("g", g, finite_problems(g, "g"), metadata)

            # Use the same public numerical kernel as production.
            all_vjt = get_all_choices(
                inv, x1_new, x2, debt, period, model_parameters, type_id
            )
            recorder.check(
                "initial_vjt",
                all_vjt,
                vjt_problems(all_vjt),
                metadata,
                arrays={
                    "x1": inv_row,
                    "x1_new": x1_new,
                    "x2": x2,
                    "debt_grid": debt,
                    "choices": choices,
                    "expected_consumption": expected_consumption,
                    "g": g,
                    "initial_vjt": all_vjt,
                },
            )

            home = np.flatnonzero(np.all(choices == 0, axis=1))
            if len(home) != 1:
                recorder.record(
                    "home_choice_count",
                    {**metadata, "home_choice_count": int(len(home))},
                    arrays={"choices": choices},
                )
                continue
            base = int(home[0])
            home_vjt = all_vjt[:, base]
            home_metadata = {**metadata, "choice_index": base, "choice": choices[base]}
            recorder.check(
                "home_vjt",
                home_vjt,
                finite_problems(home_vjt, "home_vjt"),
                home_metadata,
            )

            log_denom = logsumexp(all_vjt, axis=1)
            recorder.check(
                "log_denom",
                log_denom,
                finite_problems(log_denom, "log_denom"),
                home_metadata,
            )
            home_log_ccp = home_vjt - log_denom
            all_ccps = np.exp(home_log_ccp)
            recorder.check(
                "initial_home_ccp",
                all_ccps,
                ccp_problems(all_ccps),
                home_metadata,
                arrays={
                    "debt_grid": debt,
                    "choices": choices,
                    "initial_vjt": all_vjt,
                    "log_denom": log_denom,
                    "home_log_ccp": home_log_ccp,
                    "initial_home_ccp": all_ccps,
                },
            )
            results_ccp.append(all_ccps)
            names_ccp.append(f"ccp_t{period}_{inv}_{x2}")

        relative_path = f"ccp/{period}/ccp_t{period}_{inv}_em{type_id}.npz"
        save_npz_here(relative_path, names_ccp, results_ccp, compressed=True)
        if debug_config.verify_saved:
            saved_path = Path(path_out) / relative_path
            with np.load(saved_path, allow_pickle=False) as saved:
                expected_keys = set(names_ccp)
                actual_keys = set(saved.files)
                if actual_keys != expected_keys:
                    recorder.record(
                        "saved_ccp_keys_mismatch",
                        {
                            "period": int(period),
                            "missing_keys": sorted(expected_keys - actual_keys),
                            "extra_keys": sorted(actual_keys - expected_keys),
                            "path": saved_path,
                        },
                    )
                for key, expected in zip(names_ccp, results_ccp):
                    if key in saved and not np.array_equal(saved[key], expected):
                        recorder.record(
                            "saved_ccp_value_mismatch",
                            {"period": int(period), "key": key, "path": saved_path},
                            arrays={"expected": expected, "saved": saved[key]},
                        )

    return recorder.finalize()


def initial_ccp_bundle_path(period, invariant_state, type_id, output_root=None):
    """Return the exact initial-CCP filename consumed by the Bellman solver."""
    if output_root is None:
        output_root = DIR["MODEL_OUTPUT"]
    invariant_state = np.asarray(invariant_state)
    return Path(output_root) / "ccp" / str(period) / (
        f"ccp_t{period}_[{invariant_state}]_em{type_id}.npz"
    )


def missing_initial_ccp_tasks(invariant_states, output_root=None):
    """Find type/state tasks lacking at least one required period bundle."""
    missing_tasks = []
    missing_files = []
    for type_id in TYPE_IDS:
        for state_index, invariant_state in enumerate(invariant_states):
            task_missing = []
            for period in INITIAL_CCP_PERIODS:
                filename = initial_ccp_bundle_path(
                    period, invariant_state, type_id, output_root=output_root
                )
                if not filename.is_file():
                    task_missing.append(filename)
            if task_missing:
                missing_tasks.append((state_index, type_id))
                missing_files.extend(task_missing)
    return missing_tasks, missing_files


def ensure_initial_ccps(invariant_states, debt_range, model_parameters):
    """Regenerate incomplete initial-CCP tasks and verify their outputs."""
    missing_tasks, missing_files = missing_initial_ccp_tasks(invariant_states)
    if not missing_tasks:
        print(
            "Initial CCP preflight passed: all required bundles are present",
            flush=True,
        )
        return

    print(
        f"Initial CCP preflight found {len(missing_files)} missing bundles "
        f"across {len(missing_tasks)} type/state tasks; regenerating those tasks",
        flush=True,
    )
    retry_args = [
        (
            state_index,
            invariant_states,
            debt_range,
            model_parameters[type_id],
            type_id,
        )
        for state_index, type_id in missing_tasks
    ]
    with multiprocessing.Pool(processes=min(10, len(retry_args))) as pool_obj:
        pool_obj.starmap(get_all_ccps, retry_args, chunksize=1)

    remaining_tasks, remaining_files = missing_initial_ccp_tasks(invariant_states)
    if remaining_tasks:
        preview = "\n".join(str(path) for path in remaining_files[:10])
        raise FileNotFoundError(
            f"Initial CCP generation still left {len(remaining_files)} required "
            f"bundles missing. First missing paths:\n{preview}"
        )
    print(
        "Initial CCP regeneration completed and passed the preflight",
        flush=True,
    )


def load_utility_parameters(type_id, results_file=AUXILIARY_RESULTS_FILE):
    """Load the complete auxiliary choice index for one joint latent type."""
    type_components(type_id)  # Validate the public one-based joint type ID.
    with np.load(results_file, allow_pickle=False) as results:
        if "choice_parameters" not in results.files:
            raise ValueError(
                f"Auxiliary EM results {results_file} do not contain choice_parameters."
            )
        choice_parameters = np.asarray(results["choice_parameters"], dtype=float)
    expected_size = total_n + 2
    if choice_parameters.shape != (expected_size,):
        raise ValueError(
            f"Auxiliary choice parameters have shape {choice_parameters.shape}; "
            f"expected {(expected_size,)}."
        )

    # build_param_g maps the joint ID to its schooling component. The same ID
    # selects financial components below, keeping one ordering throughout.
    utility_parameters = build_param_g(type_id, choice_parameters[:total_n])
    return {
        "type_id": type_id,
        "type_index": type_index(type_id),
        "utility_parameters": utility_parameters,
        "consumption_coefficient": float(choice_parameters[total_n]),
        "debt_coefficient": float(choice_parameters[total_n + 1]),
        "financial_process": load_auxiliary_financial_process(results_file),
        "wage_parameters": load_fixed_wage_parameters(),
    }



#-----------------------------------------------------------------------------#
# Try the functions


if __name__ == '__main__':

    print("Estimation of the CCPs across periods")
    model_parameters = {
        type_id: load_utility_parameters(type_id)
        for type_id in TYPE_IDS
    }
    debt_range = ms.debt_range
    
    pool_obj = multiprocessing.Pool(processes=10)     
    args = [
        (i, ms.invariant_states, ms.debt_range, model_parameters[type_id], type_id)
        for type_id in TYPE_IDS
        for i in range(np.shape(ms.invariant_states)[0])
    ]
     
    results = pool_obj.starmap(get_all_ccps, args, chunksize=1)
    pool_obj.close()












