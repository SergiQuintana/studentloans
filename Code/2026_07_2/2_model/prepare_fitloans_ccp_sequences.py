"""Create missing auxiliary CCPs and CCP sequences for all 16 joint types."""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np

from config import DIR
from latent_types import TYPE_IDS
import model_getccp_sequence as sequence
import model_predict_ccps as initial
import model_solution_em as model


PERIODS = tuple(range(1, 10))


def sequence_bundle_path(period, invariant_state, type_id):
    invariant_state = np.asarray(invariant_state)
    return Path(DIR["MODEL_OUTPUT"]) / "evt_ccp" / str(period) / (
        f"evt_ccp_sequence_t{period}_[{invariant_state}]_em{type_id}.npz"
    )


def missing_sequence_tasks():
    tasks = []
    missing_files = []
    for type_id in TYPE_IDS:
        for state_index, invariant_state in enumerate(model.invariant_states):
            absent = [
                sequence_bundle_path(period, invariant_state, type_id)
                for period in PERIODS
                if not sequence_bundle_path(period, invariant_state, type_id).is_file()
            ]
            if absent:
                tasks.append((state_index, type_id))
                missing_files.extend(absent)
    return tasks, missing_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=10)
    args = parser.parse_args()
    processes = max(1, int(args.processes))

    print("Checking initial auxiliary CCP bundles for all 16 types", flush=True)
    parameters = {
        type_id: initial.load_utility_parameters(type_id)
        for type_id in TYPE_IDS
    }
    initial.ensure_initial_ccps(
        model.invariant_states,
        model.debt_range,
        parameters,
    )

    tasks, missing = missing_sequence_tasks()
    if not tasks:
        print("CCP-sequence preflight passed: no files are missing", flush=True)
        return
    print(
        f"Generating {len(tasks)} missing type/state sequence tasks "
        f"covering {len(missing)} period bundles",
        flush=True,
    )
    worker_args = [
        (state_index, model.invariant_states, model.debt_range, type_id)
        for state_index, type_id in tasks
    ]
    with multiprocessing.Pool(processes=min(processes, len(worker_args))) as pool:
        pool.starmap(sequence.get_ccp_sequence, worker_args, chunksize=1)

    remaining_tasks, remaining_files = missing_sequence_tasks()
    if remaining_tasks:
        preview = "\n".join(str(path) for path in remaining_files[:10])
        raise FileNotFoundError(
            f"CCP sequence generation left {len(remaining_files)} files missing. "
            f"First missing paths:\n{preview}"
        )
    print("All 16-type CCP sequence bundles are complete", flush=True)


if __name__ == "__main__":
    main()
