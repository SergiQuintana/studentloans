# -*- coding: utf-8 -*-
"""
Compare param_g estimates across Aguirregabiria-Mira iterations
and track likelihood evolution.

Loads files like:
    estimates_it1_sigma_est.npy
    likelihood_it1_sigma_est.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import your config path
from config import path_estimates

# -------------------------------------------------
# Settings
# -------------------------------------------------
iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,22,23,24,25,26,27,28,29]
save_csv = False
show_top_changes = 20

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def load_params(iteration, folder):
    file_path = Path(folder) / f"estimates_it{iteration}_sigma_est.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)

def load_likelihood(iteration, folder):
    file_path = Path(folder) / f"likelihood_it{iteration}_sigma_est.npy"
    if not file_path.exists():
        return np.nan  # allows script to continue if one file is missing

    arr = np.load(file_path)

    # Convert scalar array / 0-dim array / 1-element array to float
    if np.ndim(arr) == 0:
        return float(arr)
    elif np.size(arr) == 1:
        return float(arr.reshape(-1)[0])
    else:
        # If somehow a non-scalar got saved, take first element and warn
        print(f"Warning: likelihood file for iteration {iteration} is not scalar. Using first element.")
        return float(arr.reshape(-1)[0])

def compare_two_vectors(p1, p2, it1, it2):
    if p1.shape != p2.shape:
        raise ValueError(
            f"Shape mismatch: iteration {it1} has shape {p1.shape}, "
            f"iteration {it2} has shape {p2.shape}"
        )

    diff = p2 - p1
    abs_diff = np.abs(diff)

    summary = {
        "iteration_from": it1,
        "iteration_to": it2,
        "n_params": len(p1),
        "mean_abs_diff": abs_diff.mean(),
        "median_abs_diff": np.median(abs_diff),
        "max_abs_diff": abs_diff.max(),
        "l2_norm_diff": np.linalg.norm(diff),
        "n_almost_unchanged_1e_8": np.sum(abs_diff < 1e-8),
        "n_small_change_1e_4": np.sum(abs_diff < 1e-4),
        "n_large_change_1e_2": np.sum(abs_diff > 1e-2),
    }

    df = pd.DataFrame({
        "param_index": np.arange(len(p1)),
        f"it_{it1}": p1,
        f"it_{it2}": p2,
        "diff": diff,
        "abs_diff": abs_diff,
    }).sort_values("abs_diff", ascending=False).reset_index(drop=True)

    return summary, df

# -------------------------------------------------
# Load all iterations
# -------------------------------------------------
params = {}
likelihoods = {}

for it in iterations:
    params[it] = load_params(it, path_estimates)
    likelihoods[it] = load_likelihood(it, path_estimates)
    print(f"Loaded iteration {it}: shape = {params[it].shape}, likelihood = {likelihoods[it]}")

# -------------------------------------------------
# Put all parameter vectors into one dataframe
# -------------------------------------------------
param_table = pd.DataFrame({
    "param_index": np.arange(len(params[iterations[0]]))
})

for it in iterations:
    param_table[f"it_{it}"] = params[it]

for i in range(1, len(iterations)):
    it_prev = iterations[i - 1]
    it_curr = iterations[i]
    param_table[f"diff_{it_prev}_to_{it_curr}"] = params[it_curr] - params[it_prev]
    param_table[f"absdiff_{it_prev}_to_{it_curr}"] = np.abs(params[it_curr] - params[it_prev])

# -------------------------------------------------
# Build likelihood evolution table
# -------------------------------------------------
likelihood_table = pd.DataFrame({
    "iteration": iterations,
    "likelihood": [likelihoods[it] for it in iterations],
})

likelihood_table["likelihood_change"] = likelihood_table["likelihood"].diff()
likelihood_table["abs_likelihood_change"] = likelihood_table["likelihood_change"].abs()
likelihood_table["pct_likelihood_change"] = (
    likelihood_table["likelihood"].pct_change() * 100
)

print("\n" + "=" * 70)
print("LIKELIHOOD EVOLUTION")
print("=" * 70)
print(likelihood_table.to_string(index=False))

# -------------------------------------------------
# Print pairwise summaries
# -------------------------------------------------
print("\n" + "=" * 70)
print("PAIRWISE COMPARISON SUMMARY")
print("=" * 70)

all_summaries = []
for i in range(1, len(iterations)):
    it1 = iterations[i - 1]
    it2 = iterations[i]

    summary, detail_df = compare_two_vectors(params[it1], params[it2], it1, it2)

    # Add likelihood info
    ll1 = likelihoods[it1]
    ll2 = likelihoods[it2]
    ll_change = ll2 - ll1 if not (np.isnan(ll1) or np.isnan(ll2)) else np.nan

    summary["likelihood_from"] = ll1
    summary["likelihood_to"] = ll2
    summary["likelihood_change"] = ll_change
    summary["abs_likelihood_change"] = np.abs(ll_change) if not np.isnan(ll_change) else np.nan

    all_summaries.append(summary)

    print(f"\nIteration {it1} -> {it2}")
    for k, v in summary.items():
        if k not in ["iteration_from", "iteration_to"]:
            print(f"  {k}: {v}")

    print(f"\nTop {show_top_changes} largest parameter changes ({it1} -> {it2}):")
    print(detail_df.head(show_top_changes).to_string(index=False))

    if save_csv:
        out_file = Path(path_estimates) / f"compare_it{it1}_to_it{it2}.csv"
        detail_df.to_csv(out_file, index=False)
        print(f"\nSaved detailed comparison: {out_file}")

# -------------------------------------------------
# Save full stacked table
# -------------------------------------------------
if save_csv:
    full_out = Path(path_estimates) / "param_g_all_iterations_comparison.csv"
    param_table.to_csv(full_out, index=False)
    print(f"\nSaved full parameter table: {full_out}")

    ll_out = Path(path_estimates) / "likelihood_evolution.csv"
    likelihood_table.to_csv(ll_out, index=False)
    print(f"Saved likelihood table: {ll_out}")

# -------------------------------------------------
# Summary dataframe
# -------------------------------------------------
summary_df = pd.DataFrame(all_summaries)

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(summary_df.to_string(index=False))

if save_csv:
    summary_out = Path(path_estimates) / "param_g_comparison_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved summary table: {summary_out}")