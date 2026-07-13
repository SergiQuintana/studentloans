# -*- coding: utf-8 -*-
"""
Compare old vs new final utility estimates.

Goal:
- Compare old "sigma1.4" estimates against new "sigma_est" estimates
- Compare both raw parameter vectors and structured utility blocks
  using build_param_g()

This helps diagnose where convergence changed.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import your model module that contains build_param_g and path_estimates
import model_solution_em as ms
from config import path_estimates

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

# Set the final iteration you want to compare
final_iteration = 29

# Adjust these names if your files differ
old_file = Path(path_estimates) / f"estimates_it{final_iteration-1}_sigma_est.npy"
new_file = Path(path_estimates) / f"estimates_it{final_iteration}_sigma_est.npy"

# If your old/new files are instead stored under different exact names,
# replace the two lines above by, e.g.:
# old_file = Path(path_estimates) / "param_g_sigma1.4.npy"
# new_file = Path(path_estimates) / "param_g.npy"

save_csv = False
top_n = 25
em_type = 2   # usually type 2 if you want full type effects included

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def load_vector(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    x = np.load(path)
    return np.asarray(x).reshape(-1)

def summarize_diff(a, b, name="object"):
    diff = b - a
    abs_diff = np.abs(diff)
    return {
        "block": name,
        "shape": str(np.shape(a)),
        "n_elements": a.size,
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "max_abs_diff": float(abs_diff.max()),
        "l2_norm_diff": float(np.linalg.norm(diff)),
        "n_small_1e_6": int(np.sum(abs_diff < 1e-6)),
        "n_small_1e_4": int(np.sum(abs_diff < 1e-4)),
        "n_large_1e_2": int(np.sum(abs_diff > 1e-2)),
    }

def flatten_block(x):
    if isinstance(x, list):
        flat_parts = [np.asarray(z).reshape(-1) for z in x]
        return np.concatenate(flat_parts)
    return np.asarray(x).reshape(-1)

def compare_block(old_block, new_block, block_name):
    old_flat = flatten_block(old_block)
    new_flat = flatten_block(new_block)

    if old_flat.shape != new_flat.shape:
        raise ValueError(
            f"Shape mismatch in {block_name}: "
            f"{old_flat.shape} vs {new_flat.shape}"
        )

    summary = summarize_diff(old_flat, new_flat, block_name)

    detail = pd.DataFrame({
        "index": np.arange(old_flat.size),
        "old": old_flat,
        "new": new_flat,
        "diff": new_flat - old_flat,
        "abs_diff": np.abs(new_flat - old_flat),
    }).sort_values("abs_diff", ascending=False).reset_index(drop=True)

    return summary, detail

# -------------------------------------------------
# LOAD RAW PARAMETER VECTORS
# -------------------------------------------------

old_vec = load_vector(old_file)
new_vec = load_vector(new_file)

if old_vec.shape != new_vec.shape:
    raise ValueError(
        f"Raw vector shapes differ: old {old_vec.shape}, new {new_vec.shape}"
    )

print("=" * 80)
print("RAW VECTOR COMPARISON")
print("=" * 80)
print(f"Old file: {old_file}")
print(f"New file: {new_file}")
print(f"Vector length: {old_vec.size}")

raw_summary = summarize_diff(old_vec, new_vec, name="raw_param_vector")
for k, v in raw_summary.items():
    print(f"{k}: {v}")

raw_detail = pd.DataFrame({
    "param_index": np.arange(old_vec.size),
    "old": old_vec,
    "new": new_vec,
    "diff": new_vec - old_vec,
    "abs_diff": np.abs(new_vec - old_vec),
}).sort_values("abs_diff", ascending=False).reset_index(drop=True)

print(f"\nTop {top_n} raw parameter differences:")
print(raw_detail.head(top_n).to_string(index=False))

# -------------------------------------------------
# BUILD STRUCTURED UTILITY PARAMETERS
# -------------------------------------------------

old_util = ms.build_param_g(em_type, old_vec)
new_util = ms.build_param_g(em_type, new_vec)

block_names = [
    "param_g_x1",
    "param_g_work",
    "param_g_last",
    "param_educ",
    "param_period",
    "param_period_work",
    "param_first",
    "param_exp",
    "param_type",
]

print("\n" + "=" * 80)
print("BLOCK-LEVEL COMPARISON AFTER build_param_g")
print("=" * 80)

all_block_summaries = []
block_details = {}

for name, old_block, new_block in zip(block_names, old_util, new_util):
    summary, detail = compare_block(old_block, new_block, name)
    all_block_summaries.append(summary)
    block_details[name] = detail

    print(f"\n{name}")
    for k, v in summary.items():
        if k != "block":
            print(f"  {k}: {v}")

    print(f"\n  Top {top_n} differences in {name}:")
    print(detail.head(top_n).to_string(index=False))

summary_df = pd.DataFrame(all_block_summaries)

print("\n" + "=" * 80)
print("BLOCK SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

# -------------------------------------------------
# OPTIONAL: inspect nested parts of param_first
# -------------------------------------------------

print("\n" + "=" * 80)
print("PARAM_FIRST SUB-BLOCKS")
print("=" * 80)

param_first_labels = ["param_first2", "param_first4", "param_firstgrad"]
for lbl, old_sub, new_sub in zip(param_first_labels, old_util[6], new_util[6]):
    s, d = compare_block(old_sub, new_sub, lbl)
    print(f"\n{lbl}")
    for k, v in s.items():
        if k != "block":
            print(f"  {k}: {v}")
    print(f"\n  Top {top_n} differences in {lbl}:")
    print(d.head(top_n).to_string(index=False))

# -------------------------------------------------
# SAVE
# -------------------------------------------------

if save_csv:
    raw_detail.to_csv(Path(path_estimates) / "compare_old_new_raw_detail.csv", index=False)
    summary_df.to_csv(Path(path_estimates) / "compare_old_new_block_summary.csv", index=False)

    for name, df in block_details.items():
        df.to_csv(Path(path_estimates) / f"compare_old_new_{name}.csv", index=False)

    print("\nSaved CSV comparison files.")