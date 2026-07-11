# -*- coding: utf-8 -*-
"""
Fix debt (index on the evt grid), discover ALL continuation_*.npz files in pathcont,
parse sigma_u robustly from filenames, load the right key, and plot evt[debt_idx] vs sigma_u.

This version DOES NOT reconstruct filenames from floats (avoids 2.0 vs 2, rounding, etc.).
It opens the exact files found in the directory.

@author: S.Quintana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Your existing setup
# ---------------------------
os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")
from config import DIR
pathcont = DIR["MODEL_CONTINUATION_FINAL"]

lastschool = 5
educ = 4
major = 1

# optional filters (set to None to not filter)
s_filter = 0
eth_filter = 0

# fixed debt point (index in evt array)
debt_idx = 0  # change

# ---------------------------
# Discovery: find matching files and parse sigma as a string + float
# ---------------------------
def discover_continuation_files(path):
    """
    Find files like:
      continuation_s{S}_eth{ETH}_sigma{SIGMA}.npz
    with SIGMA being ANY string up to '.npz' (including scientific notation).
    Returns list of dicts: {'file','fullpath','s','eth','sigma_str','sigma_val'}
    """
    pat = re.compile(
        r"^continuation_s(?P<s>-?\d+)_eth(?P<eth>-?\d+)_sigma(?P<sigma>.+)\.npz$"
    )
    out = []
    for fn in os.listdir(path):
        if not fn.endswith(".npz"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        sigma_str = m.group("sigma")
        # Try to convert to float (keeps sigma_str regardless)
        try:
            sigma_val = float(sigma_str)
        except ValueError:
            sigma_val = np.nan  # still keep the file; you can handle non-numeric sigma if it exists
        out.append({
            "file": fn,
            "fullpath": os.path.join(path, fn),
            "s": int(m.group("s")),
            "eth": int(m.group("eth")),
            "sigma_str": sigma_str,
            "sigma_val": sigma_val
        })
    return out

def load_evt_from_file(npz_path, lastschool, educ, major):
    key = f"con_last{lastschool}_educ{educ}_major{major}"
    with np.load(npz_path) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not in {os.path.basename(npz_path)}")
        return np.asarray(z[key], dtype=float).ravel()

# ---------------------------
# Main
# ---------------------------
files = discover_continuation_files(pathcont)

# Apply optional filters
if s_filter is not None:
    files = [d for d in files if d["s"] == s_filter]
if eth_filter is not None:
    files = [d for d in files if d["eth"] == eth_filter]

if not files:
    raise FileNotFoundError(
        f"No matching continuation files found in:\n{pathcont}\n"
        f"(after filters s={s_filter}, eth={eth_filter})"
    )

# Keep only those with numeric sigma (recommended for plotting)
num_files = [d for d in files if np.isfinite(d["sigma_val"])]
if not num_files:
    raise ValueError(
        "Found continuation files, but none had sigma that could be parsed as a float.\n"
        "Example sigma strings:\n" + "\n".join(sorted({d["sigma_str"] for d in files})[:10])
    )

# Sort by sigma value
num_files.sort(key=lambda d: d["sigma_val"])

sigmas = []
evt_at_debt = []
used_files = []

for d in num_files:
    evt = load_evt_from_file(d["fullpath"], lastschool, educ, major)

    if debt_idx < 0 or debt_idx >= evt.size:
        raise IndexError(
            f"debt_idx={debt_idx} out of bounds for evt length {evt.size} "
            f"in file {d['file']}"
        )

    sigmas.append(d["sigma_val"])
    evt_at_debt.append(evt[debt_idx])
    used_files.append(d["file"])

sigmas = np.array(sigmas, dtype=float)
evt_at_debt = np.array(evt_at_debt, dtype=float)

print(f"Using {len(sigmas)} files from: {pathcont}")
print("First few files:", used_files[:5])
print("Sigmas:", sigmas)

# Plot
plt.figure(figsize=(9, 4.5))
plt.plot(sigmas, evt_at_debt, marker="o")
plt.xlabel("sigma_u")
plt.ylabel(f"evt[debt_idx={debt_idx}]")
plt.title(
    f"Continuation value at fixed debt across sigma_u\n"
    f"lastschool={lastschool}, educ={educ}, major={major}, s={s_filter}, eth={eth_filter}"
)
plt.tight_layout()
plt.show()

# ---------------------------
# Optional sanity-check: overlay curves across sigma_u
# ---------------------------
# Uncomment if you want to see full evt(debt) curves per sigma
# plt.figure(figsize=(9, 4.5))
# for d in num_files:
#     evt = load_evt_from_file(d["fullpath"], lastschool, educ, major)
#     x = np.arange(evt.size)
#     plt.plot(x, evt, alpha=0.6, label=f"{d['sigma_val']:g}")
# plt.axvline(debt_idx, linestyle="--", label=f"fixed debt_idx={debt_idx}")
# plt.xlabel("debt index")
# plt.ylabel("evt")
# plt.title("evt(debt) across sigma_u (overlay)")
# plt.legend(ncols=2, fontsize=8)
# plt.tight_layout()
# plt.show()
