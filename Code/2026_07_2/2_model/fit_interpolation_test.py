# -*- coding: utf-8 -*-
"""
Same 2D interpolation exercise as before, but with the REAL debt grid values.

- Uses debt_range = get_debt_range() from model_solution_em
- Interpolator is built on (sigma_u, debt_real) axes.
- Two plots + R^2:
  A) Predict evt across REAL debt at a held-out sigma (LOO sigma)
  B) Predict evt across sigma at a held-out REAL debt point (LOO debt)

Copy-paste runnable.

@author: S.Quintana
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Your existing setup
# ---------------------------
os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")
from config import DIR
pathcont = DIR["MODEL_CONTINUATION_FINAL"]

from model_solution_em import get_debt_range
debt_range = np.asarray(get_debt_range(), dtype=float).ravel()  # REAL debt grid

lastschool = 5
educ = 4
major = 1
s_filter = 0
eth_filter = 0

key = f"con_last{lastschool}_educ{educ}_major{major}"

# Choose which sigma and debt point (by index) to use for the plots
sigma_idx_for_plot = None   # None -> take middle sigma
debt_idx_for_plot  = 50     # index into debt_range (0..len(debt_range)-1)

# ---------------------------
# Helpers: load grid
# ---------------------------
def discover_continuation_files(path):
    pat = re.compile(r"^continuation_s(?P<s>-?\d+)_eth(?P<eth>-?\d+)_sigma(?P<sigma>.+)\.npz$")
    out = []
    for fn in os.listdir(path):
        if not fn.endswith(".npz"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        try:
            sigma_val = float(m.group("sigma"))
        except ValueError:
            continue
        out.append({
            "file": fn,
            "fullpath": os.path.join(path, fn),
            "s": int(m.group("s")),
            "eth": int(m.group("eth")),
            "sigma_val": sigma_val,
        })
    return out

def load_evt_from_file(npz_path, key):
    with np.load(npz_path) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not in {os.path.basename(npz_path)}")
        return np.asarray(z[key], dtype=float).ravel()

files = discover_continuation_files(pathcont)
files = [d for d in files if d["s"] == s_filter and d["eth"] == eth_filter]
if not files:
    raise FileNotFoundError(f"No continuation files found for s={s_filter}, eth={eth_filter} in:\n{pathcont}")

files.sort(key=lambda d: d["sigma_val"])
sigmas = np.array([d["sigma_val"] for d in files], dtype=float)

evt_list = [load_evt_from_file(d["fullpath"], key) for d in files]
n_debt_evt = evt_list[0].size
if any(e.size != n_debt_evt for e in evt_list):
    raise ValueError("Not all evt arrays have the same length (debt grid size).")

if debt_range.size != n_debt_evt:
    raise ValueError(
        f"debt_range length ({debt_range.size}) != evt length ({n_debt_evt}). "
        "Make sure get_debt_range() matches the evt grid."
    )

EVT = np.vstack(evt_list)  # shape (n_sigma, n_debt)

print("Loaded EVT grid shape:", EVT.shape)
print("Sigma values:", sigmas)
print("Debt range: min=", float(debt_range.min()), "max=", float(debt_range.max()))

# Choose plot indices
if sigma_idx_for_plot is None:
    sigma_idx_for_plot = len(sigmas) // 2
sigma_idx_for_plot = int(sigma_idx_for_plot)
debt_idx_for_plot = int(debt_idx_for_plot)

if not (0 <= sigma_idx_for_plot < len(sigmas)):
    raise IndexError("sigma_idx_for_plot out of bounds.")
if not (0 <= debt_idx_for_plot < debt_range.size):
    raise IndexError("debt_idx_for_plot out of bounds.")

sigma0 = float(sigmas[sigma_idx_for_plot])
debt0 = float(debt_range[debt_idx_for_plot])

# ---------------------------
# Helpers: interpolation + R^2
# ---------------------------
def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def make_interpolator(sigmas_axis, debt_axis, values):
    """
    Returns f(sigma, debt)->evt_hat using linear interpolation on a regular grid.
    Prefers SciPy RegularGridInterpolator; falls back to manual bilinear interpolation.
    No extrapolation (errors outside grid).
    """
    sigmas_axis = np.asarray(sigmas_axis, float)
    debt_axis = np.asarray(debt_axis, float)
    values = np.asarray(values, float)

    # Ensure increasing axes
    if not np.all(np.diff(sigmas_axis) > 0):
        order = np.argsort(sigmas_axis)
        sigmas_axis = sigmas_axis[order]
        values = values[order, :]
    if not np.all(np.diff(debt_axis) > 0):
        order = np.argsort(debt_axis)
        debt_axis = debt_axis[order]
        values = values[:, order]

    try:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (sigmas_axis, debt_axis),
            values,
            method="linear",
            bounds_error=True
        )

        def f(sigma, debt):
            sigma_arr = np.atleast_1d(sigma).astype(float)
            debt_arr = np.atleast_1d(debt).astype(float)
            pts = np.column_stack([sigma_arr, debt_arr])
            out = interp(pts)
            return out[0] if np.isscalar(sigma) and np.isscalar(debt) else out

        return f

    except Exception:
        # Manual bilinear interpolation (scalar sigma,debt)
        def f(sigma, debt):
            s = float(sigma)
            d = float(debt)

            i1 = np.searchsorted(sigmas_axis, s)
            if i1 == 0 or i1 == len(sigmas_axis):
                raise ValueError(f"sigma={s} outside [{sigmas_axis.min()}, {sigmas_axis.max()}]")
            i0 = i1 - 1

            j1 = np.searchsorted(debt_axis, d)
            if j1 == 0 or j1 == len(debt_axis):
                raise ValueError(f"debt={d} outside [{debt_axis.min()}, {debt_axis.max()}]")
            j0 = j1 - 1

            s0, s1 = sigmas_axis[i0], sigmas_axis[i1]
            d0, d1 = debt_axis[j0], debt_axis[j1]
            ws = (s - s0) / (s1 - s0) if s1 != s0 else 0.0
            wd = (d - d0) / (d1 - d0) if d1 != d0 else 0.0

            v00 = values[i0, j0]
            v01 = values[i0, j1]
            v10 = values[i1, j0]
            v11 = values[i1, j1]

            v0 = (1 - wd) * v00 + wd * v01
            v1 = (1 - wd) * v10 + wd * v11
            return (1 - ws) * v0 + ws * v1

        return f

def predict_over_debt(interp_fun, sigma_fixed, debt_points):
    debt_points = np.asarray(debt_points, float)
    try:
        return interp_fun(np.full_like(debt_points, float(sigma_fixed)), debt_points)

    except Exception:
        return np.array([interp_fun(float(sigma_fixed), float(d)) for d in debt_points], dtype=float)

def predict_over_sigma(interp_fun, sigma_points, debt_fixed):
    sigma_points = np.asarray(sigma_points, float)
    try:
        return interp_fun(sigma_points, np.full_like(sigma_points, float(debt_fixed)))
    except Exception:
        return np.array([interp_fun(float(s), float(debt_fixed)) for s in sigma_points], dtype=float)

# ---------------------------
# EXERCISE A: debt curve at held-out sigma (sigma interpolation)
# ---------------------------
mask_sigma = np.ones(len(sigmas), dtype=bool)
mask_sigma[sigma_idx_for_plot] = False

# Important: if you hold out an ENDPOINT sigma, interpolation is impossible without extrapolation.
# If that happens, pick an interior sigma index.
if sigma_idx_for_plot in (0, len(sigmas) - 1):
    print("WARNING: You held out an endpoint sigma. Interpolation would require extrapolation.")
    print("         Consider choosing sigma_idx_for_plot inside (1..len(sigmas)-2).")

sigmas_train = sigmas[mask_sigma]
EVT_train_A = EVT[mask_sigma, :]  # keep all debt points

interp_A = make_interpolator(sigmas_train, debt_range, EVT_train_A)

real_debt_curve = EVT[sigma_idx_for_plot, :]
pred_debt_curve = predict_over_debt(interp_A, sigma0, debt_range)

r2_debt_given_sigma = r2(real_debt_curve, pred_debt_curve)

# ---------------------------
# EXERCISE B: sigma curve at held-out debt point (debt interpolation)
# ---------------------------
mask_debt = np.ones(debt_range.size, dtype=bool)
mask_debt[debt_idx_for_plot] = False

# Same caveat: holding out endpoint debt breaks interpolation (needs extrapolation).
if debt_idx_for_plot in (0, debt_range.size - 1):
    print("WARNING: You held out an endpoint debt. Interpolation would require extrapolation.")
    print("         Consider choosing debt_idx_for_plot inside (1..len(debt_range)-2).")

debt_train = debt_range[mask_debt]
EVT_train_B = EVT[:, mask_debt]

interp_B = make_interpolator(sigmas, debt_train, EVT_train_B)

real_sigma_curve = EVT[:, debt_idx_for_plot]
pred_sigma_curve = predict_over_sigma(interp_B, sigmas, debt0)

r2_sigma_given_debt = r2(real_sigma_curve, pred_sigma_curve)

# ---------------------------
# Plots (with REAL debt values)
# ---------------------------
plt.figure(figsize=(9, 4.5))
plt.plot(debt_range, real_debt_curve, label=f"Real (sigma={sigma0:g})")
plt.plot(debt_range, pred_debt_curve, linestyle="--",
         label=f"Predicted (LOO sigma) | R²={r2_debt_given_sigma:.3f}")
plt.xlabel("debt (real values)")
plt.ylabel("evt")
plt.title(
    "Interpolation test: predict evt(debt) at a sigma held out\n"
    f"lastschool={lastschool}, educ={educ}, major={major}, s={s_filter}, eth={eth_filter}"
)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4.5))
plt.plot(sigmas, real_sigma_curve, marker="o", label=f"Real (debt={debt0:g})")
plt.plot(sigmas, pred_sigma_curve, linestyle="--",
         label=f"Predicted (LOO debt) | R²={r2_sigma_given_debt:.3f}")
plt.xlabel("sigma_u")
plt.ylabel(f"evt at debt={debt0:g}")
plt.title(
    "Interpolation test: predict evt(sigma) at a debt held out\n"
    f"lastschool={lastschool}, educ={educ}, major={major}, s={s_filter}, eth={eth_filter}"
)
plt.legend()
plt.tight_layout()
plt.show()

print("---- R^2 results ----")
print(f"R^2 (predict evt across REAL debt | LOO sigma={sigma0:g}): {r2_debt_given_sigma:.6f}")
print(f"R^2 (predict evt across sigma | LOO debt={debt0:g}): {r2_sigma_given_debt:.6f}")
