# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:21:25 2026

@author: S.Quintana
"""

import os
os.chdir(r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Code\2026_01\2_model")
from config import DIR
pathcont  = DIR["MODEL_CONTINUATION_FINAL"]
import numpy as np
import matplotlib.pyplot as plt


sigma_u = 1.35 
lastschool = 5
educ = 4
major = 1 
evt = np.load(f"{pathcont}/continuation_s{0}_eth{0}_sigma{sigma_u}.npz")[f"con_last{lastschool}_educ{educ}_major{major}"]

 
plt.plot(evt)


import numpy as np
import matplotlib.pyplot as plt

def _design_matrix_pw_quadratic(x, tau):
    """Build X = [1, x, x^2, (x-tau)+, (x-tau)+^2]."""
    x = np.asarray(x, dtype=float)
    h1 = np.maximum(0.0, x - float(tau))
    h2 = h1**2
    return np.column_stack([np.ones_like(x), x, x**2, h1, h2])

def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    # Handle constant y edge case
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def fit_quadratic_with_breakpoint_cv(evt, k=5, min_frac=0.1, max_frac=0.9, seed=0):
    """
    Quadratic with a data-driven breakpoint tau:

        y = a + b*x + c*x^2 + d*(x-tau)_+ + e*(x-tau)_+^2 + error

    Breakpoint tau is chosen to minimize K-fold CV SSE.
    Returns:
      - tau, coef, yhat_in_sample
      - cv_r2 (predictive R^2 from out-of-fold predictions)
      - oof_pred (out-of-fold preds)
    """
    y = np.asarray(evt, dtype=float).ravel()
    n = y.shape[0]
    x = np.arange(n, dtype=float)

    # Candidate tau range (avoid edges)
    lo = int(np.floor(min_frac * n))
    hi = int(np.ceil(max_frac * n))
    lo = max(lo, 1)
    hi = min(hi, n - 2)
    taus = np.arange(lo, hi + 1)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # Build folds (contiguous chunks of shuffled indices)
    folds = np.array_split(idx, k)

    best_tau = None
    best_cv_sse = np.inf

    # Select tau by CV SSE
    for tau in taus:
        cv_sse = 0.0
        for fi in range(k):
            test_idx = folds[fi]
            train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

            Xtr = _design_matrix_pw_quadratic(x[train_idx], tau)
            ytr = y[train_idx]
            coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

            Xte = _design_matrix_pw_quadratic(x[test_idx], tau)
            yte = y[test_idx]
            pred = Xte @ coef
            cv_sse += float(np.sum((yte - pred) ** 2))

        if cv_sse < best_cv_sse:
            best_cv_sse = cv_sse
            best_tau = int(tau)

    # Out-of-fold predictions using the selected tau
    oof_pred = np.empty(n, dtype=float)
    for fi in range(k):
        test_idx = folds[fi]
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        Xtr = _design_matrix_pw_quadratic(x[train_idx], best_tau)
        ytr = y[train_idx]
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

        Xte = _design_matrix_pw_quadratic(x[test_idx], best_tau)
        oof_pred[test_idx] = Xte @ coef

    cv_r2 = _r2(y, oof_pred)

    # Final fit on all data (for a smooth fitted curve to plot)
    Xall = _design_matrix_pw_quadratic(x, best_tau)
    coef_all, *_ = np.linalg.lstsq(Xall, y, rcond=None)
    yhat_all = Xall @ coef_all

    return {
        "tau": best_tau,
        "coef": coef_all,      # [a, b, c, d, e]
        "yhat": yhat_all,      # in-sample fitted curve (for plotting)
        "cv_r2": cv_r2,        # predictive R^2 (out-of-fold)
        "oof_pred": oof_pred,  # out-of-fold predictions
        "cv_sse": best_cv_sse,
    }


# ---------------------------
# Usage (x = 0..99)
# ---------------------------
evt = np.asarray(evt)  # your numpy array, length 100
x = np.arange(100, dtype=float)

out = fit_quadratic_with_breakpoint_cv(evt, k=5, seed=0)

print("Estimated breakpoint τ:", out["tau"])
print("Coefficients [a, b, c, d, e]:", out["coef"])
print("Predictive R^2 (5-fold OOF):", out["cv_r2"])

# Plot: real vs fitted + breakpoint
plt.figure(figsize=(9, 4.5))
plt.plot(x, evt, label="Real (evt)")
plt.plot(x, out["yhat"], label="Quadratic with breakpoint (fit)")
plt.axvline(out["tau"], linestyle="--", label=f"Breakpoint τ={out['tau']}")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Quadratic-with-breakpoint fit | Predictive R²={out['cv_r2']:.3f}")
plt.legend()
plt.tight_layout()
plt.show()
