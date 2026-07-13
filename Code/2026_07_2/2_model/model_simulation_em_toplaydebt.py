"""
Full pipeline:

1. Load "normal" real data (state_t*.npy etc. are assumed to exist).
2. Use model_em_algorithm.get_feasible_pubid() to:
   - identify feasible PUBIDs (legal states + feasible choices),
   - build the superfeasible panel,
   - save state_superfeasible_t*.npy, income_superfeasible_t*.npy, etc.
3. Load the superfeasible data.
4. (Optional) Run debt-fit across all periods using REAL wages/transfers/grants.

This script just orchestrates calls to functions already defined in:
- model_em_algorithm.py
- model_simulation_em_toplaydebt.py
"""

import os
import numpy as np

import model_em_algorithm as em
import model_simulation_em as ms


# --------------------------------------------------------
# 0. Build superfeasible sample: normal → feasible → super
# --------------------------------------------------------

def build_superfeasible_from_normal():
    """
    Calls the existing feasibility pipeline in model_em_algorithm:

    - get_data_pubid(period) loads the original ("normal") real data.
    - get_feasible_pubid():
        * finds, period by period, the PUBIDs that lie in feasible states
          and make feasible choices,
        * intersects across periods (get_superfeasible),
        * stores only those PUBIDs in superfeasible files via
          get_data_superfeasible().

    After calling this, you will have:
        state_superfeasible_t{t}.npy
        invariant_state_superfeasible_t{t}.npy
        debt_superfeasible_t{t}.npy
        choice_superfeasible_t{t}.npy
        income_superfeasible_t{t}.npy
        debtchoice_superfeasible_t{t}.npy
    in em.path (DIR["MODEL_REALDATA"]).
    """
    print("=== Building superfeasible sample from NORMAL real data ===")
    em.get_feasible_pubid()
    print("=== Superfeasible files saved in:", em.path, "===\n")


# --------------------------------------------------------
# 1. Loader: superfeasible real data for period t
# --------------------------------------------------------

def keep_educ_deciders(
    x1, x2, choice_tuples,
    b_old_idx, b_old_amt,
    b_next_idx, b_next_amt,
    w_real, grants_real, transfers_real,
    types
):
    """
    Keep only observations where the individual is making an
    educational decision (educ > 0). For non-enrolled individuals
    (educ == 0), student loans are not a choice, so we drop them
    from the debt-choice fit.
    """

    # educ is the second column of choice_tuples: [field, educ, work]
    educ = choice_tuples[:, 1].astype(int)
    mask = educ > 0

    # If no one is making an educational decision in this period,
    # return empty arrays to signal skipping this period.
    if mask.sum() == 0:
        return (None,) * 11

    x1_f            = x1[mask, :]
    x2_f            = x2[mask, :]
    choice_f        = choice_tuples[mask, :]
    b_old_idx_f     = b_old_idx[mask]
    b_old_amt_f     = b_old_amt[mask]
    b_next_idx_f    = b_next_idx[mask]
    b_next_amt_f    = b_next_amt[mask]
    w_real_f        = w_real[mask]
    grants_real_f   = grants_real[mask]
    transfers_real_f= transfers_real[mask]
    types_f         = types[mask]

    return (x1_f, x2_f, choice_f,
            b_old_idx_f, b_old_amt_f,
            b_next_idx_f, b_next_amt_f,
            w_real_f, grants_real_f, transfers_real_f,
            types_f)

SENTINEL_UTILITY = -100000  # same value used in get_debt_rules for infeasible choices


def safe_argmax_matrix(U, sentinel=SENTINEL_UTILITY):
    """
    U : 2D array of utilities, shape (N, B) where N = individuals, B = debt grid size.

    Returns
    -------
    idx : int array of length N
        For each row i, returns the index of the best *feasible* choice,
        treating all entries <= sentinel as infeasible.

    If an entire row is infeasible (all <= sentinel), returns np.nan for that row.
    """
    N, B = U.shape
    out = np.empty(N, dtype=float)

    for i in range(N):
        u_row = U[i, :]
        # Feasible = strictly greater than sentinel
        feasible = u_row > sentinel

        if not feasible.any():
            out[i] = np.nan
            continue

        feasible_idx = np.where(feasible)[0]
        # argmax only over feasible entries
        best_local = np.argmax(u_row[feasible])
        out[i] = feasible_idx[best_local]

    return out.astype(int)


def recover_epsilons(
    sigma_u,
    x1,
    x2,
    b_old_idx,
    w_real,
    grants_real,
    transfers_real,
    choice_tuples,
    debt_observed_idx,
    period,
    conterfactual,
    types,
    maxdebt,
    eps_grid=None
):
    """
    For each individual, find eps that rationalizes observed debt, under:

        C(b') = c0 + (1+eps)*b'

    Strategy:
      - For each eps in eps_grid, compute optimal debt index using safe_argmax_matrix.
      - For borrowers: choose eps where predicted index == observed index (or closest).
      - For non-borrowers: find smallest eps where predicted debt > 0
        (threshold eps_bar at which they start borrowing → left-censored).

    Returns:
      eps_hat_borrowers : array of eps for those with observed debt > 0
      eps_hat_all       : array of eps or thresholds (same length as N)
      is_borrower       : boolean mask
    """

    N = x1.shape[0]
    debt_grid = ms.get_debt_range()
    j_idx = map_choice_tuples_to_index(choice_tuples)

    if eps_grid is None:
        # Adjust the range / resolution as you like
        eps_grid = np.linspace(-0.5, 0.5, 101)

    # Store predicted debt index for each eps: shape (len(eps_grid), N)
    pred_idx_over_eps = np.empty((len(eps_grid), N), dtype=int)

    for k, eps in enumerate(eps_grid):
        vjt_eps = get_conditional_agents_real_eps(
            sigma_u=sigma_u,
            x1=x1,
            x2=x2,
            b_idx=b_old_idx,
            w_real=w_real,
            grants_real=grants_real,
            transfers_real=transfers_real,
            j_idx=j_idx,
            period=period,
            conterfactual=conterfactual,
            types=types,
            maxdebt=maxdebt,
            eps=eps
        )
        # SAFE argmax across debt grid, respecting infeasibility
        pred_idx_over_eps[k, :] = safe_argmax_matrix(vjt_eps)

    # Borrowers vs non-borrowers (based on observed next-period debt)
    is_borrower = debt_grid[debt_observed_idx] > 0

    eps_hat_all = np.empty(N)
    eps_hat_all[:] = np.nan

    # Loop over individuals
    for i in range(N):
        obs_idx = debt_observed_idx[i]

        # All eps where model predicts the observed index
        matches = np.where(pred_idx_over_eps[:, i] == obs_idx)[0]

        if is_borrower[i]:
            # Borrower: want eps such that predicted index == observed index
            if len(matches) > 0:
                # Take the average eps across the matching region
                eps_hat_all[i] = eps_grid[matches].mean()
            else:
                # No exact match on the grid: choose eps where predicted index is closest
                diffs = np.abs(pred_idx_over_eps[:, i] - obs_idx)
                k_star = np.argmin(diffs)
                eps_hat_all[i] = eps_grid[k_star]
        else:
            # Non-borrower: threshold eps at which they start borrowing
            debt_amt_over_eps = debt_grid[pred_idx_over_eps[:, i]]
            pos = np.where(debt_amt_over_eps > 0)[0]
            if len(pos) > 0:
                # First eps where predicted debt becomes positive
                eps_hat_all[i] = eps_grid[pos[0]]
            else:
                # Even at max eps they never borrow → right-censored
                eps_hat_all[i] = eps_grid[-1]

    eps_hat_borrowers = eps_hat_all[is_borrower]

    return eps_hat_borrowers, eps_hat_all, is_borrower

def load_real_data_superfeasible(period: int):

    # --- LOAD from superfeasible panel ---
    x1, x2, b_old_idx, choice_tuples = em.load_data_superfeasible(period)
    income = np.load(os.path.join(em.path, f"income_superfeasible_t{period}.npy"))
    b_next_idx = np.load(os.path.join(em.path, f"debtchoice_superfeasible_t{period}.npy"))

    # --- Debt grid ---
    debt_grid = em.get_debt_range()
    b_old_amt  = debt_grid[b_old_idx.astype(int)]
    b_next_amt = debt_grid[b_next_idx.astype(int)]

    # ---------------------------------------------------------
    #  INCOME DECODING  (matches your Stata→Python pipeline)
    # ---------------------------------------------------------
    # Columns:
    # 0 = log hourly wage
    # 1 = total grants
    # 2 = parental help
    # 3 = parental loan
    # ---------------------------------------------------------
    w_loghour   = income[:, 0]
    grants_real = income[:, 1]
    transfers_real = income[:, 2] + income[:, 3]

    # ---------------------------------------------------------
    #  ANNUALIZE WAGES BASED ON WORK CHOICE
    # ---------------------------------------------------------
    # work: 0=no work, 1=part-time, 2=full-time
    work = choice_tuples[:, 2].astype(int)

    hours_week = np.where(work == 2, 40,
                   np.where(work == 1, 20, 0))

    # Annual wage
    w_real = np.exp(w_loghour) * hours_week * 52

    # ---------------------------------------------------------
    #  Types (placeholder)
    # ---------------------------------------------------------
    types = np.ones(x1.shape[0], dtype=int)

    return (x1, x2, choice_tuples,
            b_old_idx.astype(int), b_old_amt,
            b_next_idx.astype(int), b_next_amt,
            w_real, grants_real, transfers_real,
            types)


# --------------------------------------------------------
# 2. Map observed choice tuples to model's choice index
# --------------------------------------------------------

def map_choice_tuples_to_index(choice_tuples):
    """
    Map observed discrete choices (field, educ_status, work) to the
    row index in ms.get_total_choices().
    """
    total_choices = ms.get_total_choices()  # (J,3)
    total_choices = np.asarray(total_choices, dtype=int)
    choice_tuples = np.asarray(choice_tuples, dtype=int)

    matches = (total_choices[None, :, :] == choice_tuples[:, None, :])
    matches = matches.all(axis=2)   # (N,J)

    j_idx = matches.argmax(axis=1)  # first True in each row
    return j_idx


# --------------------------------------------------------
# 3. Real-data utility & conditional value over debt grid
# --------------------------------------------------------

def get_utility_agents_real(sigma_u, c0, b1, b_idx, maxdebt):
    """
    Flow utility given REAL flows (w, grants, transfers):

        c0_i = h_real_i + w_real_i - (1+r)*b_old_i - tuition_i
        c_i(b') = c0_i + b'

    Args:
        sigma_u : risk aversion
        c0      : (N,) baseline resources
        b1      : (B,) debt grid
        b_idx   : (N,) current debt index
        maxdebt : boolean, same as in your model

    Returns:
        u       : (N,B) flow utility for each (i, b') pair
    """
    # Expand c0 over all b'
    c = c0[..., None]
    c = np.repeat(c, b1.shape[0], axis=1)  # (N,B)

    # Add candidate future debt levels b'
    b2 = np.broadcast_to(b1, c.shape)
    c = c + b2

    # CRRA utility
    u = 0.1 * ((0.00001 * c) ** (1 - sigma_u) / (1 - sigma_u))

    # Enforce debt rules from the structural model
    u = ms.get_debt_rules(c, u, b_idx, maxdebt)
    return u


def get_conditional_agents_real(
    sigma_u,
    x1,
    x2,
    b_idx,
    w_real,
    grants_real,
    transfers_real,
    j_idx,
    period,
    conterfactual,
    types,
    maxdebt
):
    """
    Conditional value (over b') for the OBSERVED discrete choice j,
    using REAL wages, grants and transfers.

    Returns:
        vjt_real : (N,B) value for each candidate next-period debt b'
    """
    # Debt grid & interest rate
    b1 = ms.get_debt_range()
    debt_range = ms.debt_range
    r = 0.05

    # Choice tuples for each individual
    total_choices = ms.get_total_choices()
    j = total_choices[j_idx, :]  # (N,3)

    # Tuition for observed choice j
    tuition = ms.tuition_agents(conterfactual, j)  # (N,)

    # Current debt amount
    b_old_amount = debt_range[b_idx.astype(int)]  # (N,)

    # Real help + real wages
    h_real = grants_real + transfers_real
    c0 = h_real + w_real - (1 + r) * b_old_amount - tuition  # (N,)

    # Flow utility for all b'
    u = get_utility_agents_real(sigma_u, c0, b1, b_idx, maxdebt)

    # Continuation value from solved V_t
    continuation = ms.beta * ms.VT_agents(x1, x2, b1, period, j,
                                          conterfactual, types, maxdebt)

    vjt_real = u + continuation
    return vjt_real


# --------------------------------------------------------
# 4. Debt fit statistics (on superfeasible sample)
# --------------------------------------------------------

def compute_debt_fit(
    sigma_u,
    x1,
    x2,
    b_old_idx,
    w_real,
    grants_real,
    transfers_real,
    choice_tuples,
    debt_observed_idx,
    period,
    conterfactual,
    types,
    maxdebt
):
    """
    Computes ONLY the SMM-style moments for debt:

        m1_data  = share with b_{t+1} > 0 in the real data
        m2_data  = mean debt among indebted individuals, real data

        m1_model = share with b' > 0 from model prediction
        m2_model = mean predicted debt among indebted individuals

    Uses safe_argmax_matrix so that infeasible debt choices (utility = SENTINEL_UTILITY)
    are never chosen as optimal.
    """

    debt_grid = ms.get_debt_range()

    # Observed discrete-choice index
    j_idx = map_choice_tuples_to_index(choice_tuples)

    # Conditional values over b'
    vjt_real = get_conditional_agents_real(
        sigma_u=sigma_u,
        x1=x1,
        x2=x2,
        b_idx=b_old_idx,
        w_real=w_real,
        grants_real=grants_real,
        transfers_real=transfers_real,
        j_idx=j_idx,
        period=period,
        conterfactual=conterfactual,
        types=types,
        maxdebt=maxdebt
    )

    # Predicted optimal debt (index & amount) using SAFE argmax
    debt_pred_idx = safe_argmax_matrix(vjt_real)
    debt_pred_amt = debt_grid[debt_pred_idx]

    # Real observed next-period debt
    debt_true_idx = debt_observed_idx
    debt_true_amt = debt_grid[debt_true_idx]

    # -------------------------
    #  MOMENTS (real data)
    # -------------------------
    m1_data = np.mean(debt_true_amt > 0)
    if np.any(debt_true_amt > 0):
        m2_data = np.mean(debt_true_amt[debt_true_amt > 0])
    else:
        m2_data = np.nan

    # -------------------------
    #  MOMENTS (model)
    # -------------------------
    m1_model = np.mean(debt_pred_amt > 0)
    if np.any(debt_pred_amt > 0):
        m2_model = np.mean(debt_pred_amt[debt_pred_amt > 0])
    else:
        m2_model = np.nan

    stats = {
        "m1_share_debt_data": float(m1_data),
        "m2_mean_debt_indebted_data": float(m2_data),
        "m1_share_debt_model": float(m1_model),
        "m2_mean_debt_indebted_model": float(m2_model),
    }

    details = {
        "debt_pred_idx": debt_pred_idx,
        "debt_pred_amt": debt_pred_amt,
        "debt_true_idx": debt_true_idx,
        "debt_true_amt": debt_true_amt,
    }

    return stats, details


def get_utility_agents_real_eps(sigma_u, c0, b1, b_idx, maxdebt, eps):
    """
    Flow utility given REAL flows + per-dollar budget shock eps:

        C(b') = c0 + (1+eps)*b'

    where:
        c0_i = h_real_i + w_real_i - (1+r)*b_old_i - tuition_i

    eps is a scalar applied to all individuals in this evaluation.
    """

    # Expand c0 over all b'
    c = c0[..., None]                        # (N,1)
    c = np.repeat(c, b1.shape[0], axis=1)    # (N,B)

    # Debt grid
    b2 = np.broadcast_to(b1, c.shape)        # (N,B)

    # Budget shock interacts linearly with b'
    c = c + (1.0 + eps) * b2

    # CRRA utility
    u = 0.1 * ((0.00001 * c) ** (1 - sigma_u) / (1 - sigma_u))

    # Enforce debt rules
    u = ms.get_debt_rules(c, u, b_idx, maxdebt)
    return u


def get_conditional_agents_real_eps(
    sigma_u,
    x1,
    x2,
    b_idx,
    w_real,
    grants_real,
    transfers_real,
    j_idx,
    period,
    conterfactual,
    types,
    maxdebt,
    eps
):
    """
    Conditional value (over b') for the OBSERVED discrete choice j,
    with REAL wages/transfers AND a per-dollar budget shock eps.
    """

    b1 = ms.get_debt_range()
    debt_range = ms.debt_range
    r = 0.05

    total_choices = ms.get_total_choices()
    j = total_choices[j_idx, :]  # (N,3)

    # Tuition
    tuition = ms.tuition_agents(conterfactual, j)  # (N,)

    # Current debt amount
    b_old_amount = debt_range[b_idx.astype(int)]   # (N,)

    # Real resources before new borrowing
    h_real = grants_real + transfers_real
    c0 = h_real + w_real - (1 + r) * b_old_amount - tuition  # (N,)

    # Flow utility with eps
    u = get_utility_agents_real_eps(sigma_u, c0, b1, b_idx, maxdebt, eps)

    # Continuation value unchanged
    continuation = ms.beta * ms.VT_agents(x1, x2, b1, period, j,
                                          conterfactual, types, maxdebt)

    vjt = u + continuation
    return vjt

# --------------------------------------------------------
# 5. Main: FULL LOOP over periods 1–10
# --------------------------------------------------------

def main():
    # build_superfeasible_from_normal()  # if you want to rebuild from raw data

    conterfactual = 0
    maxdebt = True
    eps_grid = np.linspace(-200000, 20000, 201) 

    all_stats = {}
    all_eps   = []   # to stack epsilons across periods

    for period in range(1, 11):
        print(f"\n=== Period {period} ===")

        (x1, x2, choice_tuples,
         b_old_idx, b_old_amt,
         b_next_idx, b_next_amt,
         w_real, grants_real, transfers_real,
         types) = load_real_data_superfeasible(period)

        # Keep only individuals making an educational decision (educ > 0)
        (x1_f, x2_f, choice_f,
         b_old_idx_f, b_old_amt_f,
         b_next_idx_f, b_next_amt_f,
         w_real_f, grants_real_f, transfers_real_f,
         types_f) = keep_educ_deciders(
            x1, x2, choice_tuples,
            b_old_idx, b_old_amt,
            b_next_idx, b_next_amt,
            w_real, grants_real, transfers_real,
            types
        )

        if x1_f is None:
            print("No individuals making an educational decision in this period. Skipping.")
            continue

        # 1) Debt moments (share indebted & mean among indebted)
        stats, details = compute_debt_fit(
            sigma_u=ms.sigma_u,
            x1=x1_f,
            x2=x2_f,
            b_old_idx=b_old_idx_f,
            w_real=w_real_f,
            grants_real=grants_real_f,
            transfers_real=transfers_real_f,
            choice_tuples=choice_f,
            debt_observed_idx=b_next_idx_f,
            period=period,
            conterfactual=conterfactual,
            types=types_f,
            maxdebt=maxdebt
        )

        all_stats[period] = stats

        for k, v in stats.items():
            print(f"{k:35s}: {v:.4f}")

        # 2) Recover individual-level budget shocks epsilon_i
        eps_hat = recover_epsilons(
            sigma_u=ms.sigma_u,
            x1=x1_f,
            x2=x2_f,
            b_old_idx=b_old_idx_f,
            w_real=w_real_f,
            grants_real=grants_real_f,
            transfers_real=transfers_real_f,
            choice_tuples=choice_f,
            debt_observed_idx=b_next_idx_f,
            period=period,
            conterfactual=conterfactual,
            types=types_f,
            maxdebt=maxdebt,
            eps_grid=eps_grid
        )
        import matplotlib.pyplot as plt
        plt.hist(eps_hat)

        all_eps.append(eps_hat)

    # Stack epsilons across all periods
    if all_eps:
        eps_all = np.concatenate(all_eps)
        np.save("budget_shocks_eps_hat.npy", eps_all)
        print("\nSaved recovered budget shocks to budget_shocks_eps_hat.npy")

    # Optionally also save the moments
    # np.save("debt_fit_stats_superfeasible_educonly.npy", all_stats)


main()



main()


