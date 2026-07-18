# Current model-estimation status

Last updated: 2026-07-18.

This is the entry point for future work on the structural estimation in this
folder. It describes the code that is active now, not the desired final model.
Before changing an economic assumption, moment definition, state transition,
or parameter restriction, confirm it with the researcher. The current stage of
the project is deliberately experimenting only with the budget-shock
parameterization and the SMM loss.

## Non-negotiable data and state conventions

- Parental income is `parinc = x1[:, 0]` and takes values 1, 2, 3, or 4. Do
  not create, load, or substitute `aveparinc` or demographic-average income.
- Ability is `x1[:, 1]` and also takes values 1 through 4.
- Program year is part of the dynamic state `x2`. The pre-choice experience
  columns are 1 for two-year school, 2 for four-year school, and 3 for graduate
  school. The state is never recoded. Only budget-parameter lookup is grouped:
  two-year years 1, 2, 3+; four-year years 1, 2, 3, 4, 5+; graduate years 1,
  2+. `budget_shock.budget_education_cell_from_state` performs this lookup.
- An education-cell code is `100 * education + program_year`, where education
  is 1 (two-year), 2 (four-year), or 3 (graduate).
- The endogenous debt decision remains next-period debt stock. Do not change
  debt grids, state tracking, interest accumulation, annual caps, lifetime
  caps, or education-state transitions as part of a budget-SMM experiment.
- Enrollment, not graduation, defines the sample for education-cell loan
  moments.

## Shared structural interfaces

`budget_shock.py` is the single source of truth for:

- optimizer-vector unpacking;
- education-cell lookup;
- conditional budget-shock means and sigmas;
- the resource slope and its $10,000 scaling;
- fixed simulation draws and Bellman quadrature;
- parental-income risk aversion;
- parental-income debt penalties;
- saving, validation, and loading of the canonical parameter bundle.

The consumers are:

- `model_fitloans_dynamic.py`: SMM data loading, simulations, moments and loss;
- `model_solution_em.py`: backward solution and shock quadrature;
- `model_simulation_em.py`: forward simulation and shock draws.

Do not reproduce the parameter mappings independently in these consumers.
Changes to a distribution must be implemented through `budget_shock.py` so the
estimator, solution, and simulation remain consistent.

`debt_limits.py` is the single source of truth for the interest rate,
consumption floor, annual caps, lifetime caps, and debt-grid bound mappings.
Its primitives are Numba compiled. `model_solution_em.py` retains the fast
Bellman local-window search, `model_fitloans_dynamic.py` retains its parallel
SMM maximizers, and `model_simulation_em.py` retains its forward debt-choice
logic; those consumers import feasibility rules rather than duplicating them.
The shared module exposes separate strict Bellman/forward bounds and the
production SMM's retained nearest-grid mapping, so this consolidation does not
silently change either numerical convention.

The baseline and income-driven-repayment forward simulation are now converted
to the same sixteen-type interface. `simulation_fit_conterfactual_em.py` loads
the full posterior from `auxiliary_em_results.npz`, solves every type in
`TYPE_IDS`, and draws one persistent joint type per individual.
`model_simulation_em.py` uses that joint type's grant, transfer, and loan
components. Grants and transfers are realized from the auxiliary-EM hurdle
models, wages use the structural wage equations, and the resulting pre-choice
resources enter the canonical education-cell budget-shock realization. The
separate no-debt solver and simulation remain on the legacy interface and are
not run by this driver.

`latent_types.py` owns the ordering of the sixteen permanent joint types
`(school, grant, transfer, loan)`. Loan type is the fastest-moving component,
so adjacent type IDs differ only in loan type. The authoritative posterior is
`q` in `Model/Estimates/auxiliary_em_results.npz`; rows are validated to sum to
one.

## Production NPL driver

The production entry point is `estimation_all_em.py`. Its current top-level
configuration is:

```text
solve_model        = False
solve_continuation = False
solve_qs           = False
solve_initial_ccps = False
get_budget         = True
NPL iterations     = 30
```

Thus the current run reuses the saved continuation objects, auxiliary EM
posterior, and initial auxiliary CCPs. It does not re-estimate the auxiliary EM.
It verifies that the complete initial-CCP grid exists before beginning.

Within each NPL iteration it:

1. Builds type- and state-specific CCP continuation sequences with
   `model_getccp_sequence.get_ccp_sequence`.
2. Estimates the joint education-cell budget-shock process with
   `model_fitloans_dynamic.estimate_budget_shock_all_education`.
3. Reloads the newly saved budget parameters.
4. Solves the Bellman model for all invariant states and sixteen joint types.
5. Prepares observed value arrays and updates the flow-utility parameters.
6. Uses the new structural CCPs in the next NPL iteration.

Production uses `ccp_cache_mode="off"`: CCPs change after each NPL update, so
the continuation sequences must be reconstructed. The reusable CCP cache is
only for repeated standalone budget-SMM tests with unchanged CCPs.

Both Bellman calls in `estimation_all_em.py` now pass `maxdebt=True`. This is
the intended estimation model and activates annual/lifetime loan limits plus
the guarded maximization routine. Do not revert it to `False` casually. A
previous run with `False` reached `argmax` on an empty candidate set after
introducing estimated budget shocks.

The active guarded maximizer treats `c >= 2000` as the feasible consumption
set, matching the production budget SMM. It retains the existing local-window
search over feasible debt points. If no legally admissible debt choice reaches
the consumption floor, it forces the maximum admissible next-period debt and
evaluates that fallback at the consumption floor so the Bellman value remains
finite.

## Active production budget SMM

`estimate_budget_shock_all_education` estimates ten grouped cells jointly:
two-year years 1, 2, 3+; four-year years 1, 2, 3, 4, 5+; and graduate years 1,
2+. These supports are centralized in `budget_shock.py` and printed at startup.

The raw parameter vector therefore has 68 entries: six for each of ten cells,
plus eight shared preference parameters. Each cell has:

1. four shock-mean levels, one for each parental-income group;
2. one budget-shock sigma;
3. one common-within-cell slope of the shock mean on pre-choice resources,
   measured as dollars of shock per $10,000 of resources.

The final eight parameters are shared across all education cells:

- four parental-income-specific risk-aversion levels;
- four parental-income-specific debt-penalty levels.

The saved schema converts level parameters into its baseline-plus-deviations
representation where needed. Risk aversion and debt penalties are estimated,
but are restricted to be constant across program years and education sectors.
Shock means, sigma, and the resource slope may differ by education cell.

Each of the four parental-income debt parameters is a per-period flow-utility
penalty shared across all cells. In the budget-SMM debt-choice shortcut, a
positive candidate debt receives
`debt_penalty * (1 - beta**(T-period)) / (1-beta)`. With explicit model periods
1 through 9 and terminal period 10, period 1 receives nine discounted flow
penalties and period 9 receives one. The terminal continuation is unchanged.
The full Bellman recursion applies the single flow penalty once in each
explicit period: education choices use candidate `b1 > 0`, while a non-school
path uses its current persistent debt. It must not apply the multiplier
internally.

The production specification is currently homogeneous across the two latent
loan types. There is no loan-type mean shift, sigma ratio, or loan-type-specific
risk aversion in the active production vector. Earlier experimental code for
loan-type heterogeneity remains available but is not used by
`estimation_all_em.py`.

## Simulation inside the objective

- One complete joint type is drawn for each unique individual from that
  individual's sixteen-column EM posterior. The draw is persistent across all
  of the individual's periods and education cells. The current production SMM
  does not integrate exactly over all sixteen types.
- The sampled joint type selects internally consistent grant and transfer
  components. Loan type is retained even though the active budget parameters
  are currently homogeneous across loan types.
- Production uses `resource_mode="simulated"`: wages, grants, and parental
  transfers are simulated from their estimated processes. The residual budget
  shock is additional uncertainty and does not replace those shocks.
- `resource_mode="observed"` remains an available testing alternative.
- All wage, grant, transfer, type and budget standard draws are prepared with a
  fixed seed and reused at every objective evaluation (common random numbers).
- Production currently uses 100 simulation draws and does not compute
  Monte-Carlo standard errors.

## Active moments and loss

Production currently uses `moment_spec="flow_plus_stock"`. New annual borrowing is
defined consistently in data and simulation as
`b_next - (1 + interest_rate) * b_current`. For every education cell and
parental-income group it targets:

1. mean positive new-loan flow, conditional on receiving a new loan;
2. share receiving a positive new-loan flow;
3. standard deviation of positive new-loan flow;
4. 80th percentile of positive new-loan flow;
5. mean positive end-of-period debt stock;
6. share with positive end-of-period debt stock.

Each error is divided by the absolute data moment, with a small numerical
floor. Flow mean and receipt share receive weight 4 each; flow standard
deviation and p80 receive weight 1 each; stock mean and indebtedness share
receive weight 2 each. Parental-income groups themselves receive equal weight,
regardless of their sample sizes. The four-moment `fast_flow`, stock-based
`fast_stock`, and mixed `flow_stock` specifications remain available for
comparison, but are not the active production moments.

Across education cells, the cell loss is weighted by enrolled observations:
`N_cell / mean(N_cell)`. This gives more influence to education-year groups
with more data without changing the equal treatment of parental-income groups
inside a cell. Every printed fit block reports the cell's N, normalized weight,
raw loss, and weighted contribution.

The current objective therefore prioritizes the average positive new loan and
the new-loan receipt share, while using stock mean and participation to
discipline accumulated borrowing and persistence. Interest accumulation on
existing debt is removed before determining whether an individual receives a
new loan.

Model fit is printed every ten objective evaluations for every education cell
and parental-income group. Preserve these data/simulation comparisons when
changing the objective.

## Optimizer and parallelization

The production optimizer is `hybrid`:

1. SciPy dual annealing explores the bounded parameter space with
   `maxfun=500` and `no_local_search=True`.
2. Nelder-Mead starts from the best annealing vector and locally refines it with
   `maxiter=1000`.

If a compatible canonical estimate already exists, `restart=True` uses its raw
`budgetshock_bestx.npy` vector as the starting point. The annealing and local
evaluation counts are retained on the result object.

Current parallel settings in `estimation_all_em.py` are:

```text
CCP sequence tasks             60 processes
CCP-path loading               60 workers
SMM education-cell workers     automatic: one persistent process per cell
Numba threads per cell worker  1
Bellman solution               60 processes
```

On Linux, the SMM creates a persistent `fork` pool after all education-cell
data and common random numbers have been prepared. Workers retain those large
objects in memory and receive only the current cell parameter block plus shared
parameters on each objective call. Cells are evaluated simultaneously and the
parent process adds their losses and prints fit. On systems without `fork`, the
cell evaluation falls back to serial execution.

This design parallelizes across the ten grouped education cells, so CPU use is
bounded by ten workers when each worker has one Numba thread. Increase inner Numba
threads only after checking for oversubscription and benchmarking on the
server.

## Saved outputs and loading path

Production writes the canonical files:

```text
Model/Estimates/budgetshock_bestx.npy
Model/Estimates/budgetshock_params.npy
Model/Estimates/risk_aversion.npy
```

`budgetshock_params.npy` is the schema-version-5 named bundle consumed by the
solution and simulation. It includes grouped education-cell codes, grouping
and debt-penalty timing conventions, mean blocks, sigmas, resource slopes,
shared preference parameters, the estimation parameterization, and SMM
metadata. Named one-cell test estimates use longer filename prefixes and do
not overwrite the canonical production bundle.

After SMM, the parent calls `model_solution_em.reload_budgetshock_params()`.
Each Bellman worker also calls it through its multiprocessing initializer, so
workers load the newly estimated canonical bundle rather than stale globals.
The vector-to-bundle ordering and education-cell mapping were checked with a
synthetic round-trip test on 2026-07-17.

## Reproducible server workflow

From local PowerShell in the project root, push only the current model code and
update the server clone with the existing helper:

```powershell
python Code\2026_07_2\0_server\puschodes.py
```

The filename is intentionally recorded with its existing spelling:
`puschodes.py`.

To run manually from the server terminal:

```bash
python3 /home/ubuntu/work/studentloans/Code/2026_07_2/2_model/estimation_all_em.py
```

Server model inputs and estimates live under `/home/ubuntu/work/Model`, outside
the Git clone. Code paths are resolved through `config.py`; estimation programs
must not change the working directory.

## Files future agents should read first

1. `README_MODEL_ESTIMATION.md` (this file).
2. `estimation_all_em.py` for the active switches and production constants.
3. `budget_shock.py` for the canonical schema and structural mappings.
4. The education-cell section of `model_fitloans_dynamic.py`.
5. `model_solution_em.py` around `load_params_frombudget`,
   `get_expected_conditional`, and the two debt maximizers.
6. `latent_types.py` for the sixteen-type ordering.
7. `Code/2026_07_2/0_server/README.md` for push, server, tmux and result-sync
   workflows.

## Open questions; do not resolve without confirmation

- Whether later specifications should add more direct debt-persistence moments
  beyond the active stock mean and indebtedness share.
- Whether the present grouped education-year support should be revised after
  the joint estimates are evaluated.
- Whether and how latent loan type should enter the structural budget process
  or borrowing preferences.
- Whether later specifications should add persistence, parental-income/ability
  interactions, loan dispersion, or other moments.
- Whether production should eventually use observed rather than simulated
  wages, grants, and transfers.
- Whether the resource slope and shock sigma require tighter economically
  motivated bounds after inspecting the joint multi-cell estimates.

When experimenting, preserve the current version as an option, change one
parameterization or loss feature at a time, retain common random numbers, and
keep the printed model-fit tables.
