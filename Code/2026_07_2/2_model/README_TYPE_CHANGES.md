# Temporary tracker: extending the structural model to eight latent types

Status: implementation in progress. The shared type layout, typed auxiliary
financial-process interface, and eight-type auxiliary CCP predictor are implemented.
The Bellman solver and structural likelihood still use their old interfaces.

This file tracks the changes required to extend the complete model from the old
two-schooling-type interface to the eight joint types already estimated by the
auxiliary EM algorithm. Update the checkboxes as implementation proceeds. This
is a temporary working document and can be merged into
`README_MODEL_ESTIMATION.md` after the conversion is complete.

## Target economic specification

The permanent joint type is part of the state:

```
V_t(x1, x2_t, b_t, type_id)
```

The public type identifier is one-based (`1, ..., 8`) to preserve the current
`em_type` convention and output filenames. A shared mapping translates it into
three binary components:

| `type_id` | School type | Grant type | Transfer type | Name |
|---:|---:|---:|---:|:---|
| 1 | 0 | 0 | 0 | `S0G0T0` |
| 2 | 0 | 0 | 1 | `S0G0T1` |
| 3 | 0 | 1 | 0 | `S0G1T0` |
| 4 | 0 | 1 | 1 | `S0G1T1` |
| 5 | 1 | 0 | 0 | `S1G0T0` |
| 6 | 1 | 0 | 1 | `S1G0T1` |
| 7 | 1 | 1 | 0 | `S1G1T0` |
| 8 | 1 | 1 | 1 | `S1G1T1` |

The components have separate roles:

- school type selects the schooling-preference effect in flow utility;
- grant type selects the grant receipt and positive-amount equations;
- transfer type selects the parental-transfer receipt and positive-amount equations;
- the joint type is permanent: `type_id(t + 1) = type_id(t)`;
- posterior probabilities do not enter the Bellman equation. Bellman problems
  are solved conditional on type; posterior weights enter structural
  likelihoods and simulation draws.

The model must solve eight conditional dynamic programs instead of two. This
adds eight copies of the Bellman solution for each invariant state while leaving
the endogenous `x2` grid unchanged. Relative to the present two-type solution,
this portion is expected to be approximately four times as expensive.

## Implementation conventions

- [x] Add `latent_types.py` as the single source of truth.
- [x] Define `TYPE_IDS`, `TYPE_COMPONENTS`, `TYPE_NAMES`, and `N_TYPES` there.
- [x] Add `type_components(type_id)` with bounds validation.
- [x] Keep `type_id` one-based at public interfaces and in filenames.
- [ ] Use `type_index = type_id - 1` only for NumPy/posterior/list indexing.
- [ ] Do not append type to the existing `x2` arrays. Pass it separately as a
      permanent state index, as the code currently passes `em_type`.
- [ ] Replace hard-coded `range(1, 3)` loops with `TYPE_IDS`, not `range(1, 9)`.
- [ ] Replace paired names such as `type1/type2` with type-indexed collections.
- [ ] Load posterior `q` and the type layout from the same full EM results file
      and validate that their dimensions and ordering agree.
- [ ] Keep collapsed two-column posterior files only for deliberate reporting
      or backward compatibility, never as structural-model inputs.

## 1. Shared latent-type definition (new file)

### `latent_types.py`

- [x] Create the one-based joint type interface and binary component mapping.
- [x] Add helpers to validate a type ID and map it to school/grant/transfer type.
- [x] Add a helper to validate `q`: two-dimensional, `N_TYPES` columns, finite,
      nonnegative, and rows summing to one.
- [ ] Import this module everywhere below instead of recreating mappings.
- [x] Add validation ensuring its ordering exactly matches `auxiliary_em_results.npz`.

## 2. Auxiliary EM and structural likelihood

### `model_em_algorithm.py`

The new auxiliary block already estimates eight types. Older structural
functions in the same file still assume two.

- [x] Import the shared type layout and remove the local duplicate definitions
      (`TYPE_NAMES`, `TYPE_SCHOOL`, `TYPE_GRANT`, `TYPE_TRANSFER`,
      `N_AUXILIARY_TYPES`) or alias them from the shared module.
- [ ] Preserve the current eight-type auxiliary likelihood and EM M-steps.
- [ ] In `prepare_vjt_feasible`, replace the two-type loop with `TYPE_IDS`.
- [ ] Continue saving prepared VJT arrays with `_em{type_id}` filenames.
- [ ] In `load_all_arrays_feasible`, replace `vjt_all_type1` and
      `vjt_all_type2` with one `vjt_all_types` collection.
- [ ] Change `likelihood` to accept `vjt_all_types` and infer/check the number
      of types from the shared layout and `q`.
- [ ] Loop over every joint `type_id` in the structural likelihood.
- [ ] Weight the conditional contribution for type `k` by `q[:, k]`.
- [ ] In the structural `get_all_g`, map joint type to school type instead of
      interpreting joint IDs 1 and 2 as the two schooling types.
- [ ] In `jacobian_likelihood_numba` / `jacobian_likelihood`, use the mapped
      school type to activate the two schooling-type coefficients.
- [ ] Generalize `temp_jacobian`, `notlogs_likelihood`, and any retained
      structural diagnostic likelihoods that are still used.
- [ ] Stop treating `em_q_typeff2.npy` as the active structural posterior.
- [ ] Use `auxiliary_em_results.npz` (preferred) or
      `auxiliary_q_eight_types.npy` for the full posterior.
- [ ] Keep saving the full type layout in EM checkpoint/result files.
- [ ] Validate saved/checkpoint type ordering against `latent_types.py`.

Relevant current markers to search:

```
range(1,3)
range(1, 3)
vjt_all_type1
vjt_all_type2
number_types=2
em_type == 1
em_type == 2
q[:,em_type-1]
em_q_typeff2.npy
```

## 3. Shared financial-resource processes

### `financial_process.py`

The structural solver and forward simulation currently load old,
type-independent financial equations. Both must use the type-specific equations
saved by the auxiliary EM.

- [x] Add a loader for financial parameters in `auxiliary_em_results.npz`.
- [x] Load grant receipt, positive-amount, and residual-sigma parameters for
      two-year, four-year, and graduate enrollment.
- [x] Load transfer receipt, positive-amount, and residual-sigma parameters.
- [x] Extend `expected_grant_scalar` with a `grant_type` argument.
- [x] Extend `expected_grants_vectorized` with individual grant types.
- [x] Add `expected_transfer_scalar` with a `transfer_type` argument.
- [x] Add `expected_transfers_vectorized` with individual transfer types.
- [x] Apply the same lognormal mean correction used by the auxiliary EM.
- [x] Preserve zero grant/transfer amounts for ineligible alternatives.
- [x] Validate coefficient shapes and education-level ordering on load.
- [ ] Use this one module in both backward solution and forward simulation.
- [ ] Do not independently duplicate the financial formulas in either consumer.

## 4. Structural Bellman solution

### `model_solution_em.py`

- [ ] Import the shared latent-type helpers.
- [ ] Load the EM-estimated type-specific financial process.
- [ ] Treat the existing `em_type` argument as joint `type_id` (renaming can be
      postponed to minimize the initial diff).
- [ ] Pass joint type through the complete Bellman call chain:
  - [ ] `get_all_evt`
  - [ ] `loop_over_states`
  - [ ] `loop_rows`
  - [ ] `get_all_choices`
  - [ ] `get_expected_conditional`
  - [ ] `get_conditional`
  - [ ] `get_debt_income`
  - [ ] `get_debt_income_home`
  - [ ] `fin_help`
- [ ] In `fin_help`, map joint type to grant and transfer types and select the
      corresponding expected-resource equations.
- [ ] In `load_param_g` / `build_param_g`, map joint type to school type.
- [ ] Set the schooling-type utility block to zero when `school_type == 0`, not
      only when `em_type == 1`.
- [ ] Keep the two existing schooling-type utility coefficients; the eight
      joint types do not require eight separate schooling-effect vectors.
- [ ] Verify that terminal values remain type-independent under the maintained
      specification. If type affects only schooling-period utility/resources,
      the terminal interpolation does not need eight versions.
- [ ] Verify every VJT/EVT/CCP save and load path uses the joint type ID.

The existing `_em{em_type}` filename convention can represent `_em1` through
`_em8`; a new filename schema is not required.

## 5. Main structural-estimation driver

### `estimation_all_em.py`

- [ ] Import `TYPE_IDS` and `N_TYPES`.
- [ ] Replace every solution loop over `range(1, 3)` with `TYPE_IDS`.
- [ ] Load the complete eight-column posterior `q`.
- [ ] Replace `utility_parameters1` and `utility_parameters2` with a
      type-indexed `utility_parameters` collection.
- [ ] Create multiprocessing arguments for all eight conditional solutions.
- [ ] Do this in the initial solution, every NPL iteration, and the optional
      budget-shock/CCP preparation paths.
- [ ] Receive `vjt_all_types` from `load_all_arrays_feasible`.
- [ ] Pass `vjt_all_types` and the full `q` to the structural likelihood.
- [ ] Validate that all eight VJT sets exist before likelihood evaluation.

## 6. Initial CCP construction

### `model_predict_ccps.py`

- [x] Import the shared type layout.
- [x] Make `load_utility_parameters(type_id)` map joint type to school type.
- [x] Replace `utility_parameters1` / `utility_parameters2` with a type-indexed
      collection.
- [x] Build `get_all_ccps` tasks for every joint type.
- [x] Save CCP files as `_em1` through `_em8`.
- [x] Load the full auxiliary `choice_parameters` rather than truncating
      `param_em_latest.npy` to the legacy utility block.
- [x] Include the auxiliary expected-consumption coefficient using the same
      expected wage, grant, transfer, tuition, eligibility, and home normalization.
- [x] Include the auxiliary current-debt-versus-home coefficient at every debt
      grid point.
- [x] Compute the home-production CCP with a stable log-sum-exp calculation.
- [ ] Check that joint types sharing a school component share direct schooling
      utility effects but may have different CCPs through resource-dependent
      continuation values.

## 7. CCP continuation sequences

### `model_getccp_sequence.py`

The internal interface already accepts `em_type` and is mostly generic.

- [ ] Validate the incoming joint type using `latent_types.py`.
- [ ] Ensure callers request sequences for every `type_id`.
- [ ] Ensure it reads and writes `_em1` through `_em8` files.
- [ ] Replace any documentation saying the input has only two education types.

## 8. Active loan/budget-shock estimation

### `model_fitloans_dynamic.py`

- [ ] Load the complete eight-column posterior, not `em_q_typeff2.npy`.
- [ ] Replace `ccp_path_type1` / `ccp_path_type2` with a list or array of paths
      for all `TYPE_IDS`.
- [ ] Replace the binary schooling-type draw with one categorical joint-type
      draw from each individual's `q[i, :]`.
- [ ] Select each sampled individual's CCP path using `type_id - 1`.
- [ ] Use grant and transfer resources selected by the same joint type.
- [ ] Preserve fixed/common random draws within an optimization.
- [ ] Remove comments and conditions interpreting type as only type 1/type 2.
- [ ] Validate that the solution, CCP sequence, and observed individual all use
      the same type ordering.

## 9. Forward structural simulation

### `model_simulation_em.py`

- [ ] Load the EM-estimated type-specific financial process.
- [ ] Change `get_types` from a binomial draw using `q[:, 1]` to a categorical
      draw using all posterior columns.
- [ ] Return and save joint type IDs `1, ..., 8`.
- [ ] Reuse the saved baseline type draw in every counterfactual.
- [ ] Replace the two-element `uparams` structure with a joint-type-indexed
      collection or map joint type to its schooling-utility object.
- [ ] In `get_expected_conditional_x`, load VJT using the joint type and use the
      mapped school type for direct flow utility.
- [ ] Pass individual joint types into `get_conditional_agents` and the
      financial-resource calculation.
- [ ] Extend `fin_help_agents` to select grant and transfer equations by the
      individual component types.
- [ ] When separating education/noneducation agents, subset their type vector
      using exactly the same indices before passing it to conditional functions.
- [ ] Keep `move_types` synchronized with every state/choice reordering.
- [ ] Check all VJT/EVT loads accept type IDs 1 through 8.
- [ ] Keep a permanent type fixed across all periods.

## 10. Baseline and counterfactual simulation driver

### `simulation_fit_conterfactual_em.py`

- [ ] Import the shared type layout.
- [ ] Replace `utility_parameters1`, `utility_parameters2`, and two-element
      `uparams` with type-indexed collections.
- [ ] Generate baseline Bellman tasks for all eight types.
- [ ] Load the complete posterior `q`.
- [ ] Generate counterfactual Bellman tasks for all eight types.
- [ ] Apply the same change to every baseline, debt-policy, no-tuition, and
      no-debt block in this driver.
- [ ] Reuse the same saved type draw across policy regimes.
- [ ] Check that all required `_em1` through `_em8` artifacts exist before
      forward simulation starts.

## 11. No-debt solution and simulation

### `model_solution_nodebt.py`

- [ ] Mirror the joint-type mapping used by `model_solution_em.py`.
- [ ] Pass joint type through the no-debt Bellman call chain.
- [ ] Select school utility, grants, and transfers by their mapped components.
- [ ] Replace `em_type == 1` schooling logic in `build_param_g`.
- [ ] Solve and save no-debt values for all eight joint types.

### `model_simulation_nodebt.py`

- [ ] Use the same stored joint type draw as the baseline simulation.
- [ ] Select joint-type-specific VJT/EVT files.
- [ ] Use the mapped school utility and financial-resource equations.
- [ ] Preserve type during all state reorderings.

## 12. Reporting and documentation

### `tables_types.py`

- [ ] Confirm it continues to read the full joint posterior and shared layout.
- [ ] Prefer importing/validating the shared type layout rather than maintaining
      a separate interpretation of the saved columns.

### `tables.py`

- [ ] Check type labels and prior-probability rows against the shared mapping.

### `figures.py`

- [ ] Replace binary constructions such as `type2 = type - 1` and
      `type1 = 1 - type2` where they refer to latent types.
- [ ] Decide plot by plot whether to report eight joint types or an explicit
      marginal collapse by school, grant, or transfer component.
- [ ] Label every deliberate collapse so it cannot be mistaken for the joint
      type distribution used in the structural model.

### `README_MODEL_ESTIMATION.md`

- [ ] Remove the statement that financial types are auxiliary-only.
- [ ] Document the joint type as a permanent structural state.
- [ ] Document the eight conditional Bellman solutions and full-posterior
      structural likelihood/simulation.
- [ ] Merge this temporary checklist into the main README when completed.

## 13. Files not expected to change under the current specification

These should remain type-independent unless the economic specification is
expanded again:

- `budget_shock.py`
- `model_interpolate_terminal.py`
- `model_continuation_final.py`
- `solve_many_continuations.py`
- `config.py`
- wage equations
- graduation equations
- endogenous `x2` state-grid construction

Re-audit this list if latent type is later allowed to affect wages, graduation,
the budget-shock distribution, debt preferences, or terminal outcomes.

## 14. Legacy and optional code

The following files contain two-type assumptions but are not part of the active
pipeline described in `README_MODEL_ESTIMATION.md`. Do not modify them during
the first pass unless they are reactivated:

- `model_fitloans_dynamic_fast.py`
- `model_fitloans_fast.py`
- `model_fitloans_ability.py`
- `model_fitloans_time.py`
- `model_fitloans_nosigma.py`
- `moldel_fitloans_uniperiod_debugger.py`
- `model_fitdebt_new2.py`
- `esimation_share.py`
- `simulation_fit_conterfactual_adjusted.py`
- files under `Old Codes/`
- files under `Unused/`

If any of these becomes active, audit at least the following patterns:

```
em_q_typeff2.npy
ccp_path_type1
ccp_path_type2
evt1
evt2
vjt_type1
vjt_type2
np.random.binomial
range(1, 3)
em_type == 1
em_type == 2
```

## 15. Validation checklist

### Type layout and posterior

- [ ] `TYPE_IDS` contains exactly `1, ..., N_TYPES`.
- [ ] The shared layout matches the EM result layout exactly.
- [ ] `q.shape == (n_individuals, N_TYPES)`.
- [ ] Every posterior row is finite, nonnegative, and sums to one.
- [ ] Categorical simulation draws always lie in `TYPE_IDS`.
- [ ] Empirical draw frequencies reproduce posterior probabilities in a large
      synthetic simulation.

### Economic mapping

- [ ] Types 1--4 have low schooling utility effects; types 5--8 have high
      schooling utility effects.
- [ ] Types 1, 2, 5, 6 use the low grant equations; types 3, 4, 7, 8 use the
      high grant equations.
- [ ] Odd types use the low transfer equations; even types use the high
      transfer equations.
- [ ] Types sharing a component receive identical direct effects from that
      component, holding all other inputs fixed.
- [ ] Type is unchanged across every simulated period and policy regime.

### Solution artifacts

- [ ] VJT, EVT, and CCP artifacts exist for every required period, invariant
      state, and type ID.
- [ ] The structural likelihood refuses to run when any type artifact is missing.
- [ ] The loan estimator refuses to run when any type CCP sequence is missing.
- [ ] Baseline and counterfactual simulations load the intended type-specific
      solution files.

### Regression and consistency tests

- [ ] With all grant/transfer type shifts set to zero, types sharing school type
      produce identical Bellman values.
- [ ] With all school, grant, and transfer shifts set to zero, all eight types
      produce identical Bellman values.
- [ ] A one-hot posterior assigns every individual to the expected joint type.
- [ ] A two-type compatibility fixture reproduces the previous structural
      likelihood and simulation logic within numerical tolerance.
- [ ] Scalar backward and vectorized forward financial-resource functions agree
      for the same individual, alternative, and joint type.
- [ ] Analytical structural gradients still agree with numerical gradients.

## 16. Completion definition

The conversion is complete only when:

- [ ] no active model file loads the collapsed two-column posterior;
- [ ] no active model loop hard-codes two latent types;
- [ ] all type-dependent utility and resource calculations use the shared mapping;
- [ ] the solver, CCP pipeline, likelihood, loan estimator, and simulations use
      the same joint type ordering;
- [ ] baseline and every reported counterfactual work with eight types;
- [ ] the validation checklist above passes;
- [ ] this tracker is reconciled with `README_MODEL_ESTIMATION.md`.
