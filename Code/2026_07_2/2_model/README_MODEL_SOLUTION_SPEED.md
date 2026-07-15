# Potential speed gains in the structural model solution

This note focuses on reducing the cost of `model_solution_em.py` and the
Bellman-solution stage called by `estimation_all_em.py`. It assumes that all
sixteen permanent S x G x T x L types are economically distinct. In
particular, loan type L will shift the budget-shock distribution, so the model
must retain sixteen type-conditional value functions.

Period 10 is intentionally not a focus of this note. The priorities below are
the repeated nonterminal calculations over states, choices, shocks, and debt.

## 1. First complete the loan-type budget specification

The data and auxiliary posterior already retain the loan component, but the
saved budget-shock specification and backward quadrature do not yet condition
on it. At present:

- `model_fitloans_dynamic.py` samples and stores `loan_type`;
- `budget_shock.conditional_mean`, `conditional_sigma`, `realization`, and
  `quadrature` do not accept a loan-type argument;
- `model_solution_em.get_quadrature_budget` calls the common quadrature without
  passing the joint type's loan component.

The intended interface should make loan type explicit in both estimation and
solution:

```python
conditional_mean(spec, x1, period, loan_type)
conditional_sigma(spec, period, loan_type)
realization(spec, x1, period, standard_draws, loan_type)
quadrature(spec, x1, period, loan_type, degree=5)
```

`get_all_evt` should map the joint type once into `(school_type, grant_type,
transfer_type, loan_type)`. The selected `loan_type` should then be passed as a
small integer through the Bellman call chain. It should not be repeatedly
looked up inside state or choice loops.

The forward realization and backward quadrature must use exactly the same
loan-type parameterization. Useful regression tests are:

1. When the L1 shifts are set to zero, L0/L1 pairs with the same S, G, and T
   components must produce identical quadrature nodes and value functions.
2. With a nonzero loan-type mean shift, the two types must have different
   nodes but the same standardized Gauss-Hermite weights.
3. With a nonzero loan-type scale shift, both nodes and the implied shock
   variance must differ as specified.
4. Fixed standard draws passed through `realization` must reproduce the same
   conditional distribution used by `quadrature`.

Once this change is made, collapsing sixteen Bellman problems into eight is
not valid and is not proposed here.

## 2. Measure the model solution separately from surrounding I/O

Before changing the solver, record timings for one representative
`(type_id, invariant_state)` task and for one complete NPL iteration. At a
minimum separate:

- loading CCPs and other inputs;
- generating or loading period state arrays;
- evaluating noneducation choices;
- evaluating education choices;
- budget and wage quadrature;
- debt maximization;
- continuation-value lookup;
- construction of CCP/EVT arrays;
- writing VJT, EVT, and CCP artifacts.

The existing period-level timer is useful but too aggregated to decide whether
the next optimization should target arithmetic, Python dispatch, or storage.
The benchmark should use a warm process after Numba compilation, because
first-call compilation time is not representative of a long estimation.

All optimization claims should be evaluated against a fixed benchmark task and
checked numerically against the current solver. For exact refactors, CCPs and
value functions should agree to a tight tolerance.

## 3. Build the dynamic-state graph once

The current solution repeatedly reconstructs objects that depend only on the
state grid and fixed auxiliary parameters. Examples include:

- feasible choices `Jx` for each dynamic state;
- expanded state covariates from `get_x2_new`;
- successor states from `move_state_grad`;
- graduation and no-graduation successor states;
- graduation probabilities;
- choice-specific wage-parameter selection;
- tuition and education-stage indicators;
- the mapping from state arrays to continuation-value dictionary keys.

These objects should be compiled once into an integer-indexed state graph for
each period. A possible representation is:

```text
period_state_offsets
choices[state_start:state_end]
next_state_no_grad[state_choice]
next_state_grad[state_choice]
prob_grad[invariant_state, state_choice]
choice_is_education[state_choice]
choice_wage_equation[state_choice]
```

The Bellman kernel can then retrieve next-period values by integer indexing.
It will no longer need to create arrays describing choices, copy state vectors,
or construct long string dictionary keys for every type and NPL iteration.

This refactor preserves the complete state space and all sixteen type-specific
solutions. It changes only how transitions are represented. It is likely the
most important structural improvement because it removes Python work from
every state-choice evaluation and creates the array layout needed for further
Numba optimization.

## 4. Cache quadrature correctly, including loan type

`scipy.special.roots_hermite` currently regenerates standard nodes and weights
inside the solution. The standard degree-five nodes and weights are constant
and should be created once.

Conditional wage quadrature can then be assembled from cached standard nodes
using the choice-specific wage variance. Budget-shock quadrature should be
cached with a key containing every determinant of its distribution, at least:

```text
(period, invariant_state_index, loan_type, quadrature_degree)
```

If the final budget specification adds education-sector or dynamic-state
effects, those indices must also enter the cache key. A cache that omits
`loan_type` would silently give the wrong value functions after the planned
extension.

The joint wage-budget nodes and weights for education choices can also be
precomputed. This avoids repeated calls to `tile`, `repeat`, and `kron` in
`get_expected_conditional`.

## 5. Precompute debt transitions for noneducation choices

For work and home choices, the current code calculates tomorrow's continuous
debt and then constructs a full squared-distance matrix against the debt grid
to find the closest grid point. With five wage nodes and 100 debt points, this
mapping is recomputed many times even though it is fixed for a given:

```text
(period, invariant state, dynamic state, choice, wage node, current debt)
```

There are two exact improvements:

1. Replace the full distance matrix with a nearest-neighbor lookup based on
   `np.searchsorted`, exploiting the sorted debt grid.
2. Preferably, compute the resulting debt-index transition array once and
   store it in the state graph.

The Bellman iteration would then gather next-period EVT values using an integer
transition array. This removes an operation that is currently quadratic in the
number of debt-grid points from the repeated solution path.

Education-choice borrowing bounds are also functions of the current debt,
education stage, and debt grid. The lower and upper feasible indices should be
precomputed by state and education choice instead of calling
`get_debt_region_bounds` for every type and iteration.

## 6. Restrict the solution to reachable current-debt states

There are two distinct debt restrictions, and the solver should exploit both:

1. conditional feasibility of next-period debt `b[t+1]` given current debt
   `b[t]`; and
2. reachability of current debt `b[t]` given the complete history that can lead
   to the current dynamic state.

The first restriction is partly represented by `get_debt_region_bounds`, which
calculates borrowing intervals using accrued debt and annual and lifetime
limits. However, the main estimation path currently sets `maxdebt=False`, so
`get_maximum` calls `get_maximum_loop_modified_c` rather than
`get_maximum_loop_modified_c_maxdebt`. The active routine narrows the search
using positive consumption and a window around the previous argmax, but it does
not fully use the explicit statutory borrowing intervals. Before optimizing
this area, verify whether `maxdebt=False` is economically intended for the
baseline solution. If the loan estimator applies annual and lifetime limits
while the structural solver does not, this is an estimator-solver consistency
issue as well as a speed issue.

The second restriction is not currently exploited. The solver attaches the
complete debt grid to every dynamic state, although many `(x2[t], b[t])`
combinations cannot arise. Examples include high accumulated debt in early
periods, debt above what could have been borrowed given prior enrollment, and
post-school debt points that cannot be generated by any preceding borrowing
and repayment path.

An exact feasible-policy reachability graph can be constructed forward from
the initial state and initial debt. For every reachable `(x2[t], b[t])`, mark
all successors produced by:

- every feasible choice;
- every possible graduation outcome;
- every relevant repayment shock/quadrature outcome; and
- every permitted next-debt choice for education alternatives.

The result can be stored as either boolean masks or compact debt-index lists:

```python
reachable_debt_indices[period][state_index]
```

The backward solver would calculate values only for those current-debt indices.
For each one, education maximization would consider only its precomputed
feasible next-debt interval. This reduces both expensive dimensions of the
problem: impossible current debts and impossible future debts.

The reachability graph must include debt states attainable under **any feasible
policy**, not only debt states selected by the current optimal policy. A mask
based on optimal choices would change with utility parameters and CCPs and
could incorrectly remove alternatives during estimation. A feasible-policy
mask is fixed and can be reused across NPL iterations.

Loan type will change the payoff and therefore the optimal debt choice, but it
does not necessarily change which debt choices are legally feasible. If loan
type affects only the distribution of the additive budget shock, the structural
reachability mask can be shared across all sixteen types. If it changes
borrowing eligibility, limits, or a hard consumption-feasibility condition,
the relevant type component and shock state must enter the reachability
definition.

Reachability should be validated by comparing the restricted and unrestricted
solvers on small state grids. Every retained value and CCP must agree, and a
forward simulation from valid initial conditions must never request a removed
state-debt combination.

## 7. Remove large temporary arrays from education maximization

For an education choice, `get_utility` constructs consumption for the Cartesian
product of:

- wage nodes;
- budget-shock nodes;
- current debt points;
- next-debt choices.

It does so using several `repeat`, `tile`, and broadcasting operations and
materializes the full consumption matrix. The debt maximization then scans
only feasible portions of that matrix.

A faster Numba kernel can compute consumption and utility only for candidate
next-debt indices. The outer loops should be over shock nodes and current debt;
the inner loop should scan the precomputed feasible debt interval. This avoids
materializing values that are immediately discarded.

The exact version should search every feasible next-debt point. The current
local-window search around the previous argmax can remain as an optional fast
method, but it should be tested against a full-search reference. Monotonicity of
the optimal debt choice may allow a safe monotone search, but it should not be
assumed without exhaustive comparison across states, types, and trial
parameters.

The preferred kernel interface is roughly:

```text
education_value(
    next_evt,
    deterministic_cash,
    wage_nodes,
    budget_nodes_for_loan_type,
    joint_weights,
    feasible_debt_lo,
    feasible_debt_hi,
    debt_grid,
    risk_aversion,
    debt_penalty,
)
```

This makes loan-type dependence explicit while keeping all other inputs
contiguous and numerical.

## 8. Move the complete period calculation into compiled kernels

Several individual helpers are Numba compiled, but the outer loops over states
and choices remain Python loops. That limits the benefit of Numba and creates
many small array allocations and function calls.

After constructing the integer state graph, the nonterminal calculation should
be organized into one or a small number of compiled kernels. A practical split
is:

- one kernel for noneducation choices;
- one kernel for education choices and debt maximization;
- one kernel for the log-sum-exp/CCP update.

The kernels should receive arrays rather than dictionaries or Python lists.
Parallelism inside Numba should only be introduced after benchmarking against
the existing process-level parallelism. Running many multiprocessing workers
that each start many Numba threads will oversubscribe the machine.

Stable `logsumexp` calculations should replace direct expressions such as
`log(sum(exp(value)))`. This is primarily a numerical-correctness improvement,
but it also allows a single fused pass over choice values.

## 9. Separate solution output needed for recursion, likelihood, and storage

The current solver writes complete compressed VJT and EVT bundles for every
period, invariant state, and type. These objects have different purposes:

- next-period EVT is needed immediately for backward recursion;
- full-grid CCPs are needed by the next NPL iteration;
- the structural likelihood needs VJTs only at observed state/debt rows;
- complete EVT/VJT archives may be useful as checkpoints or for later
  simulation, but are not necessarily needed during every estimation
  iteration.

During estimation, the solver can keep EVT in memory while moving backward and
write only:

1. the full CCP grid required for the next NPL iteration; and
2. a compact likelihood-ready VJT array selected at the observed state/debt
   indices.

This would eliminate the subsequent pattern in `prepare_vjt_feasible` that
opens a compressed VJT archive once for each observed individual. If retaining
the existing file interface initially, group observations by invariant state,
open each archive once, and select all relevant dynamic states and debt rows in
one pass.

Temporary estimation artifacts should preferably be uncompressed and stored
on local scratch storage. Compression and Dropbox synchronization can be
reserved for final checkpoints. Storage changes must be benchmarked separately
from arithmetic improvements.

## 10. Tune multiprocessing instead of fixing the pool at 60 workers

The natural coarse task is one `(joint type, invariant state)` Bellman problem.
With sixteen types and 32 invariant states, there are 512 tasks per model
solution, which is enough to distribute across a large server.

The best worker count is not automatically 60. Performance depends on physical
cores, memory bandwidth, compression, filesystem throughput, and Numba/BLAS
thread settings. Benchmark several worker counts and record both elapsed time
and peak memory. Worker initializers should:

- load common parameters once;
- load or attach the immutable state graph once;
- initialize the type-specific financial and quadrature caches;
- restrict BLAS and Numba inner threads when process-level parallelism is used.

The pool can remain alive across NPL iterations if workers can receive updated
utility parameters and CCP inputs without reloading all invariant objects.
Avoid returning large arrays through multiprocessing pipes; write to assigned
shared-memory or scratch locations and return only small status records.

## 11. Reduce the number of full model solutions

The driver currently requests 30 NPL iterations. Even a highly optimized
Bellman solver will be expensive if the fixed-point loop performs unnecessary
solutions.

Exact-target improvements include:

- stop when both the structural parameters and CCP mapping satisfy explicit
  convergence tolerances;
- report the maximum and weighted mean CCP changes after every iteration;
- use adaptive damping rather than a fixed parameter average;
- test Anderson acceleration of the NPL fixed-point mapping;
- avoid re-estimating the budget block in an iteration when its inputs have not
  changed enough to affect its optimum materially.

Acceleration changes the path to the fixed point, not the target fixed point,
provided the final convergence conditions are enforced. It should nevertheless
be guarded by objective and residual checks because accelerated NPL mappings
can become unstable.

## 12. Optional approximations, only after the exact refactor

The following methods can reduce time further but change intermediate numerical
accuracy and therefore require final full-resolution validation:

- use fewer wage or budget quadrature nodes in early NPL iterations;
- use a coarser debt grid early and restore the 100-point grid near convergence;
- interpolate continuation values over debt;
- solve only a transition-closed set of states relevant for the observed
  likelihood;
- use a reduced candidate set for next-period debt after verifying the omitted
  alternatives never bind in a full-search benchmark.

A useful coarse-to-fine protocol would run inexpensive early NPL updates and
then continue to convergence using the original debt grid and quadrature. Final
reported estimates and counterfactuals should always be produced with the full
solver unless a separate approximation error analysis supports otherwise.

GPU/JAX conversion is not a first priority. The current bottleneck contains
Python dictionaries, irregular feasible-choice sets, repeated file access, and
many allocations. Converting to integer-indexed contiguous arrays is necessary
for efficient CPU execution and is also the prerequisite for any later GPU
implementation.

## 13. Recommended implementation order

The proposed order emphasizes exact changes and creates a benchmark after each
step:

1. Add loan type to `budget_shock.py`, `model_fitloans_dynamic.py`, and the
   solver's backward quadrature; validate the distribution mapping.
2. Add warm-run profiling for a representative Bellman task and a complete
   model solution.
3. Compile the integer-indexed state/choice/successor graph.
4. Construct feasible-policy reachable debt sets and validate them against the
   unrestricted solver.
5. Cache standard and conditional quadrature using keys that include loan type.
6. Precompute noneducation debt transitions and education borrowing bounds.
7. Replace education-choice temporary matrices with a fused full-search Numba
   kernel and compare it with the current implementation.
8. Move the remaining period state-choice loops into compiled array kernels.
9. Produce likelihood-ready VJTs directly and reduce compressed intermediate
   output.
10. Benchmark process counts, persistent workers, thread limits, and local
   scratch storage.
11. Add NPL convergence stopping and evaluate acceleration.
12. Only then test coarse-to-fine or other approximate methods.

The most promising first model-solution gains are the state graph, reachable
current-debt sets, cached debt transitions, loan-type-aware quadrature caches,
and a fused education debt-maximization kernel. They retain all sixteen state
spaces and the intended economic specification while removing repeated work
inside each one.
