# Differential verifier

The differential verifier is a developer tool for checking small PunyTorch
computation graphs against an independent NumPy evaluator. It exercises the
public `Tensor` API, checks every forward node, differentiates a serialized
scalar loss, and compares input gradients with central finite differences.
It is kept under `devtools/` and is not imported by normal PunyTorch code.

## Run it

The smoke profile is intended for normal local checks and CI:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-punytorch \
  uv run python -m devtools.differential_verifier run \
  --profile smoke --seed 0 --cases 12
```

The extended profile permits slightly larger shapes. A useful local sweep is:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-punytorch \
  uv run python -m devtools.differential_verifier run \
  --profile extended --seed 1000 --cases 240
```

`--seed` and `--cases` are required so a run is always explicit. Every case
derives its own case seed from the master seed and case index. A failure writes
a versioned JSON artifact under `.punytorch-differential-failures/` and prints
an exact replay command. Replay consumes the saved values and graph; it does
not regenerate that case or any preceding case:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-punytorch \
  uv run python -m devtools.differential_verifier replay \
  .punytorch-differential-failures/seed-7-case-0003-0123456789ab.json
```

Replay refuses a different Git revision or source fingerprint by default. The
fingerprint covers the PunyTorch runtime, verifier implementation, project
metadata, and lockfile, including uncommitted verifier work. This makes the
saved source state, scenario, tolerance policy, and operation order part of the
reproduction contract. `--allow-revision-mismatch` is available for deliberate
cross-source comparisons.

## What a scenario contains

The stable scenario schema records:

- schema version, scenario ID, master seed, case index, derived case seed,
  generation profile, full Git revision, and source fingerprint;
- input names, shapes, `float64` dtype, actual values, differentiability, and
  bounded/positive/nonzero domain constraints;
- an ordered graph of named operations, references, and parameters;
- the output and scalar loss reduction, including serialized weights for a
  weighted sum;
- forward, gradient, and finite-difference tolerances; and
- ordered metamorphic checks.

The format is deliberately small and reducer-ready. Generated inputs are
embedded in the failure artifact, so replay never depends on random-number
generator behavior.

## Verification classes

1. **Forward differential:** every named node is compared with the independent
   NumPy evaluator. The first bad node is therefore more informative than only
   comparing the final loss.
2. **Backward differential:** the real PunyTorch scalar loss calls
   `Tensor.backward()`. Each differentiable input is checked against
   second-order central finite differences of the NumPy reference loss.
3. **Metamorphic:** generated cases exercise reshape round trips, nested versus
   direct reductions, `logsumexp(x + c) == logsumexp(x) + c`, and a small
   analytic repeated-index accumulation oracle.
4. **Determinism:** tests assert identical scenario serialization, fingerprints,
   numerical results, and comparison ordering for identical inputs and
   revisions.
5. **Verifier self-tests:** test-only observation hooks deliberately corrupt a
   forward value or gradient and assert that the intended node/input fails.
   They do not edit or mutate production source files.

Outcomes are classified as `pass`, `mismatch`, `invalid_graph`, `unsupported`,
`numerical_skip`, `candidate_error`, or `generator_rejection`. Reference
validation runs before PunyTorch execution, so a malformed generated graph is
not reported as a framework defect. Non-finite values, non-positive logarithm
inputs, extreme exponential inputs, and near-zero divisors are numerical-domain
skips rather than comparisons with meaningless results.

## Initial operation slice

The generated slice is intentionally coherent rather than comprehensive:

- elementwise `add`, `sub`, `mul`, and `div`, including broadcasting;
- reductions `sum`, `mean`, and numerically stable `logsumexp`;
- `reshape` and two-dimension `transpose`;
- nonlinear `tanh`, `sigmoid`, `exp`, and `log`;
- `stack` and `cat`; and
- public indexing and `gather`, including repeated indices.

The generator constrains general values to modest magnitudes, divisors away
from zero, exponential inputs to a safe range, and logarithm inputs to positive
intermediate results.

The first slice deliberately excludes `mod` (ambiguous derivative), `abs`,
`relu`, and `max` (non-smooth points/tie policy), general `pow` (domain-sensitive
base/exponent combinations), `matmul` (better served by a dedicated shape
family), `softmax`, losses, dtype casts, and neural-network modules. These are
high-value extensions once each has an explicit domain and oracle policy.

## Numerical policy

Generated cases currently use `float64` only:

- forward: `atol=1e-10`, `rtol=1e-10`;
- gradient: `atol=3e-5`, `rtol=3e-5`; and
- central finite difference base step: `1e-6`, scaled by
  `max(1, abs(input_value))`.

Forward evaluation is expected to be nearly identical, while a looser gradient
tolerance accounts for subtractive cancellation across multi-operation scalar
losses. This policy is local to the verifier and does not change any existing
PunyTorch numerical test. Adding `float32` requires a separate step and
tolerance policy rather than silently reusing these values.

## Read a failure

A failure report includes the entire small operation graph, input metadata and
values, comparison class, target node or input, expected and observed arrays,
maximum absolute and relative error, tolerance, and first failing index. The
JSON artifact stores the same semantic comparison fields without relying on a
verbose text snapshot.

Example shape:

```text
FAIL scenario=seed-7-case-0003 outcome=mismatch
comparison=gradient target=input:x
expected=[[...]]
observed=[[...]]
max_absolute_error=0.125
max_relative_error=...
tolerance=atol:3e-05 rtol:3e-05
first_failing_index=(0, 0)
replay: cd ... && UV_CACHE_DIR=... uv run python -m devtools.differential_verifier replay ...
```

## Add an operation

1. Add the PunyTorch public call to the matching dispatch table in
   `candidate.py`, or use a small named handler when the operation has
   operation-specific parameters.
2. Add an independently written NumPy formula and numerical-domain policy in
   `reference.py`.
3. Add the operation to a shape-compatible family in `generator.py`.
4. Add a small known-good reference test, a real Tensor/autograd integration
   test, and—where useful—an equivalent-formulation or analytic property.
5. Run the focused verifier tests, the repository suite, the smoke profile, and
   an extended seeded sweep.

Do not call PunyTorch operation or backward internals from the reference
evaluator. Sharing serialization and comparison code is fine; sharing the
mathematical implementation would defeat the differential check.

## Promote a failure to a regression

Replay the artifact first. Reduce its values or graph by editing a copy of the
JSON while keeping the failure and revision explicit. Once the case is small,
translate it into a named scenario fixture under `tests/differential/` and
assert semantic comparison fields and numerical behavior. Do not check in a
large generated corpus or snapshot a verbose report.

Automatic shrinking is not implemented in this first version. Graph suffix
removal is unsafe when it changes the scalar loss, shape simplification needs
operation-aware rewrites, and value reduction must preserve domain constraints.
The serialized named DAG makes a bounded deterministic reducer a contained
follow-up without making replay depend on it.

Highest-value next extensions are a trustworthy reducer, a dedicated
`matmul`/broadcast family, non-smooth subgradient policies, tuple-axis
reductions, and an explicit `float32` tolerance profile.
