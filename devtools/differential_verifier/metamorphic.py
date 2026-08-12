"""Independent equivalent-formulation and analytic-property checks."""

from __future__ import annotations

import numpy as np

from devtools.differential_verifier.compare import Comparison, compare_arrays
from devtools.differential_verifier.errors import UnsupportedFeature
from devtools.differential_verifier.models import MetamorphicSpec, Scenario
from punytorch import Tensor


def _input_array(scenario: Scenario, name: str) -> np.ndarray:
    return next(item.array() for item in scenario.inputs if item.name == name)


def _reshape_roundtrip(scenario: Scenario, check: MetamorphicSpec) -> Comparison:
    original = _input_array(scenario, check.input)
    target = tuple(int(size) for size in check.params["shape"])
    observed = Tensor(original).reshape(target).reshape(original.shape).data
    return compare_arrays(
        kind="metamorphic",
        target="reshape_roundtrip",
        expected=original,
        observed=observed,
        atol=scenario.tolerances.forward_atol,
        rtol=scenario.tolerances.forward_rtol,
        detail=f"{check.input} -> {target} -> {original.shape}",
    )


def _reduction_consistency(scenario: Scenario, check: MetamorphicSpec) -> Comparison:
    original = _input_array(scenario, check.input)
    axis = int(check.params["axis"])
    tensor = Tensor(original)
    observed = tensor.sum(axis=axis).sum().data
    expected = tensor.sum().data
    return compare_arrays(
        kind="metamorphic",
        target="reduction_consistency",
        expected=expected,
        observed=observed,
        atol=scenario.tolerances.forward_atol,
        rtol=scenario.tolerances.forward_rtol,
        detail=f"sum({check.input}) == sum(sum({check.input}, axis={axis}))",
    )


def _logsumexp_shift(scenario: Scenario, check: MetamorphicSpec) -> Comparison:
    original = _input_array(scenario, check.input)
    shift = float(check.params["shift"])
    axis = check.params.get("axis")
    keepdims = bool(check.params.get("keepdims", False))
    tensor = Tensor(original)
    observed = (tensor + shift).logsumexp(axis=axis, keepdims=keepdims).data
    expected = (tensor.logsumexp(axis=axis, keepdims=keepdims) + shift).data
    return compare_arrays(
        kind="metamorphic",
        target="logsumexp_shift",
        expected=expected,
        observed=observed,
        atol=scenario.tolerances.forward_atol,
        rtol=scenario.tolerances.forward_rtol,
        detail=f"logsumexp({check.input} + {shift}) == logsumexp({check.input}) + {shift}",
    )


def _repeated_index_gradient(scenario: Scenario, check: MetamorphicSpec) -> Comparison:
    original = _input_array(scenario, check.input)
    indices = np.asarray(check.params["indices"], dtype=np.int64)
    tensor = Tensor(original, requires_grad=True)
    gathered = tensor.gather(indices)
    gathered.sum().backward()

    expected = np.zeros_like(original, dtype=np.float64)
    for index_value in indices.flat:
        expected[int(index_value)] += 1.0
    return compare_arrays(
        kind="metamorphic",
        target="repeated_index_gradient",
        expected=expected,
        observed=tensor.grad,
        atol=scenario.tolerances.gradient_atol,
        rtol=scenario.tolerances.gradient_rtol,
        detail=f"repeated indices {indices.tolist()} must accumulate",
    )


def run_metamorphic_checks(scenario: Scenario) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for check in scenario.metamorphic_checks:
        if check.kind == "reshape_roundtrip":
            comparisons.append(_reshape_roundtrip(scenario, check))
        elif check.kind == "reduction_consistency":
            comparisons.append(_reduction_consistency(scenario, check))
        elif check.kind == "logsumexp_shift":
            comparisons.append(_logsumexp_shift(scenario, check))
        elif check.kind == "repeated_index_gradient":
            comparisons.append(_repeated_index_gradient(scenario, check))
        else:
            raise UnsupportedFeature(f"metamorphic check {check.kind!r} is unsupported")
    return comparisons
