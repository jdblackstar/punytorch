"""Independent NumPy evaluation of serialized computation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from devtools.differential_verifier.errors import InvalidScenario, NumericalDomainSkip, UnsupportedFeature
from devtools.differential_verifier.models import NodeSpec, Scenario

SUPPORTED_OPERATIONS = (
    "add",
    "sub",
    "mul",
    "div",
    "sum",
    "mean",
    "reshape",
    "transpose",
    "tanh",
    "sigmoid",
    "exp",
    "log",
    "stack",
    "cat",
    "index",
    "gather",
    "logsumexp",
)


@dataclass(frozen=True)
class ReferenceExecution:
    node_values: dict[str, np.ndarray]
    output: np.ndarray
    loss: float


def _axis(params: dict, name: str = "axis"):
    axis = params.get(name)
    if isinstance(axis, list):
        return tuple(int(value) for value in axis)
    return axis


def _stable_logsumexp(value: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    shifted = value - maximum
    result = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + maximum
    return result if keepdims else np.squeeze(result, axis=axis)


def _ensure_arity(node: NodeSpec, expected: int | tuple[int, ...]) -> None:
    allowed = (expected,) if isinstance(expected, int) else expected
    if len(node.inputs) not in allowed:
        formatted = ", ".join(str(value) for value in allowed)
        raise InvalidScenario(f"operation {node.op!r} expects input count in ({formatted}), got {len(node.inputs)}")


def _evaluate_node(node: NodeSpec, values: list[np.ndarray]) -> np.ndarray:
    op = node.op
    params = node.params
    if op not in SUPPORTED_OPERATIONS:
        raise UnsupportedFeature(f"operation {op!r} is not in the verifier's supported slice")

    if op in {"add", "sub", "mul", "div"}:
        _ensure_arity(node, 2)
        left, right = values
        if op == "add":
            return np.add(left, right)
        if op == "sub":
            return np.subtract(left, right)
        if op == "mul":
            return np.multiply(left, right)
        if np.any(np.abs(right) < 1e-8):
            raise NumericalDomainSkip(f"node {node.name!r} divides by a value with magnitude below 1e-8")
        return np.divide(left, right)

    if op in {"sum", "mean", "logsumexp"}:
        _ensure_arity(node, 1)
        axis = _axis(params)
        keepdims = bool(params.get("keepdims", False))
        if op == "sum":
            return np.sum(values[0], axis=axis, keepdims=keepdims)
        if op == "mean":
            return np.mean(values[0], axis=axis, keepdims=keepdims)
        return _stable_logsumexp(values[0], axis=axis, keepdims=keepdims)

    if op == "reshape":
        _ensure_arity(node, 1)
        shape = tuple(int(size) for size in params["shape"])
        return np.reshape(values[0], shape)

    if op == "transpose":
        _ensure_arity(node, 1)
        return np.swapaxes(values[0], int(params["dim0"]), int(params["dim1"]))

    if op in {"tanh", "sigmoid", "exp", "log"}:
        _ensure_arity(node, 1)
        value = values[0]
        if op == "tanh":
            return np.tanh(value)
        if op == "sigmoid":
            return 1.0 / (1.0 + np.exp(-value))
        if op == "exp":
            if np.any(np.abs(value) > 20):
                raise NumericalDomainSkip(f"node {node.name!r} has exp input outside [-20, 20]")
            return np.exp(value)
        if np.any(value <= 0):
            raise NumericalDomainSkip(f"node {node.name!r} applies log to non-positive values")
        return np.log(value)

    if op in {"stack", "cat"}:
        _ensure_arity(node, tuple(range(2, 17)))
        axis = int(params.get("axis", 0))
        if op == "stack":
            return np.stack(values, axis=axis)
        return np.concatenate(values, axis=axis)

    if op in {"index", "gather"}:
        _ensure_arity(node, 1)
        indices = np.asarray(params["indices"], dtype=np.int64)
        return values[0][indices]

    raise AssertionError(f"unreachable operation {op}")


def _loss_value(scenario: Scenario, output: np.ndarray) -> float:
    if scenario.loss.op == "sum":
        return float(np.sum(output))
    if scenario.loss.op == "mean":
        return float(np.mean(output))
    weights = np.asarray(scenario.loss.weights, dtype=np.float64)
    if weights.shape != output.shape:
        raise InvalidScenario(
            f"weighted_sum loss weights have shape {weights.shape}, but output has shape {output.shape}"
        )
    return float(np.sum(output * weights))


class ReferenceEvaluator:
    """Evaluate a scenario without importing or calling PunyTorch operations."""

    def evaluate(
        self,
        scenario: Scenario,
        *,
        input_overrides: Mapping[str, np.ndarray] | None = None,
    ) -> ReferenceExecution:
        scenario.validate_static()
        overrides = input_overrides or {}
        environment: dict[str, np.ndarray] = {}
        for input_spec in scenario.inputs:
            value = overrides.get(input_spec.name)
            if value is None:
                value = input_spec.array()
            value = np.asarray(value, dtype=np.dtype(input_spec.dtype))
            if value.shape != input_spec.shape:
                raise InvalidScenario(
                    f"override for {input_spec.name!r} has shape {value.shape}, expected {input_spec.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise NumericalDomainSkip(f"override for {input_spec.name!r} contains non-finite values")
            environment[input_spec.name] = value.copy()

        node_values: dict[str, np.ndarray] = {}
        for node in scenario.nodes:
            try:
                value = np.asarray(_evaluate_node(node, [environment[name] for name in node.inputs]))
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise InvalidScenario(f"node {node.name!r} ({node.op}) is invalid: {error}") from error
            if not np.all(np.isfinite(value)):
                raise NumericalDomainSkip(f"node {node.name!r} ({node.op}) produced non-finite values")
            environment[node.name] = value
            node_values[node.name] = value.copy()

        output = environment[scenario.output]
        loss = _loss_value(scenario, output)
        if not np.isfinite(loss):
            raise NumericalDomainSkip("scalar reference loss is non-finite")
        return ReferenceExecution(node_values=node_values, output=output.copy(), loss=loss)
