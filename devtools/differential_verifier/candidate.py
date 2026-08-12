"""Execution through the real public PunyTorch Tensor/autograd API."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np

from devtools.differential_verifier.errors import InvalidScenario, UnsupportedFeature
from devtools.differential_verifier.models import NodeSpec, Scenario
from punytorch import Tensor

ObservedCorruptor = Callable[[str, np.ndarray], np.ndarray]
GradientCorruptor = Callable[[str, np.ndarray], np.ndarray]
BinaryOperation = Callable[[Tensor, Tensor], Tensor]
UnaryOperation = Callable[[Tensor], Tensor]
ReductionOperation = Callable[..., Tensor]
ParameterizedOperation = Callable[[NodeSpec, list[Tensor]], Tensor]


@dataclass(frozen=True)
class CandidateExecution:
    node_values: dict[str, np.ndarray]
    output: np.ndarray
    loss: float
    gradients: dict[str, np.ndarray]


_BINARY_OPERATIONS: dict[str, BinaryOperation] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}

_UNARY_OPERATIONS: dict[str, UnaryOperation] = {
    "tanh": Tensor.tanh,
    "sigmoid": Tensor.sigmoid,
    "exp": Tensor.exp,
    "log": Tensor.log,
}

_REDUCTION_OPERATIONS: dict[str, ReductionOperation] = {
    "sum": Tensor.sum,
    "mean": Tensor.mean,
    "logsumexp": Tensor.logsumexp,
}


def _reshape(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return values[0].reshape(tuple(int(size) for size in node.params["shape"]))


def _transpose(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return values[0].transpose(int(node.params["dim0"]), int(node.params["dim1"]))


def _stack(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return Tensor.stack(values, axis=int(node.params.get("axis", 0)))


def _cat(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return Tensor.cat(values, dim=int(node.params.get("axis", 0)))


def _indices(node: NodeSpec) -> np.ndarray:
    return np.asarray(node.params["indices"], dtype=np.int64)


def _index(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return values[0][_indices(node)]


def _gather(node: NodeSpec, values: list[Tensor]) -> Tensor:
    return values[0].gather(_indices(node))


_PARAMETERIZED_OPERATIONS: dict[str, ParameterizedOperation] = {
    "reshape": _reshape,
    "transpose": _transpose,
    "stack": _stack,
    "cat": _cat,
    "index": _index,
    "gather": _gather,
}

_OPERATION_GROUPS = (
    _BINARY_OPERATIONS,
    _UNARY_OPERATIONS,
    _REDUCTION_OPERATIONS,
    _PARAMETERIZED_OPERATIONS,
)
SUPPORTED_OPERATIONS = frozenset(name for group in _OPERATION_GROUPS for name in group)

if sum(len(group) for group in _OPERATION_GROUPS) != len(SUPPORTED_OPERATIONS):
    raise RuntimeError("candidate operation names must belong to exactly one dispatch group")


def _apply_node(node: NodeSpec, values: list[Tensor]) -> Tensor:
    binary = _BINARY_OPERATIONS.get(node.op)
    if binary is not None:
        return binary(values[0], values[1])

    unary = _UNARY_OPERATIONS.get(node.op)
    if unary is not None:
        return unary(values[0])

    reduction = _REDUCTION_OPERATIONS.get(node.op)
    if reduction is not None:
        return reduction(
            values[0],
            axis=node.params.get("axis"),
            keepdims=bool(node.params.get("keepdims", False)),
        )

    parameterized = _PARAMETERIZED_OPERATIONS.get(node.op)
    if parameterized is not None:
        return parameterized(node, values)

    raise UnsupportedFeature(f"candidate executor does not support operation {node.op!r}")


def _apply_loss(scenario: Scenario, output: Tensor) -> Tensor:
    if scenario.loss.op == "sum":
        return output.sum()
    if scenario.loss.op == "mean":
        return output.mean()
    if scenario.loss.op == "weighted_sum":
        weights = np.asarray(scenario.loss.weights, dtype=np.float64)
        if weights.shape != output.shape:
            raise InvalidScenario(
                f"weighted_sum loss weights have shape {weights.shape}, but output has shape {output.shape}"
            )
        return (output * Tensor(weights)).sum()
    raise InvalidScenario(f"unknown loss reduction {scenario.loss.op!r}")


class CandidateExecutor:
    """Execute with optional observation hooks used only by verifier self-tests."""

    def __init__(
        self,
        *,
        observed_corruptor: ObservedCorruptor | None = None,
        gradient_corruptor: GradientCorruptor | None = None,
    ):
        self.observed_corruptor = observed_corruptor
        self.gradient_corruptor = gradient_corruptor

    def execute(self, scenario: Scenario) -> CandidateExecution:
        scenario.validate_static()
        environment: dict[str, Tensor] = {
            item.name: Tensor(item.array(), requires_grad=item.requires_grad) for item in scenario.inputs
        }
        observed: dict[str, np.ndarray] = {}

        for node in scenario.nodes:
            value = _apply_node(node, [environment[name] for name in node.inputs])
            environment[node.name] = value
            snapshot = np.asarray(value.data).copy()
            if self.observed_corruptor is not None:
                snapshot = np.asarray(self.observed_corruptor(node.name, snapshot)).copy()
            observed[node.name] = snapshot

        output = environment[scenario.output]
        loss = _apply_loss(scenario, output)
        if np.asarray(loss.data).size != 1:
            raise InvalidScenario(f"loss {scenario.loss.op!r} did not produce a scalar")
        loss.backward()

        gradients: dict[str, np.ndarray] = {}
        for item in scenario.inputs:
            if not item.requires_grad:
                continue
            tensor = environment[item.name]
            if tensor.grad is None:
                raise RuntimeError(f"candidate did not produce a gradient for input {item.name!r}")
            gradient = np.asarray(tensor.grad, dtype=np.float64).copy()
            if self.gradient_corruptor is not None:
                gradient = np.asarray(self.gradient_corruptor(item.name, gradient), dtype=np.float64).copy()
            gradients[item.name] = gradient

        return CandidateExecution(
            node_values=observed,
            output=np.asarray(output.data).copy(),
            loss=float(np.asarray(loss.data).reshape(-1)[0]),
            gradients=gradients,
        )
