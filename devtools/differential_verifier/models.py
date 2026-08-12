"""Serializable data model for deterministic differential scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from devtools.differential_verifier.errors import InvalidScenario, UnsupportedFeature

SCENARIO_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_DTYPES = ("float64",)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ValueDomain:
    kind: str
    minimum: float
    maximum: float
    min_abs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "min_abs": self.min_abs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueDomain":
        try:
            return cls(
                kind=str(data["kind"]),
                minimum=float(data["minimum"]),
                maximum=float(data["maximum"]),
                min_abs=float(data.get("min_abs", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid value domain: {error}") from error

    def validate(self, name: str, array: np.ndarray) -> None:
        if self.kind not in {"bounded", "positive", "nonzero"}:
            raise InvalidScenario(f"input {name!r} has unknown domain kind {self.kind!r}")
        if self.minimum > self.maximum or self.min_abs < 0:
            raise InvalidScenario(f"input {name!r} has inconsistent domain bounds")
        if not np.all(np.isfinite(array)):
            raise InvalidScenario(f"input {name!r} contains non-finite values")
        if np.any(array < self.minimum) or np.any(array > self.maximum):
            raise InvalidScenario(f"input {name!r} violates its declared [{self.minimum}, {self.maximum}] domain")
        if self.kind == "positive" and np.any(array <= 0):
            raise InvalidScenario(f"input {name!r} must be positive")
        if self.kind == "nonzero" and np.any(np.abs(array) < self.min_abs):
            raise InvalidScenario(f"input {name!r} violates min_abs={self.min_abs}")


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    values: Any
    domain: ValueDomain
    requires_grad: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "values": _plain(self.values),
            "domain": self.domain.to_dict(),
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputSpec":
        try:
            shape = tuple(int(size) for size in data["shape"])
            return cls(
                name=str(data["name"]),
                shape=shape,
                dtype=str(data["dtype"]),
                values=data["values"],
                domain=ValueDomain.from_dict(data["domain"]),
                requires_grad=bool(data.get("requires_grad", True)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid input specification: {error}") from error

    def array(self) -> np.ndarray:
        if self.dtype not in SUPPORTED_DTYPES:
            raise UnsupportedFeature(
                f"dtype {self.dtype!r} is unsupported; supported dtypes: {', '.join(SUPPORTED_DTYPES)}"
            )
        if any(size <= 0 for size in self.shape):
            raise InvalidScenario(f"input {self.name!r} must have a positive shape")
        try:
            array = np.asarray(self.values, dtype=np.dtype(self.dtype))
        except (TypeError, ValueError) as error:
            raise InvalidScenario(f"input {self.name!r} values cannot be converted to {self.dtype}") from error
        if array.shape != self.shape:
            raise InvalidScenario(
                f"input {self.name!r} declares shape {self.shape}, but values have shape {array.shape}"
            )
        self.domain.validate(self.name, array)
        return array.copy()


@dataclass(frozen=True)
class NodeSpec:
    name: str
    op: str
    inputs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "inputs": list(self.inputs),
            "params": _plain(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeSpec":
        try:
            return cls(
                name=str(data["name"]),
                op=str(data["op"]),
                inputs=tuple(str(name) for name in data["inputs"]),
                params=dict(data.get("params", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid node specification: {error}") from error


@dataclass(frozen=True)
class LossSpec:
    op: str
    input: str
    weights: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"op": self.op, "input": self.input}
        if self.weights is not None:
            data["weights"] = _plain(self.weights)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossSpec":
        try:
            return cls(op=str(data["op"]), input=str(data["input"]), weights=data.get("weights"))
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid loss specification: {error}") from error


@dataclass(frozen=True)
class Tolerances:
    forward_atol: float
    forward_rtol: float
    gradient_atol: float
    gradient_rtol: float
    finite_difference_step: float

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_atol": self.forward_atol,
            "forward_rtol": self.forward_rtol,
            "gradient_atol": self.gradient_atol,
            "gradient_rtol": self.gradient_rtol,
            "finite_difference_step": self.finite_difference_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tolerances":
        try:
            result = cls(
                forward_atol=float(data["forward_atol"]),
                forward_rtol=float(data["forward_rtol"]),
                gradient_atol=float(data["gradient_atol"]),
                gradient_rtol=float(data["gradient_rtol"]),
                finite_difference_step=float(data["finite_difference_step"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid tolerances: {error}") from error
        if any(value < 0 for value in result.to_dict().values()) or result.finite_difference_step == 0:
            raise InvalidScenario("tolerances must be non-negative and the finite-difference step must be positive")
        return result


@dataclass(frozen=True)
class MetamorphicSpec:
    kind: str
    input: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "input": self.input, "params": _plain(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetamorphicSpec":
        try:
            return cls(
                kind=str(data["kind"]),
                input=str(data["input"]),
                params=dict(data.get("params", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid metamorphic check: {error}") from error


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    case_index: int
    revision: str
    source_fingerprint: str
    inputs: tuple[InputSpec, ...]
    nodes: tuple[NodeSpec, ...]
    output: str
    loss: LossSpec
    tolerances: Tolerances
    metamorphic_checks: tuple[MetamorphicSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "case_index": self.case_index,
            "revision": self.revision,
            "source_fingerprint": self.source_fingerprint,
            "inputs": [item.to_dict() for item in self.inputs],
            "nodes": [item.to_dict() for item in self.nodes],
            "output": self.output,
            "loss": self.loss.to_dict(),
            "tolerances": self.tolerances.to_dict(),
            "metamorphic_checks": [item.to_dict() for item in self.metamorphic_checks],
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        try:
            scenario = cls(
                schema_version=int(data["schema_version"]),
                scenario_id=str(data["scenario_id"]),
                seed=int(data["seed"]),
                case_index=int(data["case_index"]),
                revision=str(data["revision"]),
                source_fingerprint=str(data["source_fingerprint"]),
                inputs=tuple(InputSpec.from_dict(item) for item in data["inputs"]),
                nodes=tuple(NodeSpec.from_dict(item) for item in data["nodes"]),
                output=str(data["output"]),
                loss=LossSpec.from_dict(data["loss"]),
                tolerances=Tolerances.from_dict(data["tolerances"]),
                metamorphic_checks=tuple(
                    MetamorphicSpec.from_dict(item) for item in data.get("metamorphic_checks", [])
                ),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScenario(f"invalid scenario: {error}") from error
        scenario.validate_static()
        return scenario

    def validate_static(self) -> None:
        if self.schema_version != SCENARIO_SCHEMA_VERSION:
            raise InvalidScenario(
                f"scenario schema {self.schema_version} is unsupported; expected {SCENARIO_SCHEMA_VERSION}"
            )
        if not self.scenario_id or not self.revision or not self.source_fingerprint:
            raise InvalidScenario("scenario_id, revision, and source_fingerprint must be non-empty")
        if self.case_index < 0:
            raise InvalidScenario("case_index must be non-negative")
        Tolerances.from_dict(self.tolerances.to_dict())
        if not self.inputs:
            raise InvalidScenario("scenario must contain at least one input")
        available: set[str] = set()
        for input_spec in self.inputs:
            if not input_spec.name or input_spec.name in available:
                raise InvalidScenario(f"duplicate or empty value name {input_spec.name!r}")
            input_spec.array()
            available.add(input_spec.name)
        for node in self.nodes:
            if not node.name or node.name in available:
                raise InvalidScenario(f"duplicate or empty value name {node.name!r}")
            missing = [name for name in node.inputs if name not in available]
            if missing:
                raise InvalidScenario(f"node {node.name!r} references unavailable values: {missing}")
            if not node.inputs:
                raise InvalidScenario(f"node {node.name!r} must have at least one input")
            available.add(node.name)
        if self.output not in available:
            raise InvalidScenario(f"output {self.output!r} does not exist")
        if self.loss.input != self.output:
            raise InvalidScenario("loss input must name the scenario output")
        if self.loss.op not in {"sum", "mean", "weighted_sum"}:
            raise InvalidScenario(f"unknown scalar loss reduction {self.loss.op!r}")
        if self.loss.op == "weighted_sum" and self.loss.weights is None:
            raise InvalidScenario("weighted_sum loss requires serialized weights")
        input_names = {item.name for item in self.inputs}
        for check in self.metamorphic_checks:
            if check.input not in input_names:
                raise InvalidScenario(
                    f"metamorphic check {check.kind!r} must reference a declared input, got {check.input!r}"
                )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def scenario_from_json(data: str) -> Scenario:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise InvalidScenario(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise InvalidScenario("scenario JSON must contain an object")
    if "scenario" in payload:
        payload = payload["scenario"]
    if not isinstance(payload, dict):
        raise InvalidScenario("artifact scenario must contain an object")
    return Scenario.from_dict(payload)


def load_scenario(path: Path) -> Scenario:
    try:
        return scenario_from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InvalidScenario(f"cannot read scenario {path}: {error}") from error
