"""Deterministic, shape-compatible scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from devtools.differential_verifier.errors import GeneratorRejection
from devtools.differential_verifier.models import (
    InputSpec,
    LossSpec,
    MetamorphicSpec,
    NodeSpec,
    Scenario,
    Tolerances,
    ValueDomain,
)

INITIAL_OPERATIONS = (
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

FLOAT64_TOLERANCES = Tolerances(
    forward_atol=1e-10,
    forward_rtol=1e-10,
    gradient_atol=3e-5,
    gradient_rtol=3e-5,
    finite_difference_step=1e-6,
)

_BOUNDED = ValueDomain("bounded", -1.5, 1.5)
_NONZERO = ValueDomain("nonzero", -2.0, 2.0, min_abs=0.35)


@dataclass(frozen=True)
class GeneratedGraph:
    inputs: tuple[InputSpec, ...]
    nodes: tuple[NodeSpec, ...]
    output: str
    output_shape: tuple[int, ...]
    metamorphic_checks: tuple[MetamorphicSpec, ...] = ()
    rejection_count: int = 0


def _input(name: str, values: np.ndarray, domain: ValueDomain = _BOUNDED) -> InputSpec:
    return InputSpec(
        name=name,
        shape=values.shape,
        dtype="float64",
        values=values.tolist(),
        domain=domain,
        requires_grad=True,
    )


def _bounded_values(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.uniform(-1.25, 1.25, size=shape).astype(np.float64)


def _nonzero_values(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    magnitudes = rng.uniform(0.45, 1.8, size=shape)
    signs = rng.choice(np.array([-1.0, 1.0]), size=shape)
    return (magnitudes * signs).astype(np.float64)


def _matrix_shape(rng: np.random.Generator, profile: str) -> tuple[int, int]:
    upper = 4 if profile == "extended" else 3
    return int(rng.integers(2, upper + 1)), int(rng.integers(2, upper + 1))


def _elementwise(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    shape = _matrix_shape(rng, profile)
    x = _bounded_values(rng, shape)
    y = _bounded_values(rng, shape)
    denominator = _nonzero_values(rng, shape)
    nodes = (
        NodeSpec("sum_xy", "add", ("x", "y")),
        NodeSpec("product", "mul", ("sum_xy", "x")),
        NodeSpec("difference", "sub", ("product", "y")),
        NodeSpec("quotient", "div", ("difference", "denominator")),
    )
    return GeneratedGraph(
        inputs=(_input("x", x), _input("y", y), _input("denominator", denominator, _NONZERO)),
        nodes=nodes,
        output="quotient",
        output_shape=shape,
    )


def _nonlinear(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    shape = _matrix_shape(rng, profile)
    x = _bounded_values(rng, shape)
    nodes = (
        NodeSpec("tanh_x", "tanh", ("x",)),
        NodeSpec("sigmoid_x", "sigmoid", ("x",)),
        NodeSpec("blended", "add", ("tanh_x", "sigmoid_x")),
        NodeSpec("exponential", "exp", ("blended",)),
        NodeSpec("logged", "log", ("exponential",)),
    )
    return GeneratedGraph(inputs=(_input("x", x),), nodes=nodes, output="logged", output_shape=shape)


def _shape_and_reduction(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    rows, columns = _matrix_shape(rng, profile)
    shape = (rows, columns)
    target = (columns, rows)
    x = _bounded_values(rng, shape)
    nodes = (
        NodeSpec("reshaped", "reshape", ("x",), {"shape": list(target)}),
        NodeSpec("transposed", "transpose", ("reshaped",), {"dim0": 0, "dim1": 1}),
        NodeSpec("row_sums", "sum", ("transposed",), {"axis": 1, "keepdims": True}),
        NodeSpec("row_means", "mean", ("transposed",), {"axis": 1, "keepdims": True}),
        NodeSpec("combined", "add", ("row_sums", "row_means")),
    )
    metamorphic = (
        MetamorphicSpec("reshape_roundtrip", "x", {"shape": list(target)}),
        MetamorphicSpec("reduction_consistency", "x", {"axis": 1}),
    )
    return GeneratedGraph(
        inputs=(_input("x", x),),
        nodes=nodes,
        output="combined",
        output_shape=(rows, 1),
        metamorphic_checks=metamorphic,
    )


def _combination(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    rows, columns = _matrix_shape(rng, profile)
    shape = (rows, columns)
    x = _bounded_values(rng, shape)
    y = _bounded_values(rng, shape)
    nodes = (
        NodeSpec("stacked", "stack", ("x", "y"), {"axis": 0}),
        NodeSpec("stack_sum", "sum", ("stacked",), {"axis": 0, "keepdims": False}),
        NodeSpec("concatenated", "cat", ("x", "y"), {"axis": 1}),
        NodeSpec("cat_reshaped", "reshape", ("concatenated",), {"shape": [2, rows, columns]}),
        NodeSpec("cat_mean", "mean", ("cat_reshaped",), {"axis": 0, "keepdims": False}),
        NodeSpec("combined", "add", ("stack_sum", "cat_mean")),
    )
    return GeneratedGraph(
        inputs=(_input("x", x), _input("y", y)),
        nodes=nodes,
        output="combined",
        output_shape=shape,
    )


def _indexing(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    columns = 4 if profile == "extended" else 3
    shape = (4, columns)
    x = _bounded_values(rng, shape)
    repeated = int(rng.integers(0, shape[0]))
    other = (repeated + int(rng.integers(1, shape[0]))) % shape[0]
    indices = [repeated, repeated, other]
    reverse_indices = list(reversed(indices))
    nodes = (
        NodeSpec("indexed", "index", ("x",), {"indices": indices}),
        NodeSpec("gathered", "gather", ("x",), {"indices": reverse_indices}),
        NodeSpec("combined", "add", ("indexed", "gathered")),
    )
    metamorphic = (MetamorphicSpec("repeated_index_gradient", "x", {"indices": indices}),)
    return GeneratedGraph(
        inputs=(_input("x", x),),
        nodes=nodes,
        output="combined",
        output_shape=(3, columns),
        metamorphic_checks=metamorphic,
    )


def _stable_reduction(rng: np.random.Generator, profile: str) -> GeneratedGraph:
    rows, columns = _matrix_shape(rng, profile)
    x = _bounded_values(rng, (rows, columns))
    bias = rng.uniform(-0.35, 0.35, size=(1, columns)).astype(np.float64)
    nodes = (
        NodeSpec("shifted", "add", ("x", "bias")),
        NodeSpec("stable_lse", "logsumexp", ("shifted",), {"axis": 1, "keepdims": True}),
        NodeSpec("row_mean", "mean", ("x",), {"axis": 1, "keepdims": True}),
        NodeSpec("combined", "add", ("stable_lse", "row_mean")),
    )
    metamorphic = (MetamorphicSpec("logsumexp_shift", "x", {"axis": 1, "keepdims": False, "shift": 0.75}),)
    return GeneratedGraph(
        inputs=(_input("x", x), _input("bias", bias)),
        nodes=nodes,
        output="combined",
        output_shape=(rows, 1),
        metamorphic_checks=metamorphic,
    )


_FAMILIES: tuple[tuple[str, Callable[[np.random.Generator, str], GeneratedGraph]], ...] = (
    ("elementwise", _elementwise),
    ("nonlinear", _nonlinear),
    ("shape_reduction", _shape_and_reduction),
    ("combination", _combination),
    ("indexing", _indexing),
    ("stable_reduction", _stable_reduction),
)


def generate_scenario(
    *,
    seed: int,
    case_index: int,
    revision: str,
    source_fingerprint: str | None = None,
    profile: str = "smoke",
) -> Scenario:
    """Generate one case independently from ``seed`` and ``case_index``."""

    if profile not in {"smoke", "extended"}:
        raise GeneratorRejection(f"unknown generation profile {profile!r}")
    if case_index < 0:
        raise GeneratorRejection("case_index must be non-negative")
    if not revision:
        raise GeneratorRejection("revision must be non-empty")

    seed_sequence = np.random.SeedSequence([int(seed), int(case_index)])
    case_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(case_seed)
    family_name, factory = _FAMILIES[case_index % len(_FAMILIES)]
    generated = factory(rng, profile)
    weights = rng.uniform(-1.0, 1.0, size=generated.output_shape).astype(np.float64)
    if not np.any(np.abs(weights) > 0.1):
        weights.flat[0] = 1.0

    scenario = Scenario(
        scenario_id=f"seed-{seed}-case-{case_index:04d}",
        seed=int(seed),
        case_index=case_index,
        revision=revision,
        source_fingerprint=source_fingerprint or revision,
        inputs=generated.inputs,
        nodes=generated.nodes,
        output=generated.output,
        loss=LossSpec("weighted_sum", generated.output, weights.tolist()),
        tolerances=FLOAT64_TOLERANCES,
        metamorphic_checks=generated.metamorphic_checks,
        metadata={
            "profile": profile,
            "family": family_name,
            "case_seed": case_seed,
            "generator_rejections": generated.rejection_count,
        },
    )
    scenario.validate_static()
    return scenario
