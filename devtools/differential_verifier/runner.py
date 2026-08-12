"""Verification orchestration, finite differences, sweeps, and artifacts."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np

from devtools.differential_verifier.candidate import CandidateExecutor
from devtools.differential_verifier.compare import Comparison, compare_arrays
from devtools.differential_verifier.errors import (
    GeneratorRejection,
    InvalidScenario,
    NumericalDomainSkip,
    UnsupportedFeature,
)
from devtools.differential_verifier.generator import generate_scenario
from devtools.differential_verifier.metamorphic import run_metamorphic_checks
from devtools.differential_verifier.models import ARTIFACT_SCHEMA_VERSION, Scenario
from devtools.differential_verifier.reference import ReferenceEvaluator


class Outcome(str, Enum):
    PASS = "pass"
    MISMATCH = "mismatch"
    INVALID_GRAPH = "invalid_graph"
    UNSUPPORTED = "unsupported"
    NUMERICAL_SKIP = "numerical_skip"
    CANDIDATE_ERROR = "candidate_error"
    GENERATOR_REJECTION = "generator_rejection"


@dataclass(frozen=True)
class RunResult:
    scenario_id: str
    outcome: Outcome
    comparisons: tuple[Comparison, ...] = ()
    message: str = ""

    @property
    def failures(self) -> tuple[Comparison, ...]:
        return tuple(item for item in self.comparisons if not item.passed)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "outcome": self.outcome.value,
            "message": self.message,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


def _comparison_outcome(
    comparisons: Iterable[Comparison],
    *,
    otherwise: Outcome,
) -> Outcome:
    """Keep confirmed numerical mismatches from being masked by later control flow."""

    return Outcome.MISMATCH if any(not item.passed for item in comparisons) else otherwise


@dataclass(frozen=True)
class CaseRun:
    scenario: Scenario
    result: RunResult


@dataclass(frozen=True)
class SweepResult:
    seed: int
    requested_cases: int
    revision: str
    source_fingerprint: str
    profile: str
    cases: tuple[CaseRun, ...]
    generator_rejections: int
    elapsed_seconds: float
    operation_counts: dict[str, int]

    @property
    def successful(self) -> bool:
        return len(self.cases) == self.requested_cases and all(
            case.result.outcome in {Outcome.PASS, Outcome.NUMERICAL_SKIP} for case in self.cases
        )

    def outcome_counts(self) -> dict[str, int]:
        counts = Counter(case.result.outcome.value for case in self.cases)
        return {outcome.value: counts.get(outcome.value, 0) for outcome in Outcome}

    def check_counts(self) -> dict[str, int]:
        counts = Counter(comparison.kind for case in self.cases for comparison in case.result.comparisons)
        return {kind: counts[kind] for kind in sorted(counts)}


def _finite_difference_gradient(
    scenario: Scenario,
    input_name: str,
    evaluator: ReferenceEvaluator,
) -> np.ndarray:
    inputs = {item.name: item.array() for item in scenario.inputs}
    source = inputs[input_name]
    gradient = np.zeros_like(source, dtype=np.float64)
    base_step = scenario.tolerances.finite_difference_step

    for index in np.ndindex(source.shape):
        step = base_step * max(1.0, abs(float(source[index])))
        plus = {name: value.copy() for name, value in inputs.items()}
        minus = {name: value.copy() for name, value in inputs.items()}
        plus[input_name][index] += step
        minus[input_name][index] -= step
        plus_loss = evaluator.evaluate(scenario, input_overrides=plus).loss
        minus_loss = evaluator.evaluate(scenario, input_overrides=minus).loss
        gradient[index] = (plus_loss - minus_loss) / (2.0 * step)
    return gradient


def verify_scenario(
    scenario: Scenario,
    *,
    candidate: CandidateExecutor | None = None,
    reference: ReferenceEvaluator | None = None,
) -> RunResult:
    """Run all verification classes in deterministic report order."""

    candidate = candidate or CandidateExecutor()
    reference = reference or ReferenceEvaluator()
    try:
        scenario.validate_static()
        reference_execution = reference.evaluate(scenario)
    except InvalidScenario as error:
        return RunResult(scenario.scenario_id, Outcome.INVALID_GRAPH, message=str(error))
    except UnsupportedFeature as error:
        return RunResult(scenario.scenario_id, Outcome.UNSUPPORTED, message=str(error))
    except NumericalDomainSkip as error:
        return RunResult(scenario.scenario_id, Outcome.NUMERICAL_SKIP, message=str(error))

    try:
        candidate_execution = candidate.execute(scenario)
    except (InvalidScenario, UnsupportedFeature) as error:
        return RunResult(scenario.scenario_id, Outcome.CANDIDATE_ERROR, message=str(error))
    except Exception as error:  # Candidate exceptions are evidence, not verifier crashes.
        return RunResult(
            scenario.scenario_id,
            Outcome.CANDIDATE_ERROR,
            message=f"{type(error).__name__}: {error}",
        )

    comparisons: list[Comparison] = []
    for node in scenario.nodes:
        comparisons.append(
            compare_arrays(
                kind="forward",
                target=f"node:{node.name}",
                expected=reference_execution.node_values[node.name],
                observed=candidate_execution.node_values[node.name],
                atol=scenario.tolerances.forward_atol,
                rtol=scenario.tolerances.forward_rtol,
                detail=f"operation={node.op}",
            )
        )
    comparisons.append(
        compare_arrays(
            kind="forward",
            target="loss",
            expected=reference_execution.loss,
            observed=candidate_execution.loss,
            atol=scenario.tolerances.forward_atol,
            rtol=scenario.tolerances.forward_rtol,
            detail=f"scalar reduction={scenario.loss.op}",
        )
    )

    try:
        for input_spec in scenario.inputs:
            if not input_spec.requires_grad:
                continue
            expected_gradient = _finite_difference_gradient(scenario, input_spec.name, reference)
            comparisons.append(
                compare_arrays(
                    kind="gradient",
                    target=f"input:{input_spec.name}",
                    expected=expected_gradient,
                    observed=candidate_execution.gradients[input_spec.name],
                    atol=scenario.tolerances.gradient_atol,
                    rtol=scenario.tolerances.gradient_rtol,
                    detail=("central finite differences; " f"base_step={scenario.tolerances.finite_difference_step:g}"),
                )
            )
        comparisons.extend(run_metamorphic_checks(scenario))
    except NumericalDomainSkip as error:
        return RunResult(
            scenario.scenario_id,
            _comparison_outcome(comparisons, otherwise=Outcome.NUMERICAL_SKIP),
            comparisons=tuple(comparisons),
            message=str(error),
        )
    except UnsupportedFeature as error:
        return RunResult(
            scenario.scenario_id,
            Outcome.UNSUPPORTED,
            comparisons=tuple(comparisons),
            message=str(error),
        )
    except (InvalidScenario, KeyError, TypeError, ValueError) as error:
        return RunResult(
            scenario.scenario_id,
            Outcome.INVALID_GRAPH,
            comparisons=tuple(comparisons),
            message=str(error),
        )
    except Exception as error:
        return RunResult(
            scenario.scenario_id,
            Outcome.CANDIDATE_ERROR,
            comparisons=tuple(comparisons),
            message=f"{type(error).__name__}: {error}",
        )

    outcome = _comparison_outcome(comparisons, otherwise=Outcome.PASS)
    return RunResult(scenario.scenario_id, outcome, comparisons=tuple(comparisons))


def run_sweep(
    *,
    seed: int,
    cases: int,
    revision: str,
    source_fingerprint: str | None = None,
    profile: str,
    candidate: CandidateExecutor | None = None,
) -> SweepResult:
    if cases <= 0:
        raise ValueError("cases must be positive")
    started = time.perf_counter()
    case_runs: list[CaseRun] = []
    rejection_count = 0
    operation_counts: Counter[str] = Counter()

    for case_index in range(cases):
        try:
            scenario = generate_scenario(
                seed=seed,
                case_index=case_index,
                revision=revision,
                source_fingerprint=source_fingerprint,
                profile=profile,
            )
        except GeneratorRejection:
            rejection_count += 1
            continue
        rejection_count += int(scenario.metadata.get("generator_rejections", 0))
        operation_counts.update(node.op for node in scenario.nodes)
        case_runs.append(CaseRun(scenario, verify_scenario(scenario, candidate=candidate)))

    return SweepResult(
        seed=seed,
        requested_cases=cases,
        revision=revision,
        source_fingerprint=source_fingerprint or revision,
        profile=profile,
        cases=tuple(case_runs),
        generator_rejections=rejection_count,
        elapsed_seconds=time.perf_counter() - started,
        operation_counts={name: operation_counts[name] for name in sorted(operation_counts)},
    )


def failure_artifact_path(scenario: Scenario, directory: Path) -> Path:
    return directory / f"{scenario.scenario_id}-{scenario.fingerprint()[:12]}.json"


def write_failure_artifact(
    *,
    scenario: Scenario,
    result: RunResult,
    directory: Path,
    replay_command: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = failure_artifact_path(scenario, directory).resolve()
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenario": scenario.to_dict(),
        "result": result.to_dict(),
        "replay": {"command": replay_command},
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return path


def failing_cases(sweep: SweepResult) -> Iterable[CaseRun]:
    return (case for case in sweep.cases if case.result.outcome not in {Outcome.PASS, Outcome.NUMERICAL_SKIP})
