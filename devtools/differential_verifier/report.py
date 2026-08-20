"""Concise, deterministic human-readable reporting."""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

from devtools.differential_verifier.models import Scenario
from devtools.differential_verifier.runner import RunResult, SweepResult


def replay_command(*, artifact_path: Path, repository_root: Path) -> str:
    cache = "/private/tmp/uv-cache-punytorch"
    return (
        f"cd {shlex.quote(str(repository_root.resolve()))} && "
        f"UV_CACHE_DIR={shlex.quote(cache)} uv run python -m "
        "devtools.differential_verifier replay "
        f"{shlex.quote(str(artifact_path.resolve()))}"
    )


def _array(value) -> str:
    return np.array2string(
        np.asarray(value),
        precision=10,
        separator=", ",
        suppress_small=False,
        threshold=48,
        max_line_width=120,
    )


def _params(params: dict) -> str:
    if not params:
        return ""
    rendered = ", ".join(f"{key}={params[key]!r}" for key in sorted(params))
    return f"; {rendered}"


def render_failure(
    scenario: Scenario,
    result: RunResult,
    *,
    replay: str,
) -> str:
    lines = [
        f"FAIL scenario={scenario.scenario_id} outcome={result.outcome.value}",
        f"seed={scenario.seed} case={scenario.case_index} revision={scenario.revision}",
        f"source_fingerprint={scenario.source_fingerprint}",
        "graph:",
    ]
    for node in scenario.nodes:
        lines.append(f"  {node.name} = {node.op}({', '.join(node.inputs)}{_params(node.params)})")
    lines.append(
        f"  loss = {scenario.loss.op}({scenario.loss.input})"
        + (" with serialized weights" if scenario.loss.weights is not None else "")
    )
    if scenario.loss.weights is not None:
        lines.append(f"    weights={_array(scenario.loss.weights)}")
    lines.append("inputs:")
    for item in scenario.inputs:
        lines.append(
            f"  {item.name}: shape={item.shape} dtype={item.dtype} "
            f"domain={item.domain.kind}[{item.domain.minimum}, {item.domain.maximum}] "
            f"min_abs={item.domain.min_abs:g}"
        )
        lines.append(f"    values={_array(item.values)}")

    failure = result.failures[0] if result.failures else None
    if failure is not None:
        lines.extend(
            [
                f"comparison={failure.kind} target={failure.target}",
                f"detail={failure.detail}",
                f"expected={_array(failure.expected)}",
                f"observed={_array(failure.observed)}",
                f"max_absolute_error={failure.max_absolute_error:.12g}",
                f"max_relative_error={failure.max_relative_error:.12g}",
                f"tolerance=atol:{failure.atol:g} rtol:{failure.rtol:g}",
                f"first_failing_index={failure.first_failing_index}",
            ]
        )
    elif result.message:
        lines.append(f"reason={result.message}")
    lines.append(f"replay: {replay}")
    return "\n".join(lines)


def render_replay_success(scenario: Scenario, result: RunResult) -> str:
    counts: dict[str, int] = {}
    for comparison in result.comparisons:
        counts[comparison.kind] = counts.get(comparison.kind, 0) + 1
    checks = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    return (
        f"PASS replay scenario={scenario.scenario_id} seed={scenario.seed} " f"case={scenario.case_index} {checks}"
    ).rstrip()


def render_sweep(sweep: SweepResult) -> str:
    outcomes = sweep.outcome_counts()
    outcome_text = " ".join(f"{key}={outcomes[key]}" for key in outcomes)
    checks = sweep.check_counts()
    check_text = " ".join(f"{key}={checks[key]}" for key in checks) or "none=0"
    operation_text = ", ".join(f"{key}:{value}" for key, value in sweep.operation_counts.items())
    status = "PASS" if sweep.successful else "FAIL"
    return "\n".join(
        [
            (
                f"{status} differential sweep seed={sweep.seed} profile={sweep.profile} "
                f"requested_cases={sweep.requested_cases} executed_cases={len(sweep.cases)}"
            ),
            f"outcomes {outcome_text} generator_rejections={sweep.generator_rejections}",
            f"checks {check_text}",
            f"operations {operation_text}",
            (
                f"revision={sweep.revision} source_fingerprint={sweep.source_fingerprint} "
                f"runtime={sweep.elapsed_seconds:.3f}s"
            ),
        ]
    )
