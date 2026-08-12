from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from devtools.differential_verifier.candidate import CandidateExecutor
from devtools.differential_verifier.generator import generate_scenario
from devtools.differential_verifier.models import LossSpec, NodeSpec, load_scenario
from devtools.differential_verifier.report import render_failure, replay_command
from devtools.differential_verifier.runner import (
    Outcome,
    failure_artifact_path,
    run_sweep,
    verify_scenario,
    write_failure_artifact,
)
from tests.differential.helpers import known_core_scenario, single_node_scenario


def test_forward_self_test_detects_corrupted_observation_at_intended_node():
    scenario = known_core_scenario()
    candidate = CandidateExecutor(observed_corruptor=lambda name, value: value + 0.25 if name == "logged" else value)

    result = verify_scenario(scenario, candidate=candidate)

    assert result.outcome == Outcome.MISMATCH
    assert [failure.target for failure in result.failures] == ["node:logged"]
    assert result.failures[0].kind == "forward"
    assert result.failures[0].max_absolute_error == 0.25


def test_gradient_self_test_detects_corrupted_gradient_at_intended_input():
    scenario = known_core_scenario()
    candidate = CandidateExecutor(gradient_corruptor=lambda name, value: value + 0.125)

    result = verify_scenario(scenario, candidate=candidate)

    assert result.outcome == Outcome.MISMATCH
    assert [failure.target for failure in result.failures] == ["input:x"]
    assert result.failures[0].kind == "gradient"
    assert np.isclose(result.failures[0].max_absolute_error, 0.125, atol=1e-10)


def test_invalid_unsupported_and_numerical_skip_are_distinct_outcomes():
    valid = known_core_scenario()
    invalid = replace(
        valid,
        nodes=(NodeSpec("bad", "add", ("x", "missing")),),
        output="bad",
        loss=LossSpec("sum", "bad"),
    )
    unsupported = replace(
        valid,
        nodes=(NodeSpec("mystery", "fft", ("x",)),),
        output="mystery",
        loss=LossSpec("sum", "mystery"),
    )
    numerical = single_node_scenario(op="log", values=[[-1.0, 0.5]])

    assert verify_scenario(invalid).outcome == Outcome.INVALID_GRAPH
    assert verify_scenario(unsupported).outcome == Outcome.UNSUPPORTED
    assert verify_scenario(numerical).outcome == Outcome.NUMERICAL_SKIP


def test_report_and_result_order_are_deterministic():
    scenario = generate_scenario(
        seed=77,
        case_index=5,
        revision="test-revision",
        profile="smoke",
    )
    first = verify_scenario(scenario)
    second = verify_scenario(scenario)

    assert first.to_dict() == second.to_dict()
    assert [comparison.target for comparison in first.comparisons] == [
        *(f"node:{node.name}" for node in scenario.nodes),
        "loss",
        *(f"input:{item.name}" for item in scenario.inputs if item.requires_grad),
        *(check.kind for check in scenario.metamorphic_checks),
    ]


def test_failure_artifact_is_stable_semantic_json_and_loads_for_replay(tmp_path):
    scenario = known_core_scenario()
    result = verify_scenario(
        scenario,
        candidate=CandidateExecutor(gradient_corruptor=lambda name, value: value + 0.125),
    )
    artifact = failure_artifact_path(scenario, tmp_path)
    command = replay_command(artifact_path=artifact, repository_root=tmp_path)

    first_path = write_failure_artifact(
        scenario=scenario,
        result=result,
        directory=tmp_path,
        replay_command=command,
    )
    first_bytes = first_path.read_bytes()
    second_path = write_failure_artifact(
        scenario=scenario,
        result=result,
        directory=tmp_path,
        replay_command=command,
    )

    assert second_path.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert payload["artifact_schema_version"] == 1
    assert payload["result"]["outcome"] == "mismatch"
    assert payload["result"]["comparisons"][-1]["target"] == "input:x"
    assert load_scenario(first_path) == scenario
    assert str(first_path.resolve()) in payload["replay"]["command"]


def test_failure_artifact_serializes_nonfinite_candidate_evidence(tmp_path):
    scenario = known_core_scenario()
    result = verify_scenario(
        scenario,
        candidate=CandidateExecutor(
            observed_corruptor=lambda name, value: np.full_like(value, np.nan) if name == "gathered" else value
        ),
    )

    path = write_failure_artifact(
        scenario=scenario,
        result=result,
        directory=tmp_path,
        replay_command="replay command",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["result"]["comparisons"][0]["observed"][0][0] == "NaN"
    assert payload["result"]["comparisons"][0]["max_absolute_error"] is None


def test_failure_report_contains_replay_and_numerical_evidence(tmp_path):
    scenario = known_core_scenario()
    result = verify_scenario(
        scenario,
        candidate=CandidateExecutor(observed_corruptor=lambda name, value: value + 1.0),
    )
    command = replay_command(artifact_path=tmp_path / "failure.json", repository_root=tmp_path)

    report = render_failure(scenario, result, replay=command)

    assert "graph:" in report
    assert "shape=(4, 2) dtype=float64 domain=bounded" in report
    assert "comparison=forward target=node:gathered" in report
    assert "expected=" in report
    assert "observed=" in report
    assert "max_absolute_error=1" in report
    assert "max_relative_error=" in report
    assert "tolerance=atol:1e-10 rtol:1e-10" in report
    assert "first_failing_index=(0, 0)" in report
    assert f"replay: {command}" in report


def test_smoke_sweep_reports_coverage_and_no_rejections():
    sweep = run_sweep(
        seed=0,
        cases=12,
        revision="test-revision",
        profile="smoke",
    )

    assert sweep.successful
    assert sweep.generator_rejections == 0
    assert sweep.outcome_counts()["pass"] == 12
    assert sweep.check_counts() == {"forward": 66, "gradient": 20, "metamorphic": 8}
    assert set(sweep.operation_counts) == {
        "add",
        "cat",
        "div",
        "exp",
        "gather",
        "index",
        "log",
        "logsumexp",
        "mean",
        "mul",
        "reshape",
        "sigmoid",
        "stack",
        "sub",
        "sum",
        "tanh",
        "transpose",
    }
