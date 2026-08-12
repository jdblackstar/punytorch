from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from devtools.differential_verifier.candidate import SUPPORTED_OPERATIONS as CANDIDATE_OPERATIONS
from devtools.differential_verifier.errors import GeneratorRejection, InvalidScenario
from devtools.differential_verifier.generator import INITIAL_OPERATIONS, generate_scenario
from devtools.differential_verifier.models import NodeSpec, Scenario, scenario_from_json
from devtools.differential_verifier.reference import (
    SUPPORTED_OPERATIONS as REFERENCE_OPERATIONS,
    ReferenceEvaluator,
)


def test_generated_scenarios_are_deterministic_and_serializable():
    first = generate_scenario(seed=91, case_index=4, revision="abc123", profile="smoke")
    second = generate_scenario(seed=91, case_index=4, revision="abc123", profile="smoke")

    assert first.to_dict() == second.to_dict()
    assert first.canonical_json() == second.canonical_json()
    assert first.fingerprint() == second.fingerprint()
    assert scenario_from_json(first.canonical_json()) == first


def test_case_index_derives_independently_replayable_values():
    cases = [generate_scenario(seed=91, case_index=index, revision="abc123", profile="smoke") for index in range(3)]

    assert [case.scenario_id for case in cases] == [
        "seed-91-case-0000",
        "seed-91-case-0001",
        "seed-91-case-0002",
    ]
    assert len({case.metadata["case_seed"] for case in cases}) == 3


def test_generator_emits_only_valid_shapes_and_declared_domains():
    evaluator = ReferenceEvaluator()
    for case_index in range(24):
        scenario = generate_scenario(
            seed=302,
            case_index=case_index,
            revision="abc123",
            profile="extended",
        )
        scenario.validate_static()
        execution = evaluator.evaluate(scenario)
        assert np.asarray(scenario.loss.weights).shape == execution.output.shape
        for input_spec in scenario.inputs:
            input_spec.domain.validate(input_spec.name, input_spec.array())


def test_generator_covers_the_entire_declared_initial_slice():
    operations = {
        node.op
        for case_index in range(6)
        for node in generate_scenario(
            seed=0,
            case_index=case_index,
            revision="abc123",
            profile="smoke",
        ).nodes
    }

    assert operations == set(INITIAL_OPERATIONS)


def test_operation_surfaces_cannot_drift_between_generator_and_executors():
    expected = frozenset(INITIAL_OPERATIONS)

    assert CANDIDATE_OPERATIONS == expected
    assert frozenset(REFERENCE_OPERATIONS) == expected


def test_generator_rejects_unknown_profile_before_candidate_execution():
    with pytest.raises(GeneratorRejection, match="profile"):
        generate_scenario(seed=0, case_index=0, revision="abc123", profile="overnight")


def test_static_validation_rejects_forward_references():
    scenario = generate_scenario(seed=0, case_index=0, revision="abc123", profile="smoke")
    bad_node = NodeSpec("bad", "add", ("x", "future"))
    invalid = replace(
        scenario,
        nodes=(bad_node,),
        output="bad",
        loss=replace(scenario.loss, input="bad", weights=scenario.inputs[0].values),
    )

    with pytest.raises(InvalidScenario, match="unavailable"):
        invalid.validate_static()


def test_scenario_loader_requires_current_schema():
    scenario = generate_scenario(seed=0, case_index=0, revision="abc123", profile="smoke")
    data = scenario.to_dict()
    data["schema_version"] = 999

    with pytest.raises(InvalidScenario, match="schema"):
        Scenario.from_dict(data)
