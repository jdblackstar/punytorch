from __future__ import annotations

import numpy as np
import pytest

from devtools.differential_verifier.generator import generate_scenario
from devtools.differential_verifier.reference import ReferenceEvaluator
from devtools.differential_verifier.runner import Outcome, verify_scenario
from tests.differential.helpers import known_core_scenario, single_node_scenario


@pytest.mark.parametrize(
    ("op", "values", "expected"),
    [
        ("exp", [[-0.5, 0.25]], np.exp([[-0.5, 0.25]])),
        ("log", [[0.5, 2.0]], np.log([[0.5, 2.0]])),
        ("tanh", [[-0.5, 0.25]], np.tanh([[-0.5, 0.25]])),
        ("sigmoid", [[-0.5, 0.25]], 1.0 / (1.0 + np.exp(-np.array([[-0.5, 0.25]])))),
    ],
)
def test_reference_unary_evaluation(op, values, expected):
    scenario = single_node_scenario(op=op, values=values)

    execution = ReferenceEvaluator().evaluate(scenario)

    np.testing.assert_allclose(execution.output, expected)


def test_reference_logsumexp_uses_stable_shifted_formula():
    scenario = single_node_scenario(
        op="logsumexp",
        values=[[1000.0, 999.0], [-1000.0, -999.0]],
        params={"axis": 1, "keepdims": False},
    )
    expected = np.array(
        [
            1000.0 + np.log1p(np.exp(-1.0)),
            -999.0 + np.log1p(np.exp(-1.0)),
        ]
    )

    execution = ReferenceEvaluator().evaluate(scenario)

    np.testing.assert_allclose(execution.output, expected)
    assert np.all(np.isfinite(execution.output))


def test_tuple_axis_reduction_is_classified_as_unsupported():
    scenario = single_node_scenario(
        op="logsumexp",
        values=[[[1.0, 2.0]], [[3.0, 4.0]]],
        params={"axis": [0, 2], "keepdims": False},
    )

    result = verify_scenario(scenario)

    assert result.outcome == Outcome.UNSUPPORTED
    assert "tuple-axis reductions" in result.message


def test_known_core_graph_checks_exp_log_logsumexp_and_repeated_gather():
    scenario = known_core_scenario()

    result = verify_scenario(scenario)

    assert result.outcome == Outcome.PASS
    assert [comparison.kind for comparison in result.comparisons] == [
        "forward",
        "forward",
        "forward",
        "forward",
        "forward",
        "gradient",
    ]


@pytest.mark.parametrize("case_index", range(6))
def test_each_generated_family_runs_through_public_tensor_autograd(case_index):
    scenario = generate_scenario(
        seed=2026,
        case_index=case_index,
        revision="test-revision",
        profile="smoke",
    )

    result = verify_scenario(scenario)

    assert result.outcome == Outcome.PASS
    assert all(comparison.passed for comparison in result.comparisons)


def test_repeated_index_metamorphic_check_uses_an_independent_count_oracle():
    scenario = generate_scenario(
        seed=2026,
        case_index=4,
        revision="test-revision",
        profile="smoke",
    )

    result = verify_scenario(scenario)

    repeated = next(item for item in result.comparisons if item.target == "repeated_index_gradient")
    assert repeated.kind == "metamorphic"
    assert repeated.passed
    observed = np.asarray(repeated.observed)
    assert np.max(observed) == 2.0
