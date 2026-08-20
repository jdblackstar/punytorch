from __future__ import annotations

import numpy as np

from devtools.differential_verifier.generator import FLOAT64_TOLERANCES
from devtools.differential_verifier.models import (
    InputSpec,
    LossSpec,
    NodeSpec,
    Scenario,
    ValueDomain,
)


def known_core_scenario(*, revision: str = "test-revision") -> Scenario:
    values = np.array(
        [
            [0.2, -0.4],
            [0.7, 0.1],
            [-0.3, 0.9],
            [1.0, -0.8],
        ],
        dtype=np.float64,
    )
    nodes = (
        NodeSpec("gathered", "gather", ("x",), {"indices": [1, 1, 3]}),
        NodeSpec("exponential", "exp", ("gathered",)),
        NodeSpec("logged", "log", ("exponential",)),
        NodeSpec("stable_lse", "logsumexp", ("logged",), {"axis": 1, "keepdims": False}),
    )
    return Scenario(
        scenario_id="known-core",
        seed=17,
        case_index=0,
        revision=revision,
        source_fingerprint=revision,
        inputs=(
            InputSpec(
                "x",
                values.shape,
                "float64",
                values.tolist(),
                ValueDomain("bounded", -1.5, 1.5),
            ),
        ),
        nodes=nodes,
        output="stable_lse",
        loss=LossSpec("weighted_sum", "stable_lse", [0.5, -0.25, 1.25]),
        tolerances=FLOAT64_TOLERANCES,
    )


def single_node_scenario(
    *,
    op: str,
    values,
    params: dict | None = None,
    revision: str = "test-revision",
) -> Scenario:
    array = np.asarray(values, dtype=np.float64)
    lower = min(-10.0, float(np.min(array)))
    upper = max(10.0, float(np.max(array)))
    return Scenario(
        scenario_id=f"single-{op}",
        seed=3,
        case_index=0,
        revision=revision,
        source_fingerprint=revision,
        inputs=(
            InputSpec(
                "x",
                array.shape,
                "float64",
                array.tolist(),
                ValueDomain("bounded", lower, upper),
            ),
        ),
        nodes=(NodeSpec("result", op, ("x",), params or {}),),
        output="result",
        loss=LossSpec("sum", "result"),
        tolerances=FLOAT64_TOLERANCES,
    )
