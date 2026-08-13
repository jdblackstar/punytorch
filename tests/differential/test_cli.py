from dataclasses import replace

from devtools.differential_verifier.cli import _revision, _source_fingerprint, main
from devtools.differential_verifier.generator import generate_scenario
from tests.differential.helpers import single_node_scenario


def _current_scenario():
    return generate_scenario(
        seed=44,
        case_index=0,
        revision=_revision(),
        source_fingerprint=_source_fingerprint(),
        profile="smoke",
    )


def test_replay_runs_one_saved_scenario_without_regeneration(tmp_path, capsys):
    scenario = _current_scenario()
    path = tmp_path / "scenario.json"
    path.write_text(scenario.canonical_json() + "\n", encoding="utf-8")

    exit_code = main(["replay", str(path)])

    assert exit_code == 0
    assert f"PASS replay scenario={scenario.scenario_id}" in capsys.readouterr().out


def test_replay_refuses_a_different_source_fingerprint(tmp_path, capsys):
    scenario = replace(_current_scenario(), source_fingerprint="0" * 64)
    path = tmp_path / "scenario.json"
    path.write_text(scenario.canonical_json() + "\n", encoding="utf-8")

    exit_code = main(["replay", str(path)])

    assert exit_code == 2
    assert "source state mismatch" in capsys.readouterr().err


def test_replay_returns_nonzero_for_a_numerical_skip(tmp_path, capsys):
    scenario = single_node_scenario(
        op="log",
        values=[[-1.0, 0.5]],
        revision=_revision(),
    )
    scenario = replace(scenario, source_fingerprint=_source_fingerprint())
    path = tmp_path / "scenario.json"
    path.write_text(scenario.canonical_json() + "\n", encoding="utf-8")

    exit_code = main(["replay", str(path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "outcome=numerical_skip" in output
    assert "replay:" in output
