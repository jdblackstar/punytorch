from dataclasses import replace

from devtools.differential_verifier.cli import _revision, _source_fingerprint, main
from devtools.differential_verifier.generator import generate_scenario


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
