"""Command-line entry point for differential sweeps and exact replay."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from devtools.differential_verifier.errors import InvalidScenario, UnsupportedFeature
from devtools.differential_verifier.models import load_scenario
from devtools.differential_verifier.report import (
    render_failure,
    render_replay_success,
    render_sweep,
    replay_command,
)
from devtools.differential_verifier.runner import (
    Outcome,
    failing_cases,
    failure_artifact_path,
    run_sweep,
    verify_scenario,
    write_failure_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        *REPOSITORY_ROOT.glob("punytorch/**/*.py"),
        *REPOSITORY_ROOT.glob("devtools/**/*.py"),
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "uv.lock",
    ]
    for path in sorted(paths):
        relative = path.relative_to(REPOSITORY_ROOT)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devtools.differential_verifier",
        description="Generate or replay deterministic PunyTorch differential scenarios.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a deterministic generated sweep")
    run.add_argument("--seed", type=int, required=True, help="master seed")
    run.add_argument("--cases", type=int, required=True, help="number of independently replayable cases")
    run.add_argument("--profile", choices=("smoke", "extended"), default="smoke")
    run.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".punytorch-differential-failures"),
        help="directory for stable JSON failure artifacts",
    )

    replay = subparsers.add_parser("replay", help="replay one saved scenario or failure artifact")
    replay.add_argument("artifact", type=Path)
    replay.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="run even when the saved revision or source fingerprint differs from the current checkout",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    revision = _revision()
    source_fingerprint = _source_fingerprint()
    sweep = run_sweep(
        seed=args.seed,
        cases=args.cases,
        revision=revision,
        source_fingerprint=source_fingerprint,
        profile=args.profile,
    )
    print(render_sweep(sweep))
    for case in failing_cases(sweep):
        artifact = failure_artifact_path(case.scenario, args.artifacts).resolve()
        command = replay_command(artifact_path=artifact, repository_root=REPOSITORY_ROOT)
        artifact = write_failure_artifact(
            scenario=case.scenario,
            result=case.result,
            directory=args.artifacts,
            replay_command=command,
        )
        print()
        print(render_failure(case.scenario, case.result, replay=command))
        print(f"artifact: {artifact}")
    return 0 if sweep.successful else 1


def _replay(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.artifact)
    except InvalidScenario as error:
        print(f"invalid replay artifact: {error}", file=sys.stderr)
        return 2
    except UnsupportedFeature as error:
        print(f"unsupported replay artifact: {error}", file=sys.stderr)
        return 2
    revision = _revision()
    source_fingerprint = _source_fingerprint()
    revision_matches = scenario.revision == revision
    source_matches = scenario.source_fingerprint == source_fingerprint
    if (not revision_matches or not source_matches) and not args.allow_revision_mismatch:
        print(
            "source state mismatch: "
            f"scenario requires revision={scenario.revision} "
            f"source_fingerprint={scenario.source_fingerprint}, "
            f"current checkout has revision={revision} source_fingerprint={source_fingerprint}; "
            "restore the saved source state or pass --allow-revision-mismatch",
            file=sys.stderr,
        )
        return 2

    result = verify_scenario(scenario)
    if result.outcome == Outcome.PASS:
        print(render_replay_success(scenario, result))
        return 0
    command = replay_command(artifact_path=args.artifact, repository_root=REPOSITORY_ROOT)
    print(render_failure(scenario, result, replay=command))
    return 0 if result.outcome == Outcome.NUMERICAL_SKIP else 1


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.cases <= 0:
            parser.error("--cases must be positive")
        return _run(args)
    return _replay(args)
