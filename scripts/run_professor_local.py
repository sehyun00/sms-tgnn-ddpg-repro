from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "paper_revision.yaml"


def run_checked(
    arguments: Sequence[str | Path],
    *,
    cwd: Path = PROJECT_ROOT,
    environment: dict[str, str] | None = None,
) -> None:
    command = [str(value) for value in arguments]
    print("\n>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def output_checked(arguments: Sequence[str | Path], *, cwd: Path = PROJECT_ROOT) -> str:
    return subprocess.check_output(
        [str(value) for value in arguments], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_checkout() -> str:
    root = Path(output_checked(["git", "rev-parse", "--show-toplevel"])).resolve()
    if root != PROJECT_ROOT.resolve():
        raise RuntimeError(f"Run this script from its repository checkout: {PROJECT_ROOT}")
    dirty = output_checked(["git", "status", "--porcelain"])
    if dirty:
        raise RuntimeError(
            "The repository has uncommitted changes. Commit or remove them before a reportable run.\n"
            + dirty
        )
    return output_checked(["git", "rev-parse", "HEAD"])


def verify_cuda(python: Path) -> None:
    probe = (
        "import json, torch; "
        "assert torch.cuda.is_available(), "
        "'CUDA is unavailable. Install an NVIDIA driver and CUDA-enabled PyTorch.'; "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0)}))"
    )
    run_checked([python, "-c", probe])


def experiment_environment(args: argparse.Namespace) -> tuple[dict[str, str], Path, Path]:
    environment = os.environ.copy()
    prepared_dir = (
        Path(args.prepared_dir).expanduser().resolve()
        if args.prepared_dir
        else (PROJECT_ROOT / "data" / "revision" / "paper-v1").resolve()
    )
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else (PROJECT_ROOT / "results" / "revision" / "paper-v1").resolve()
    )
    environment["SMS_PREPARED_DIR"] = str(prepared_dir)
    environment["SMS_RESULTS_DIR"] = str(results_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return environment, prepared_dir, results_dir


def revision_cli(python: Path, environment: dict[str, str], arguments: Sequence[str]) -> None:
    run_checked(
        [python, "scripts/run_revision_experiments.py", "--config", CONFIG_PATH, *arguments],
        environment=environment,
    )


def prepare_or_verify(python: Path, environment: dict[str, str], prepared_dir: Path) -> None:
    required = [
        prepared_dir / "panel.csv.gz",
        prepared_dir / "fama_french_daily.csv",
        prepared_dir / "universe_manifest.json",
        prepared_dir / "data_manifest.json",
    ]
    if not all(path.exists() for path in required):
        revision_cli(python, environment, ["--stage", "prepare"])
        return
    verification = (
        "from src.revision.config import load_revision_config; "
        "from src.revision.data import load_prepared; "
        "c=load_revision_config(r'config/paper_revision.yaml'); "
        "print('Prepared data verified:', load_prepared(c)[3]['data_hash'])"
    )
    run_checked([python, "-c", verification], environment=environment)


def run_primary(
    python: Path, environment: dict[str, str], primary_seeds: str
) -> None:
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "train",
            "--fold",
            "primary",
            "--model",
            "all",
            "--seed",
            primary_seeds,
            "--frequency",
            "all",
            "--graph",
            "primary",
            "--resume",
        ],
    )
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "backtest",
            "--fold",
            "primary",
            "--model",
            "all",
            "--seed",
            primary_seeds,
            "--frequency",
            "all",
            "--graph",
            "primary",
            "--universe",
            "n10",
            "--resume",
        ],
    )


def run_full_extensions(
    python: Path, environment: dict[str, str], primary_seeds: str, secondary_seeds: str
) -> None:
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "train",
            "--fold",
            "primary",
            "--model",
            "tgnn,hybrid",
            "--seed",
            primary_seeds,
            "--frequency",
            "all",
            "--graph",
            "sector,correlation",
            "--resume",
        ],
    )
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "backtest",
            "--fold",
            "primary",
            "--model",
            "tgnn,hybrid,hybrid_fixed",
            "--seed",
            primary_seeds,
            "--frequency",
            "quarterly",
            "--graph",
            "all",
            "--universe",
            "n10",
            "--resume",
        ],
    )
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "train",
            "--fold",
            "secondary",
            "--model",
            "all",
            "--seed",
            secondary_seeds,
            "--frequency",
            "all",
            "--graph",
            "primary",
            "--resume",
        ],
    )
    revision_cli(
        python,
        environment,
        [
            "--stage",
            "backtest",
            "--fold",
            "secondary",
            "--model",
            "all",
            "--seed",
            secondary_seeds,
            "--frequency",
            "all",
            "--graph",
            "primary",
            "--universe",
            "n10",
            "--resume",
        ],
    )
    for universe in ("n5", "n15"):
        revision_cli(
            python,
            environment,
            [
                "--stage",
                "backtest",
                "--fold",
                "primary",
                "--model",
                "all",
                "--seed",
                "42",
                "--frequency",
                "quarterly",
                "--graph",
                "primary",
                "--universe",
                universe,
                "--resume",
            ],
        )
    revision_cli(python, environment, ["--stage", "aggregate"])


def verify_full_readiness(results_dir: Path) -> None:
    readiness_path = results_dir / "aggregate" / "evidence_readiness.json"
    if not readiness_path.exists():
        raise RuntimeError(f"Aggregate readiness report is missing: {readiness_path}")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("status") != "READY":
        raise RuntimeError(
            "The complete experiment matrix is not evidence-ready. "
            f"Inspect {readiness_path} and resume the same command."
        )


def package_results(
    results_dir: Path, prepared_dir: Path, commit: str, output_path: Path
) -> dict[str, object]:
    manifests = sorted(results_dir.glob("runs/*/run.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    successful = [
        (manifest, payload)
        for manifest, payload in zip(manifests, payloads)
        if payload.get("status") == "SUCCESS"
    ]
    unexpected_commits = sorted(
        {str(payload.get("code_commit")) for _, payload in successful if payload.get("code_commit") != commit}
    )
    if unexpected_commits:
        raise RuntimeError(f"Successful runs contain different code commits: {unexpected_commits}")
    data_hashes = sorted({str(payload.get("data_hash")) for _, payload in successful})
    if len(data_hashes) > 1:
        raise RuntimeError(f"Successful runs contain different data hashes: {data_hashes}")

    status_counts = {
        status: sum(payload.get("status") == status for payload in payloads)
        for status in ("SUCCESS", "RUNNING", "FAILED")
    }
    summary: dict[str, object] = {
        "code_commit": commit,
        "data_hashes": data_hashes,
        "status_counts": status_counts,
        "successful_run_ids": [payload["run_id"] for _, payload in successful],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        archive.write(summary_path, arcname=summary_path.name)
        for name in ("data_manifest.json", "universe_manifest.json"):
            source = prepared_dir / name
            if source.exists():
                archive.write(source, arcname=f"prepared/{name}")
        for manifest, _ in successful:
            run_dir = manifest.parent
            for source in sorted(
                path for path in run_dir.rglob("*") if path.is_file() and path.name != "latest.pt"
            ):
                relative = Path("results/runs") / run_dir.name / source.relative_to(run_dir)
                archive.write(source, arcname=str(relative))
        aggregate = results_dir / "aggregate"
        if aggregate.exists():
            for source in sorted(path for path in aggregate.rglob("*") if path.is_file()):
                archive.write(source, arcname=str(Path("results/aggregate") / source.relative_to(aggregate)))
    temporary.replace(output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SMS paper-revision pipeline on a local CUDA workstation."
    )
    parser.add_argument("--mode", choices=("core", "full"), default="full")
    parser.add_argument("--primary-seeds", default="all")
    parser.add_argument("--secondary-seeds", default="all")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "full" and (args.primary_seeds != "all" or args.secondary_seeds != "all"):
        raise ValueError("Full mode requires all configured primary and secondary seeds")

    commit = validate_checkout()
    python = Path(sys.executable).resolve()
    environment, prepared_dir, results_dir = experiment_environment(args)
    print(json.dumps({"code_commit": commit, "python": str(python), "mode": args.mode}, indent=2))
    verify_cuda(python)
    if not args.skip_tests:
        run_checked([python, "-m", "pytest", "-q"], environment=environment)
    prepare_or_verify(python, environment, prepared_dir)
    run_primary(python, environment, args.primary_seeds)
    if args.mode == "full":
        run_full_extensions(python, environment, args.primary_seeds, args.secondary_seeds)
        verify_full_readiness(results_dir)

    if not args.no_package:
        output_path = results_dir.parent / f"{results_dir.name}-professor-handoff.zip"
        summary = package_results(results_dir, prepared_dir, commit, output_path)
        print(json.dumps(summary["status_counts"], indent=2))
        print(f"Professor handoff: {output_path}")


if __name__ == "__main__":
    main()
