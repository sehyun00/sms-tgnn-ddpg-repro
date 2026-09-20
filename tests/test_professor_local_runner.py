import json
import zipfile

import pytest

from scripts.run_professor_local import build_parser, package_results, verify_full_readiness


def test_professor_runner_defaults_to_complete_matrix():
    args = build_parser().parse_args([])
    assert args.mode == "full"
    assert args.primary_seeds == "all"
    assert args.secondary_seeds == "all"


def test_handoff_package_includes_success_and_excludes_resume_checkpoint(tmp_path):
    commit = "a" * 40
    prepared = tmp_path / "prepared"
    results = tmp_path / "results"
    run_dir = results / "runs" / "run-one"
    aggregate = results / "aggregate"
    prepared.mkdir()
    run_dir.mkdir(parents=True)
    aggregate.mkdir(parents=True)
    (prepared / "data_manifest.json").write_text("{}", encoding="utf-8")
    (prepared / "universe_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-one",
                "status": "SUCCESS",
                "code_commit": commit,
                "data_hash": "data-one",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.pt").write_bytes(b"final")
    (run_dir / "latest.pt").write_bytes(b"resume")
    (aggregate / "primary_table.csv").write_text("model,sharpe\n", encoding="utf-8")
    output = tmp_path / "handoff.zip"

    summary = package_results(results, prepared, commit, output)

    assert summary["status_counts"]["SUCCESS"] == 1
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "results/runs/run-one/checkpoint.pt" in names
    assert "results/runs/run-one/latest.pt" not in names
    assert "results/aggregate/primary_table.csv" in names
    assert "prepared/data_manifest.json" in names


def test_handoff_package_rejects_mixed_commits(tmp_path):
    results = tmp_path / "results"
    run_dir = results / "runs" / "run-one"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-one",
                "status": "SUCCESS",
                "code_commit": "b" * 40,
                "data_hash": "data-one",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different code commits"):
        package_results(results, tmp_path / "prepared", "a" * 40, tmp_path / "handoff.zip")


def test_full_readiness_requires_ready_status(tmp_path):
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    readiness = aggregate / "evidence_readiness.json"
    readiness.write_text(json.dumps({"status": "INCOMPLETE"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not evidence-ready"):
        verify_full_readiness(tmp_path)

    readiness.write_text(json.dumps({"status": "READY"}), encoding="utf-8")
    verify_full_readiness(tmp_path)
