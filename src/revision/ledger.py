from __future__ import annotations

import hashlib
import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd

from .config import canonical_config
from .utils import (
    atomic_write_json,
    git_commit,
    git_is_dirty,
    git_worktree_fingerprint,
    hardware_snapshot,
    package_versions,
    sha256_bytes,
)


LEDGER_COLUMNS = [
    "run_id",
    "stage",
    "code_commit",
    "code_dirty",
    "code_fingerprint",
    "config_hash",
    "data_hash",
    "fold",
    "universe",
    "model",
    "seed",
    "frequency",
    "graph",
    "cost_bps",
    "status",
]


def config_hash(config: Dict[str, Any]) -> str:
    return sha256_bytes(canonical_config(config).encode("utf-8"))


def make_run_id(specification: Dict[str, Any]) -> str:
    stable = json.dumps(specification, sort_keys=True, separators=(",", ":"), default=str)
    prefix = "-".join(
        str(specification.get(key, "na"))
        for key in ("stage", "fold", "model", "seed", "frequency", "graph", "universe")
    )
    return f"{prefix}-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:12]}".lower()


def run_directory(config: Dict[str, Any], run_id: str) -> Path:
    return Path(config["paths"]["results_dir"]) / "runs" / run_id


def completed_run(
    config: Dict[str, Any],
    run_id: str,
    required: list[str] | None = None,
    data_hash: str | None = None,
) -> bool:
    directory = run_directory(config, run_id)
    manifest = directory / "run.json"
    if not manifest.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCESS":
        return False
    if not matching_run_provenance(config, run_id, data_hash=data_hash):
        return False
    return all((directory / name).exists() for name in (required or []))


def matching_run_provenance(
    config: Dict[str, Any], run_id: str, data_hash: str | None = None
) -> bool:
    manifest = run_directory(config, run_id) / "run.json"
    if not manifest.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config_hash(config):
        return False
    if data_hash is not None and payload.get("data_hash") != data_hash:
        return False
    current_commit = git_commit(config["_meta"]["project_root"])
    if current_commit != "unknown" and payload.get("code_commit") != current_commit:
        return False
    current_fingerprint = git_worktree_fingerprint(config["_meta"]["project_root"])
    if current_fingerprint != "unknown" and payload.get("code_fingerprint") != current_fingerprint:
        return False
    return True


@contextmanager
def tracked_run(
    config: Dict[str, Any],
    data_hash: str,
    specification: Dict[str, Any],
) -> Iterator[tuple[str, Path]]:
    run_id = make_run_id(specification)
    directory = run_directory(config, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    project_root = config["_meta"]["project_root"]
    base = {
        "run_id": run_id,
        **specification,
        "code_commit": git_commit(project_root),
        "code_dirty": git_is_dirty(project_root),
        "code_fingerprint": git_worktree_fingerprint(project_root),
        "config_hash": config_hash(config),
        "data_hash": data_hash,
        "command": sys.argv,
        "hardware": hardware_snapshot(),
        "packages": package_versions(["torch", "pandas", "numpy", "scipy", "scikit-learn", "pyyaml"]),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
    }
    atomic_write_json(directory / "run.json", base)
    try:
        yield run_id, directory
    except Exception as error:
        base.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(directory / "run.json", base)
        raise
    else:
        base.update({"status": "SUCCESS", "finished_at": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(directory / "run.json", base)


def build_ledger(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(results_dir).glob("runs/*/run.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        costs = payload.get("cost_bps")
        costs = costs if isinstance(costs, list) else [costs]
        for cost in costs:
            row = {column: payload.get(column) for column in LEDGER_COLUMNS}
            row["cost_bps"] = cost
            row["run_path"] = str(manifest_path.parent)
            row["started_at"] = payload.get("started_at")
            row["finished_at"] = payload.get("finished_at")
            row["error"] = payload.get("error")
            rows.append(row)
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        ledger = pd.DataFrame(columns=[*LEDGER_COLUMNS, "run_path", "started_at", "finished_at", "error"])
    output = Path(results_dir) / "ledger.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    return ledger
