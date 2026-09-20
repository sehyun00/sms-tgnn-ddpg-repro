from __future__ import annotations

import copy
import json

import pytest

from src.revision.config import load_revision_config
from src.revision.ledger import build_ledger, completed_run, make_run_id, run_directory, tracked_run


def test_failure_is_persisted_and_never_marked_complete(tmp_path):
    config = copy.deepcopy(load_revision_config("config/revision_smoke.yaml"))
    config["paths"]["results_dir"] = str(tmp_path)
    specification = {
        "stage": "backtest",
        "fold": "primary",
        "universe": "n10",
        "model": "tgnn",
        "seed": 42,
        "frequency": "monthly",
        "graph": "combined",
        "cost_bps": [0.0, 10.0],
    }
    run_id = make_run_id(specification)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        with tracked_run(config, "data-hash", specification):
            raise RuntimeError("deliberate failure")
    payload = json.loads((run_directory(config, run_id) / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["error_type"] == "RuntimeError"
    assert not completed_run(config, run_id, data_hash="data-hash")
    ledger = build_ledger(tmp_path)
    assert ledger.loc[0, "code_fingerprint"] == payload["code_fingerprint"]
    assert bool(ledger.loc[0, "code_dirty"]) == bool(payload["code_dirty"])
