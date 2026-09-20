from __future__ import annotations

import numpy as np

from src.revision.statistics import holm_adjust, paired_moving_block_inference


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert all(value >= source for value, source in zip(adjusted, raw))
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= -1e-12)


def test_paired_block_inference_returns_reproducible_interval():
    rng = np.random.default_rng(42)
    second = rng.normal(0.0002, 0.01, 120)
    first = second + rng.normal(0.0003, 0.001, 120)
    rf = np.full(120, 0.00001)
    first_result = paired_moving_block_inference(first, second, rf, 100, 10, 0.95, 7)
    second_result = paired_moving_block_inference(first, second, rf, 100, 10, 0.95, 7)
    assert first_result == second_result
    assert 0.0 <= first_result["p_value"] <= 1.0
