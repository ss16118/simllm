"""Acceptance calculations for qualified TRAF-94 timing rows.

The analysis operates on per-process summaries.  Treating all 500 iterations as
independent would hide process-to-process setup and scheduling variation and
would violate the frozen five-process resolution rule.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile with explicit small-sample behavior."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("a percentile requires at least one value")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _process_medians(
    rows: Sequence[Mapping[str, Any]], cell_id: str, timer: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("cell_id") == cell_id
            and row.get("timer") == timer
            and row.get("phase") == "ordinary"
            and row.get("qualification") == "qualified"
        ):
            grouped[int(row["process_id"])].append(float(row["raw_duration"]))
    return {process: statistics.median(values) for process, values in grouped.items()}


def paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_a: str,
    cell_b: str,
    timer: str,
    expected_processes: int,
) -> dict[str, Any]:
    """Apply the frozen paired-median and five-process IQR resolution rule."""

    medians_a = _process_medians(rows, cell_a, timer)
    medians_b = _process_medians(rows, cell_b, timer)
    expected = set(range(expected_processes))
    if set(medians_a) != expected or set(medians_b) != expected:
        raise ValueError("contrast does not contain every expected process in both cells")
    values_a = [medians_a[index] for index in range(expected_processes)]
    values_b = [medians_b[index] for index in range(expected_processes)]
    paired = [right - left for left, right in zip(values_a, values_b, strict=True)]
    iqr_a = _percentile(values_a, 0.75) - _percentile(values_a, 0.25)
    iqr_b = _percentile(values_b, 0.75) - _percentile(values_b, 0.25)
    delta = statistics.median(paired)
    threshold = 2.0 * (iqr_a + iqr_b)
    return {
        "cell_a": cell_a,
        "cell_b": cell_b,
        "timer": timer,
        "process_medians_a": values_a,
        "process_medians_b": values_b,
        "paired_process_deltas": paired,
        "paired_median_delta": delta,
        "iqr_a": iqr_a,
        "iqr_b": iqr_b,
        "resolution_threshold": threshold,
        "resolved": abs(delta) > threshold,
        "interpretation": "separate_term" if abs(delta) > threshold else "joint_interval",
    }


def score_confirmation(
    *, measured_delta: float, predicted_delta: float, frozen_repeat_spread_resolution: float
) -> dict[str, Any]:
    """Score a held-out delta without allowing the prediction to be refit."""

    if frozen_repeat_spread_resolution < 0:
        raise ValueError("frozen repeat-spread resolution must be non-negative")
    error = abs(predicted_delta - measured_delta)
    bound = max(0.10 * abs(measured_delta), frozen_repeat_spread_resolution)
    return {
        "measured_delta": measured_delta,
        "predicted_delta": predicted_delta,
        "absolute_error": error,
        "acceptance_bound": bound,
        "accepted": error <= bound,
    }
