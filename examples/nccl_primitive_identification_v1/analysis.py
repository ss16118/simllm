"""Acceptance calculations for qualified TRAF-94 timing rows.

The analysis operates on per-process summaries.  Treating all 500 iterations as
independent would hide process-to-process setup and scheduling variation and
would violate the frozen five-process resolution rule.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from examples.nccl_primitive_identification_v1.matrix import (
    content_digest,
    timing_scope_voids,
)

FIT_SCHEMA = "simllm-nccl-primitive-identification-fit-v1"
SCORE_SCHEMA = "simllm-nccl-primitive-confirmation-score-v1"
_UNIT_TO_NS = {"ns": 1.0, "us": 1_000.0, "ps": 0.001}

# Every downstream statistic starts from one median per OS process, cell, and
# timer.  The campaign contains more than one million repeat rows, so looking
# up those medians by rescanning the complete row list for every anchor makes a
# full H100 fit needlessly superlinear.  This immutable-by-convention index is
# the shared representation used by the fit and scorer; the public helpers can
# still build a single entry directly for small callers and unit tests.
_ProcessMedianIndex = Mapping[tuple[str, str], Mapping[int, float]]


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
    rows: Sequence[Mapping[str, Any]],
    cell_id: str,
    timer: str,
    *,
    process_medians: _ProcessMedianIndex | None = None,
) -> dict[int, float]:
    if process_medians is not None:
        return dict(process_medians.get((cell_id, timer), {}))

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


def _process_median_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[int, float]]:
    """Reduce raw repeats once for all later cell/timer lookups.

    Index construction intentionally applies the same row predicates and
    ``float`` conversion as :func:`_process_medians`.  It is therefore a pure
    execution optimization: process medians, paired deltas, IQR thresholds,
    and the resulting content digests remain byte-for-byte deterministic.
    """

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "ordinary" or row.get("qualification") != "qualified":
            continue
        key = (str(row["cell_id"]), str(row["timer"]), int(row["process_id"]))
        grouped[key].append(float(row["raw_duration"]))

    index: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (cell_id, timer, process), values in grouped.items():
        index[(cell_id, timer)][process] = statistics.median(values)
    return dict(index)


def paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_a: str,
    cell_b: str,
    timer: str,
    expected_processes: int,
    process_medians: _ProcessMedianIndex | None = None,
) -> dict[str, Any]:
    """Apply the frozen paired-median and five-process IQR resolution rule."""

    medians_a = _process_medians(
        rows, cell_a, timer, process_medians=process_medians
    )
    medians_b = _process_medians(
        rows, cell_b, timer, process_medians=process_medians
    )
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


def _rows_in_nanoseconds(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy qualified ordinary rows and normalize every declared timer to ns."""

    normalized = []
    for row in rows:
        if row.get("phase") != "ordinary" or row.get("qualification") != "qualified":
            continue
        unit = str(row.get("units"))
        if unit not in _UNIT_TO_NS:
            raise ValueError(f"analysis cannot normalize timer unit {unit!r}")
        value = dict(row)
        value["raw_duration"] = float(row["raw_duration"]) * _UNIT_TO_NS[unit]
        value["units"] = "ns"
        normalized.append(value)
    return normalized


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
    timer: str,
    expected_processes: int,
    process_medians: _ProcessMedianIndex | None = None,
) -> dict[str, Any]:
    """Summarize one cell from independent process medians, never raw repeats."""

    medians = _process_medians(
        rows, cell_id, timer, process_medians=process_medians
    )
    expected = set(range(expected_processes))
    if set(medians) != expected:
        raise ValueError(
            f"cell {cell_id!r}/{timer!r} does not contain all expected processes"
        )
    values = [medians[index] for index in range(expected_processes)]
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    return {
        "process_medians_ns": values,
        "median_ns": statistics.median(values),
        "q1_ns": q1,
        "q3_ns": q3,
        "iqr_ns": q3 - q1,
    }


def _request_key(cell: Mapping[str, Any], *, omit: frozenset[str]) -> str:
    """Canonical matched-cell identity after removing named interventions."""

    request = {
        name: value
        for name, value in cell["requested"].items()
        if name not in omit
    }
    return content_digest(
        {
            "protocol": cell["protocol"],
            "simple_placement": cell["simple_placement"],
            "request": request,
        }
    )


def _level_order(value: Any) -> tuple[int, float | str]:
    """Give numeric controls their natural order and place ``full`` last."""

    if value == "full":
        return (2, "full")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _paired_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_a: Mapping[str, Any],
    cell_b: Mapping[str, Any],
    timer: str,
    expected_processes: int,
    factor: str,
    level_a: Any,
    level_b: Any,
    process_medians: _ProcessMedianIndex | None = None,
) -> dict[str, Any]:
    contrast = paired_contrast(
        rows,
        cell_a=str(cell_a["cell_id"]),
        cell_b=str(cell_b["cell_id"]),
        timer=timer,
        expected_processes=expected_processes,
        process_medians=process_medians,
    )
    return {
        "stage": cell_a["stage"],
        "timer": timer,
        "units": "ns",
        "protocol": cell_a["protocol"],
        "simple_placement": cell_a["simple_placement"],
        "factor": factor,
        "level_a": level_a,
        "level_b": level_b,
        **contrast,
    }


def _matched_family_contrasts(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    timer: str,
    expected_processes: int,
    family_a: str,
    family_b: str,
    factor: str,
    process_medians: _ProcessMedianIndex | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for cell in cells:
        groups[_request_key(cell, omit=frozenset())][str(cell["family"])] = cell
    records = []
    for group in groups.values():
        if family_a not in group or family_b not in group:
            continue
        records.append(
            _paired_record(
                rows,
                cell_a=group[family_a],
                cell_b=group[family_b],
                timer=timer,
                expected_processes=expected_processes,
                factor=factor,
                level_a=family_a,
                level_b=family_b,
                process_medians=process_medians,
            )
        )
    return records


def _axis_contrasts(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    timer: str,
    expected_processes: int,
    axis: str,
    process_medians: _ProcessMedianIndex | None = None,
) -> list[dict[str, Any]]:
    """Compare adjacent levels while holding every other request field fixed."""

    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        key = (
            str(cell["protocol"]),
            str(cell["simple_placement"]),
            str(cell["family"]),
            _request_key(cell, omit=frozenset({axis})),
        )
        groups[key].append(cell)
    records = []
    for group in groups.values():
        ordered = sorted(group, key=lambda cell: _level_order(cell["requested"][axis]))
        for cell_a, cell_b in pairwise(ordered):
            records.append(
                _paired_record(
                    rows,
                    cell_a=cell_a,
                    cell_b=cell_b,
                    timer=timer,
                    expected_processes=expected_processes,
                    factor=axis,
                    level_a=cell_a["requested"][axis],
                    level_b=cell_b["requested"][axis],
                    process_medians=process_medians,
                )
            )
    return records


def fit_identification(
    rows: Sequence[Mapping[str, Any]],
    *,
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    observations_sha256: str,
) -> dict[str, Any]:
    """Fit an identification-only component profile and freeze its digest.

    The fit is deliberately nonparametric.  Cell medians are the identified
    service anchors; matched interventions are either resolved separate terms
    or named joint intervals under the frozen paired-IQR rule.  This avoids
    assigning physical meaning to a convenient but rank-deficient regression.
    Confirmation rows are never used by this function.
    """

    expected_processes = int(manifest["ordinary_processes_per_cell"])
    normalized = _rows_in_nanoseconds(rows)
    void_scopes = timing_scope_voids(rows)
    void_keys = {(row["cell_id"], row["timer"]) for row in void_scopes}
    identification_cells = [
        cell for cell in inventory["cells"] if cell["stage"] != "confirmation"
    ]
    identification_ids = {cell["cell_id"] for cell in identification_cells}
    fit_rows = [row for row in normalized if row["cell_id"] in identification_ids]

    # All anchors and matched contrasts consume the same process medians.
    # Materializing them once prevents every cell from rescanning the full
    # campaign while preserving the frozen process-first aggregation rule.
    process_medians = _process_median_index(fit_rows)

    anchors = []
    contrasts = []
    for timer in manifest["timing_boundaries"]:
        available_cells = [
            cell
            for cell in identification_cells
            if (cell["cell_id"], timer) not in void_keys
        ]
        for cell in available_cells:
            anchors.append(
                {
                    "cell_id": cell["cell_id"],
                    "stage": cell["stage"],
                    "timer": timer,
                    "units": "ns",
                    "protocol": cell["protocol"],
                    "simple_placement": cell["simple_placement"],
                    "family": cell["family"],
                    "requested": cell["requested"],
                    "summary": _summary(
                        fit_rows,
                        cell_id=cell["cell_id"],
                        timer=timer,
                        expected_processes=expected_processes,
                        process_medians=process_medians,
                    ),
                }
            )

        by_stage = {
            stage: [cell for cell in available_cells if cell["stage"] == stage]
            for stage in ("ready_publication", "reuse", "data_work", "sharing")
        }
        contrasts.extend(
            _matched_family_contrasts(
                fit_rows,
                by_stage["ready_publication"],
                timer=timer,
                expected_processes=expected_processes,
                family_a="already_ready",
                family_b="delayed_publication",
                factor="publication_state",
                process_medians=process_medians,
            )
        )
        for stage in ("data_work", "sharing"):
            contrasts.extend(
                _matched_family_contrasts(
                    fit_rows,
                    by_stage[stage],
                    timer=timer,
                    expected_processes=expected_processes,
                    family_a="copy",
                    family_b="sum",
                    factor="reduction",
                    process_medians=process_medians,
                )
            )
        contrasts.extend(
            _axis_contrasts(
                fit_rows,
                by_stage["data_work"],
                timer=timer,
                expected_processes=expected_processes,
                axis="working_set",
                process_medians=process_medians,
            )
        )
        for axis in ("delay_cycles", "reservations"):
            contrasts.extend(
                _axis_contrasts(
                    fit_rows,
                    by_stage["reuse"],
                    timer=timer,
                    expected_processes=expected_processes,
                    axis=axis,
                    process_medians=process_medians,
                )
            )
        for axis in ("active_channels", "available_sms", "working_warps"):
            contrasts.extend(
                _axis_contrasts(
                    fit_rows,
                    by_stage["sharing"],
                    timer=timer,
                    expected_processes=expected_processes,
                    axis=axis,
                    process_medians=process_medians,
                )
            )

    publication_samples: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for contrast in contrasts:
        if contrast["factor"] != "publication_state":
            continue
        cell = next(
            cell for cell in identification_cells if cell["cell_id"] == contrast["cell_a"]
        )
        # Confirmation payloads are all nonempty.  Pool only the matching
        # nonempty identification contrasts so empty-path control work cannot
        # masquerade as payload publication sensitivity.
        if int(cell["requested"].get("useful_bytes", 0)) <= 0:
            continue
        key = (
            contrast["timer"],
            contrast["protocol"],
            contrast["simple_placement"],
            int(cell["requested"]["delay_cycles"]),
        )
        publication_samples[key].append(float(contrast["paired_median_delta"]))

    publication_response = []
    for (timer, protocol, placement, delay), values in sorted(publication_samples.items()):
        publication_response.append(
            {
                "timer": timer,
                "protocol": protocol,
                "simple_placement": placement,
                "delay_cycles": delay,
                "payload_contrast_count": len(values),
                "payload_contrast_deltas_ns": values,
                "median_delta_ns": statistics.median(values),
                "q1_delta_ns": _percentile(values, 0.25),
                "q3_delta_ns": _percentile(values, 0.75),
            }
        )

    resolved = sum(bool(row["resolved"]) for row in contrasts)
    fit = {
        "schema": FIT_SCHEMA,
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "observations_sha256": observations_sha256,
        "units": "ns",
        "fit_scope": "identification_cells_only",
        "confirmation_rows_used": 0,
        "void_scopes": list(void_scopes),
        "anchors": anchors,
        "paired_contrasts": contrasts,
        "publication_response": publication_response,
        "separability": {
            "contrast_count": len(contrasts),
            "resolved_separate_terms": resolved,
            "joint_intervals": len(contrasts) - resolved,
            "rule": manifest["resolved_contrast"],
        },
        "confirmation_model": {
            "kind": "source_constrained_additive_interpolation_v1",
            "data_curve": "sum/reused data_work anchors, linear in useful bytes",
            "sharing_curve": "sum sharing anchors at 1920 bytes/channel, linear in channels",
            "payload_transfer": "C * (data(bytes_per_channel) - data(1920))",
            "rank_transfer": "multiply no-delay rank-2 collective service by ranks-1",
            "publication": "add the pooled nonempty delayed-minus-ready response once",
            "uncertainty": (
                "propagate anchor IQRs with absolute interpolation/extrapolation "
                "weights, add component bounds, and apply the source rank multiplier"
            ),
            "clamping": "none",
        },
    }
    fit["fit_digest"] = content_digest(fit)
    return fit


def _linear_interpolate(points: Sequence[tuple[float, float]], value: float) -> float:
    """Piecewise-linear interpolation with explicit end-segment extrapolation."""

    ordered = sorted((float(x), float(y)) for x, y in points)
    if len(ordered) < 2 or len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("prediction curve requires at least two unique anchors")
    if value <= ordered[0][0]:
        left, right = ordered[0], ordered[1]
    elif value >= ordered[-1][0]:
        left, right = ordered[-2], ordered[-1]
    else:
        left, right = next(
            (a, b) for a, b in pairwise(ordered) if a[0] <= value <= b[0]
        )
    weight = (value - left[0]) / (right[0] - left[0])
    return left[1] + weight * (right[1] - left[1])


def _linear_uncertainty(points: Sequence[tuple[float, float]], value: float) -> float:
    """Propagate nonnegative anchor spreads through linear interpolation.

    Extrapolation has a negative weight on one endpoint.  Absolute weights are
    intentional here: uncertainty cannot cancel merely because a prediction is
    just outside the measured range.
    """

    ordered = sorted((float(x), float(spread)) for x, spread in points)
    if (
        len(ordered) < 2
        or len({x for x, _ in ordered}) != len(ordered)
        or any(spread < 0 for _, spread in ordered)
    ):
        raise ValueError("uncertainty curve requires two unique nonnegative anchors")
    if value <= ordered[0][0]:
        left, right = ordered[0], ordered[1]
    elif value >= ordered[-1][0]:
        left, right = ordered[-2], ordered[-1]
    else:
        left, right = next(
            (a, b) for a, b in pairwise(ordered) if a[0] <= value <= b[0]
        )
    weight = (value - left[0]) / (right[0] - left[0])
    return abs(1.0 - weight) * left[1] + abs(weight) * right[1]


def _matching_anchors(
    fit: Mapping[str, Any],
    *,
    timer: str,
    protocol: str,
    placement: str,
    stage: str,
    family: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in fit["anchors"]
        if row["timer"] == timer
        and row["protocol"] == protocol
        and row["simple_placement"] == placement
        and row["stage"] == stage
        and row["family"] == family
    ]


def predict_confirmation_cell(
    fit: Mapping[str, Any], cell: Mapping[str, Any], *, timer: str
) -> dict[str, float]:
    """Predict one held-out cell from the already locked component profile."""

    requested = cell["requested"]
    protocol = str(cell["protocol"])
    placement = str(cell["simple_placement"])
    data = [
        row
        for row in _matching_anchors(
            fit,
            timer=timer,
            protocol=protocol,
            placement=placement,
            stage="data_work",
            family="sum",
        )
        if row["requested"].get("working_set") == "reused"
    ]
    data_points = [
        (float(row["requested"]["useful_bytes"]), float(row["summary"]["median_ns"]))
        for row in data
    ]
    data_spreads = [
        (float(row["requested"]["useful_bytes"]), float(row["summary"]["iqr_ns"]))
        for row in data
    ]
    payload = float(requested["useful_bytes_per_channel"])
    data_target = _linear_interpolate(data_points, payload)
    data_reference = _linear_interpolate(data_points, 1920.0)
    data_target_spread = _linear_uncertainty(data_spreads, payload)
    data_reference_spread = _linear_uncertainty(data_spreads, 1920.0)

    sharing = [
        row
        for row in _matching_anchors(
            fit,
            timer=timer,
            protocol=protocol,
            placement=placement,
            stage="sharing",
            family="sum",
        )
        if row["requested"]["available_sms"] == requested["available_sms"]
        and row["requested"]["working_warps"] == requested["working_warps"]
    ]
    sharing_points = [
        (float(row["requested"]["active_channels"]), float(row["summary"]["median_ns"]))
        for row in sharing
    ]
    sharing_spreads = [
        (float(row["requested"]["active_channels"]), float(row["summary"]["iqr_ns"]))
        for row in sharing
    ]
    channels = int(requested["active_channels"])
    sharing_reference = _linear_interpolate(sharing_points, float(channels))
    sharing_spread = _linear_uncertainty(sharing_spreads, float(channels))
    rank_two_no_delay = sharing_reference + channels * (data_target - data_reference)
    rank_multiplier = int(requested["ranks"]) - 1
    rank_scaled_no_delay = rank_multiplier * rank_two_no_delay

    publication_points = [
        (float(row["delay_cycles"]), float(row["median_delta_ns"]))
        for row in fit["publication_response"]
        if row["timer"] == timer
        and row["protocol"] == protocol
        and row["simple_placement"] == placement
    ]
    publication = _linear_interpolate(publication_points, float(requested["delay_cycles"]))
    publication_spreads = [
        (
            float(row["delay_cycles"]),
            float(row["q3_delta_ns"]) - float(row["q1_delta_ns"]),
        )
        for row in fit["publication_response"]
        if row["timer"] == timer
        and row["protocol"] == protocol
        and row["simple_placement"] == placement
    ]
    publication_spread = _linear_uncertainty(
        publication_spreads, float(requested["delay_cycles"])
    )
    propagated_spread = rank_multiplier * (
        sharing_spread + channels * (data_target_spread + data_reference_spread)
    ) + publication_spread
    return {
        "data_target_ns": data_target,
        "data_reference_ns": data_reference,
        "sharing_reference_ns": sharing_reference,
        "rank_scaled_no_delay_ns": rank_scaled_no_delay,
        "publication_delta_ns": publication,
        "predicted_ns": rank_scaled_no_delay + publication,
        "propagated_anchor_iqr_bound_ns": propagated_spread,
    }


def score_confirmations(
    rows: Sequence[Mapping[str, Any]],
    *,
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fit: Mapping[str, Any],
    observations_sha256: str,
) -> dict[str, Any]:
    """Open held-out cells and score adjacent one-factor intervention deltas."""

    unsigned_fit = dict(fit)
    recorded_digest = unsigned_fit.pop("fit_digest", None)
    if fit.get("schema") != FIT_SCHEMA or recorded_digest != content_digest(unsigned_fit):
        raise ValueError("identification fit schema or digest is invalid")
    if (
        fit["manifest_digest"] != inventory["manifest_digest"]
        or fit["inventory_digest"] != inventory["inventory_digest"]
    ):
        raise ValueError("identification fit targets a different frozen inventory")
    if fit["observations_sha256"] != observations_sha256:
        raise ValueError("identification fit and confirmation rows use different evidence")

    expected_processes = int(manifest["ordinary_processes_per_cell"])
    normalized = _rows_in_nanoseconds(rows)
    # Confirmation scoring compares many adjacent interventions against the
    # same held-out rows.  Reuse the process-first reduction exactly as the
    # identification fit does instead of scanning the complete capture for
    # each comparison.
    process_medians = _process_median_index(normalized)
    void_scopes = timing_scope_voids(rows)
    void_keys = {(row["cell_id"], row["timer"]) for row in void_scopes}
    cells = [cell for cell in inventory["cells"] if cell["stage"] == "confirmation"]
    predictions: dict[tuple[str, str], dict[str, float]] = {}
    for timer in manifest["timing_boundaries"]:
        for cell in cells:
            if (cell["cell_id"], timer) in void_keys:
                continue
            predictions[(cell["cell_id"], timer)] = predict_confirmation_cell(
                fit, cell, timer=timer
            )

    scored = []
    axes = (
        "ranks",
        "active_channels",
        "available_sms",
        "useful_bytes_per_channel",
        "delay_cycles",
    )
    for timer in manifest["timing_boundaries"]:
        timer_cells = [
            cell for cell in cells if (cell["cell_id"], timer) in predictions
        ]
        for axis in axes:
            groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
            for cell in timer_cells:
                groups[
                    (
                        str(cell["protocol"]),
                        str(cell["simple_placement"]),
                        _request_key(cell, omit=frozenset({axis})),
                    )
                ].append(cell)
            for group in groups.values():
                ordered = sorted(
                    group, key=lambda cell: _level_order(cell["requested"][axis])
                )
                for cell_a, cell_b in pairwise(ordered):
                    measured = paired_contrast(
                        normalized,
                        cell_a=cell_a["cell_id"],
                        cell_b=cell_b["cell_id"],
                        timer=timer,
                        expected_processes=expected_processes,
                        process_medians=process_medians,
                    )
                    prediction_a = predictions[(cell_a["cell_id"], timer)]
                    prediction_b = predictions[(cell_b["cell_id"], timer)]
                    predicted_delta = prediction_b["predicted_ns"] - prediction_a["predicted_ns"]
                    acceptance = score_confirmation(
                        measured_delta=float(measured["paired_median_delta"]),
                        predicted_delta=predicted_delta,
                        frozen_repeat_spread_resolution=float(
                            measured["resolution_threshold"]
                        ),
                    )
                    scored.append(
                        {
                            "timer": timer,
                            "units": "ns",
                            "protocol": cell_a["protocol"],
                            "simple_placement": cell_a["simple_placement"],
                            "factor": axis,
                            "level_a": cell_a["requested"][axis],
                            "level_b": cell_b["requested"][axis],
                            "cell_a": cell_a["cell_id"],
                            "cell_b": cell_b["cell_id"],
                            "resolved": measured["resolved"],
                            "resolution_threshold_ns": measured["resolution_threshold"],
                            "prediction_a": prediction_a,
                            "prediction_b": prediction_b,
                            **acceptance,
                            "acceptance_applicable": bool(measured["resolved"]),
                            "accepted_resolved_delta": (
                                bool(acceptance["accepted"])
                                if measured["resolved"]
                                else None
                            ),
                        }
                    )

    resolved_rows = [row for row in scored if row["acceptance_applicable"]]
    accepted_rows = [row for row in resolved_rows if row["accepted_resolved_delta"]]
    score = {
        "schema": SCORE_SCHEMA,
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "observations_sha256": observations_sha256,
        "fit_digest": fit["fit_digest"],
        "void_scopes": list(void_scopes),
        "scored_interventions": scored,
        "summary": {
            "intervention_count": len(scored),
            "resolved_interventions": len(resolved_rows),
            "unresolved_interventions": len(scored) - len(resolved_rows),
            "accepted_resolved_interventions": len(accepted_rows),
            "rejected_resolved_interventions": len(resolved_rows) - len(accepted_rows),
            "has_resolved_interventions": bool(resolved_rows),
            "all_resolved_interventions_accepted": (
                bool(resolved_rows) and len(accepted_rows) == len(resolved_rows)
            ),
            "acceptance_rule": manifest["confirmation_error_bound"],
        },
    }
    score["score_digest"] = content_digest(score)
    return score
