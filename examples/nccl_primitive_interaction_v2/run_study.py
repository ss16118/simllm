#!/usr/bin/env python3
"""Fit and score the frozen TRAF-94 v2 interaction surface.

The CUDA/NCCL execution path remains in ``nccl_primitive_identification_v1``.
This module owns only the v2 evidence transformation: it reduces raw repeats
to process medians, locks a full interaction tensor from the opened v1
confirmation grid, freezes predictions for a new inventory, and scores that
inventory without allowing a refit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.nccl_primitive_identification_v1.matrix import (
    TIMER_UNITS,
    canonical_json_bytes,
    content_digest,
    load_manifest,
    validate_inventory,
)

MODEL_SCHEMA = "simllm-nccl-primitive-interaction-model-v2"
SCORE_SCHEMA = "simllm-nccl-primitive-interaction-score-v2"
AXES = (
    "ranks",
    "active_channels",
    "available_sms",
    "useful_bytes_per_channel",
    "delay_cycles",
)
_UNIT_TO_NS = {"ns": 1.0, "us": 1_000.0, "ps": 0.001}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete object atomically, never a partial model/score."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("a percentile requires at least one value")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_control(name: str, value: Any, *, full_sms: int) -> float:
    """Map one frozen request control onto its interpolation coordinate."""

    if name == "available_sms" and value == "full":
        return float(full_sms)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"axis {name!r} has non-numeric value {value!r}")
    return float(value)


def _coordinate(requested: Mapping[str, Any], *, full_sms: int) -> tuple[float, ...]:
    return tuple(
        _numeric_control(name, requested[name], full_sms=full_sms) for name in AXES
    )


def _verified_result(path: Path, schema: str, digest_name: str) -> dict[str, Any]:
    value = _read_json(path)
    unsigned = dict(value)
    recorded = unsigned.pop(digest_name, None)
    if value.get("schema") != schema or recorded != content_digest(unsigned):
        raise ValueError(f"{path} has an invalid schema or content digest")
    return value


def _process_medians_from_jsonl(
    rows_path: Path,
    *,
    cells: Mapping[str, Mapping[str, Any]],
    timers: Sequence[str],
    expected_processes: int,
    expected_repetitions: int,
    expected_sha256: str | None,
) -> tuple[dict[tuple[str, str], list[float]], str]:
    """Stream one capture and reduce only selected cells to process medians.

    The two-gigabyte v1 capture must not be copied into a Python object graph.
    Hashing and reduction therefore happen in one pass.  Exact repetition keys
    are retained until completeness is proven, after which only the small
    process-median table remains.
    """

    durations: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    repetitions: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    digest = hashlib.sha256()
    timer_set = set(timers)
    with rows_path.open("rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            row = json.loads(raw_line)
            cell_id = str(row.get("cell_id", ""))
            if cell_id not in cells:
                continue
            timer = str(row.get("timer", ""))
            process = int(row.get("process_id", -1))
            repetition = int(row.get("repetition", -1))
            if timer not in timer_set:
                raise ValueError(f"selected row has unexpected timer {timer!r}")
            if process not in range(expected_processes):
                raise ValueError(f"selected row has invalid process {process}")
            if repetition not in range(expected_repetitions):
                raise ValueError(f"selected row has invalid repetition {repetition}")
            if row.get("phase") != "ordinary" or row.get("qualification") != "qualified":
                raise ValueError("selected row is not qualified ordinary evidence")
            if row.get("requested") != cells[cell_id]["requested"]:
                raise ValueError("selected row request differs from its frozen cell")
            unit = str(row.get("units", ""))
            if unit != TIMER_UNITS[timer] or unit not in _UNIT_TO_NS:
                raise ValueError(f"selected row has invalid unit {unit!r} for {timer!r}")
            key = (cell_id, timer, process)
            if repetition in repetitions[key]:
                raise ValueError(f"duplicate selected repetition for {key!r}")
            repetitions[key].add(repetition)
            durations[key].append(float(row["raw_duration"]) * _UNIT_TO_NS[unit])

    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("observation file differs from the frozen SHA-256")

    expected_repetition_set = set(range(expected_repetitions))
    medians: dict[tuple[str, str], list[float]] = {}
    for cell_id in cells:
        for timer in timers:
            values = []
            for process in range(expected_processes):
                key = (cell_id, timer, process)
                if repetitions.get(key) != expected_repetition_set:
                    raise ValueError(f"incomplete process repetitions for {key!r}")
                values.append(statistics.median(durations[key]))
            medians[(cell_id, timer)] = values
    return medians, observed_sha256


def _summary(values: Sequence[float]) -> dict[str, Any]:
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    return {
        "process_medians_ns": list(values),
        "median_ns": statistics.median(values),
        "q1_ns": q1,
        "q3_ns": q3,
        "iqr_ns": q3 - q1,
    }


def _bracket(levels: Sequence[float], value: float) -> tuple[tuple[float, float], ...]:
    """Return exact or two-point linear weights and reject extrapolation."""

    ordered = sorted(float(level) for level in levels)
    if len(ordered) == 1:
        if value != ordered[0]:
            raise ValueError(f"value {value} is outside singleton level {ordered[0]}")
        return ((ordered[0], 1.0),)
    if value < ordered[0] or value > ordered[-1]:
        raise ValueError(f"value {value} is outside [{ordered[0]}, {ordered[-1]}]")
    if value in ordered:
        return ((value, 1.0),)
    for lower, upper in itertools.pairwise(ordered):
        if lower < value < upper:
            upper_weight = (value - lower) / (upper - lower)
            return ((lower, 1.0 - upper_weight), (upper, upper_weight))
    raise AssertionError("a bounded interpolation value had no bracket")


def _stratum_key(timer: str, protocol: str, placement: str) -> tuple[str, str, str]:
    return (timer, protocol, placement)


def _predict_from_stratum(
    stratum: Mapping[str, Any], coordinate: Sequence[float]
) -> dict[str, Any]:
    levels = [tuple(float(value) for value in stratum["levels"][name]) for name in AXES]
    choices = [_bracket(axis_levels, float(value)) for axis_levels, value in zip(levels, coordinate, strict=True)]
    anchors = {
        tuple(float(value) for value in row["coordinate"]): row
        for row in stratum["anchors"]
    }
    predicted = 0.0
    propagated_iqr = 0.0
    contributors = []
    for selection in itertools.product(*choices):
        corner = tuple(value for value, _ in selection)
        weight = 1.0
        for _, axis_weight in selection:
            weight *= axis_weight
        anchor = anchors.get(corner)
        if anchor is None:
            raise ValueError(f"training tensor is missing interpolation corner {corner!r}")
        predicted += weight * float(anchor["summary"]["median_ns"])
        propagated_iqr += abs(weight) * float(anchor["summary"]["iqr_ns"])
        contributors.append({"cell_id": anchor["cell_id"], "weight": weight})
    return {
        "predicted_ns": predicted,
        "propagated_anchor_iqr_ns": propagated_iqr,
        "contributors": contributors,
    }


def fit_model(
    *,
    manifest: Mapping[str, Any],
    training_inventory: Mapping[str, Any],
    training_rows: Path,
    training_validation: Mapping[str, Any],
    training_validation_sha256: str,
    v1_results: Mapping[str, Any],
    holdout_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock the v2 tensor and every predictable holdout value."""

    evidence = manifest["training_evidence"]
    if training_inventory["inventory_digest"] != evidence["inventory_digest"]:
        raise ValueError("training inventory differs from the v2 freeze")
    if training_validation_sha256 != evidence["validation_sha256"]:
        raise ValueError("training validation report differs from the v2 freeze")
    if training_validation.get("failures"):
        raise ValueError("v1 training validation contains fatal failures")
    if training_validation.get("observations_sha256") != evidence["observations_sha256"]:
        raise ValueError("v1 validation targets another observation file")
    if (
        v1_results.get("observations_sha256") != evidence["observations_sha256"]
        or v1_results.get("fit_digest") != evidence["v1_fit_digest"]
        or v1_results.get("score_digest") != evidence["v1_score_digest"]
    ):
        raise ValueError("published v1 result differs from the v2 training declaration")

    manifest_digest = content_digest(manifest)
    if holdout_plan.get("manifest_digest") != manifest_digest:
        raise ValueError("holdout plan targets another manifest")
    if holdout_plan.get("state") != "planned":
        raise ValueError("holdout model must lock against the planned inventory")

    training_cells = {
        str(cell["cell_id"]): cell
        for cell in training_inventory["cells"]
        if cell.get("stage") == manifest["model_spec"]["training_stage"]
    }
    void_keys = {
        (str(row["cell_id"]), str(row["timer"]))
        for row in training_validation.get("void_scopes", ())
    }
    if any(cell_id in training_cells for cell_id, _ in void_keys):
        raise ValueError("a v1 confirmation training scope was voided")

    timers = tuple(str(value) for value in manifest["timing_boundaries"])
    medians, observations_sha256 = _process_medians_from_jsonl(
        training_rows,
        cells=training_cells,
        timers=timers,
        expected_processes=int(manifest["ordinary_processes_per_cell"]),
        expected_repetitions=int(manifest["recorded_iterations"]),
        expected_sha256=str(evidence["observations_sha256"]),
    )
    full_sms = int(manifest["model_spec"]["full_h100_sms"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in training_cells.values():
        coordinate = _coordinate(cell["requested"], full_sms=full_sms)
        for timer in timers:
            grouped[_stratum_key(timer, cell["protocol"], cell["simple_placement"])].append(
                {
                    "cell_id": cell["cell_id"],
                    "coordinate": list(coordinate),
                    "summary": _summary(medians[(cell["cell_id"], timer)]),
                }
            )

    strata = []
    for (timer, protocol, placement), anchors in sorted(grouped.items()):
        levels = {
            name: sorted({float(row["coordinate"][index]) for row in anchors})
            for index, name in enumerate(AXES)
        }
        expected_coordinates = set(itertools.product(*(levels[name] for name in AXES)))
        observed_coordinates = {
            tuple(float(value) for value in row["coordinate"]) for row in anchors
        }
        if observed_coordinates != expected_coordinates:
            raise ValueError(f"training stratum {(timer, protocol, placement)!r} is not rectangular")
        strata.append(
            {
                "timer": timer,
                "protocol": protocol,
                "simple_placement": placement,
                "levels": levels,
                "anchors": sorted(anchors, key=lambda row: tuple(row["coordinate"])),
            }
        )

    stratum_map = {
        _stratum_key(row["timer"], row["protocol"], row["simple_placement"]): row
        for row in strata
    }
    predictions = []
    unpredicted = []
    for cell in holdout_plan["cells"]:
        coordinate = _coordinate(cell["requested"], full_sms=full_sms)
        cell_predictions = []
        failure = None
        for timer in timers:
            key = _stratum_key(timer, cell["protocol"], cell["simple_placement"])
            try:
                predicted = _predict_from_stratum(stratum_map[key], coordinate)
            except ValueError as error:
                failure = str(error)
                break
            cell_predictions.append({"timer": timer, **predicted})
        if failure is not None:
            unpredicted.append({"cell_id": cell["cell_id"], "reason": failure})
        else:
            predictions.append(
                {
                    "cell_id": cell["cell_id"],
                    "coordinate": list(coordinate),
                    "timers": cell_predictions,
                }
            )

    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "manifest_digest": manifest_digest,
        "planned_inventory_digest": holdout_plan["inventory_digest"],
        "training": {
            "inventory_digest": training_inventory["inventory_digest"],
            "observations_sha256": observations_sha256,
            "validation_sha256": training_validation_sha256,
            "v1_result_digest": v1_results["result_digest"],
            "cell_count": len(training_cells),
            "v2_holdout_values_used_during_fit": 0,
        },
        "model_spec": manifest["model_spec"],
        "strata": strata,
        "holdout_predictions": predictions,
        "unpredicted_planned_cells": unpredicted,
    }
    unsigned = dict(model)
    model["model_digest"] = content_digest(unsigned)
    return model


def _paired_contrast(
    medians: Mapping[tuple[str, str], Sequence[float]],
    *,
    cell_a: str,
    cell_b: str,
    timer: str,
) -> dict[str, Any]:
    values_a = list(medians[(cell_a, timer)])
    values_b = list(medians[(cell_b, timer)])
    paired = [right - left for left, right in zip(values_a, values_b, strict=True)]
    iqr_a = _percentile(values_a, 0.75) - _percentile(values_a, 0.25)
    iqr_b = _percentile(values_b, 0.75) - _percentile(values_b, 0.25)
    delta = statistics.median(paired)
    threshold = 2.0 * (iqr_a + iqr_b)
    return {
        "process_medians_a_ns": values_a,
        "process_medians_b_ns": values_b,
        "paired_process_deltas_ns": paired,
        "paired_median_delta_ns": delta,
        "iqr_a_ns": iqr_a,
        "iqr_b_ns": iqr_b,
        "resolution_threshold_ns": threshold,
        "resolved": abs(delta) > threshold,
    }


def _rectangular_strata(cells: Sequence[Mapping[str, Any]], *, full_sms: int) -> None:
    grouped: dict[tuple[str, str], set[tuple[float, ...]]] = defaultdict(set)
    for cell in cells:
        grouped[(cell["protocol"], cell["simple_placement"])].add(
            _coordinate(cell["requested"], full_sms=full_sms)
        )
    for key, coordinates in grouped.items():
        levels = [sorted({coordinate[index] for coordinate in coordinates}) for index in range(len(AXES))]
        if coordinates != set(itertools.product(*levels)):
            raise ValueError(f"qualified holdout stratum {key!r} is not rectangular")


def score_model(
    *,
    manifest: Mapping[str, Any],
    planned_inventory: Mapping[str, Any],
    qualified_inventory: Mapping[str, Any],
    rows_path: Path,
    validation: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Score only predictions locked before the new holdout was opened."""

    manifest_digest = content_digest(manifest)
    if model.get("manifest_digest") != manifest_digest:
        raise ValueError("model targets another v2 manifest")
    if model.get("planned_inventory_digest") != planned_inventory.get("inventory_digest"):
        raise ValueError("model targets another planned holdout")
    unsigned_model = dict(model)
    recorded_model_digest = unsigned_model.pop("model_digest", None)
    if recorded_model_digest != content_digest(unsigned_model):
        raise ValueError("v2 model digest is invalid")
    if qualified_inventory.get("manifest_digest") != manifest_digest:
        raise ValueError("qualified holdout targets another manifest")
    planned_by_id = {cell["cell_id"]: cell for cell in planned_inventory["cells"]}
    qualified_cells = list(qualified_inventory["cells"])
    if any(
        cell["cell_id"] not in planned_by_id or cell != planned_by_id[cell["cell_id"]]
        for cell in qualified_cells
    ):
        raise ValueError("qualified holdout is not an exact subset of the plan")
    if validation.get("failures"):
        raise ValueError("holdout validation contains fatal failures")
    if validation.get("inventory_digest") != qualified_inventory.get("inventory_digest"):
        raise ValueError("holdout validation targets another qualified inventory")
    void_scopes = list(validation.get("void_scopes", ()))
    if void_scopes:
        raise ValueError("v2 does not score a holdout containing void timer scopes")

    prediction_rows = {
        row["cell_id"]: row for row in model["holdout_predictions"]
    }
    if any(cell["cell_id"] not in prediction_rows for cell in qualified_cells):
        raise ValueError("a qualified holdout cell had no pre-opened prediction")
    full_sms = int(manifest["model_spec"]["full_h100_sms"])
    _rectangular_strata(qualified_cells, full_sms=full_sms)

    cells_by_id = {cell["cell_id"]: cell for cell in qualified_cells}
    medians, observations_sha256 = _process_medians_from_jsonl(
        rows_path,
        cells=cells_by_id,
        timers=manifest["timing_boundaries"],
        expected_processes=int(manifest["ordinary_processes_per_cell"]),
        expected_repetitions=int(manifest["recorded_iterations"]),
        expected_sha256=validation.get("observations_sha256"),
    )
    frozen_predictions = {
        (row["cell_id"], timer_row["timer"]): timer_row
        for row in model["holdout_predictions"]
        for timer_row in row["timers"]
    }

    scored = []
    for timer in manifest["timing_boundaries"]:
        for axis in AXES:
            groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for cell in qualified_cells:
                identity = {
                    "protocol": cell["protocol"],
                    "simple_placement": cell["simple_placement"],
                    "requested": {
                        name: value
                        for name, value in cell["requested"].items()
                        if name != axis
                    },
                }
                groups[content_digest(identity)].append(cell)
            for members in groups.values():
                ordered = sorted(
                    members,
                    key=lambda cell: _numeric_control(
                        axis, cell["requested"][axis], full_sms=full_sms
                    ),
                )
                for cell_a, cell_b in itertools.pairwise(ordered):
                    level_a = _numeric_control(
                        axis, cell_a["requested"][axis], full_sms=full_sms
                    )
                    level_b = _numeric_control(
                        axis, cell_b["requested"][axis], full_sms=full_sms
                    )
                    if level_a == level_b:
                        continue
                    measured = _paired_contrast(
                        medians,
                        cell_a=cell_a["cell_id"],
                        cell_b=cell_b["cell_id"],
                        timer=timer,
                    )
                    predicted_delta = (
                        float(frozen_predictions[(cell_b["cell_id"], timer)]["predicted_ns"])
                        - float(frozen_predictions[(cell_a["cell_id"], timer)]["predicted_ns"])
                    )
                    measured_delta = float(measured["paired_median_delta_ns"])
                    error = abs(predicted_delta - measured_delta)
                    bound = max(
                        0.10 * abs(measured_delta),
                        float(measured["resolution_threshold_ns"]),
                    )
                    resolved = bool(measured["resolved"])
                    scored.append(
                        {
                            "timer": timer,
                            "protocol": cell_a["protocol"],
                            "simple_placement": cell_a["simple_placement"],
                            "factor": axis,
                            "level_a": cell_a["requested"][axis],
                            "level_b": cell_b["requested"][axis],
                            "cell_a": cell_a["cell_id"],
                            "cell_b": cell_b["cell_id"],
                            **measured,
                            "predicted_delta_ns": predicted_delta,
                            "absolute_error_ns": error,
                            "acceptance_bound_ns": bound,
                            "acceptance_applicable": resolved,
                            "accepted_resolved_delta": error <= bound if resolved else None,
                        }
                    )

    resolved_rows = [row for row in scored if row["acceptance_applicable"]]
    accepted_rows = [row for row in resolved_rows if row["accepted_resolved_delta"]]
    passed = bool(resolved_rows) and len(accepted_rows) == len(resolved_rows)
    score: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "manifest_digest": manifest_digest,
        "planned_inventory_digest": planned_inventory["inventory_digest"],
        "qualified_inventory_digest": qualified_inventory["inventory_digest"],
        "observations_sha256": observations_sha256,
        "model_digest": model["model_digest"],
        "outcome": "PASS" if passed else "FAIL",
        "scored_interventions": scored,
        "summary": {
            "intervention_count": len(scored),
            "resolved_interventions": len(resolved_rows),
            "unresolved_interventions": len(scored) - len(resolved_rows),
            "accepted_resolved_interventions": len(accepted_rows),
            "rejected_resolved_interventions": len(resolved_rows) - len(accepted_rows),
            "has_resolved_interventions": bool(resolved_rows),
            "all_resolved_interventions_accepted": passed,
            "acceptance_rule": manifest["confirmation_error_bound"],
        },
    }
    score["score_digest"] = content_digest(score)
    return score


def fit_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    training_inventory = _read_json(args.training_inventory)
    validate_inventory(training_inventory)
    holdout_plan = _read_json(args.holdout_plan)
    validate_inventory(holdout_plan)
    validation = _read_json(args.training_validation)
    v1_results = _verified_result(
        args.v1_results,
        "simllm-nccl-primitive-hardware-result-v1",
        "result_digest",
    )
    model = fit_model(
        manifest=manifest,
        training_inventory=training_inventory,
        training_rows=args.training_rows,
        training_validation=validation,
        training_validation_sha256=_sha256(args.training_validation),
        v1_results=v1_results,
        holdout_plan=holdout_plan,
    )
    _write_json(args.output, model)
    print(
        json.dumps(
            {
                "status": "complete",
                "model_digest": model["model_digest"],
                "training_cells": model["training"]["cell_count"],
                "strata": len(model["strata"]),
                "frozen_holdout_predictions": len(model["holdout_predictions"]),
                "unpredicted_planned_cells": len(model["unpredicted_planned_cells"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def score_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    planned = _read_json(args.planned_inventory)
    qualified = _read_json(args.qualified_inventory)
    validate_inventory(planned)
    validate_inventory(qualified)
    model = _verified_result(args.model, MODEL_SCHEMA, "model_digest")
    validation = _read_json(args.validation)
    score = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=args.rows,
        validation=validation,
        model=model,
    )
    _write_json(args.output, score)
    print(json.dumps({"status": "complete", **score["summary"], "outcome": score["outcome"], "score_digest": score["score_digest"], "output": str(args.output)}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--training-inventory", type=Path, required=True)
    fit.add_argument("--training-rows", type=Path, required=True)
    fit.add_argument("--training-validation", type=Path, required=True)
    fit.add_argument("--v1-results", type=Path, required=True)
    fit.add_argument("--holdout-plan", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(function=fit_command)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--planned-inventory", type=Path, required=True)
    score.add_argument("--qualified-inventory", type=Path, required=True)
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--validation", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(function=score_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
