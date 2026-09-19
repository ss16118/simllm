#!/usr/bin/env python3
"""Freeze and score the TRAF-94 v4 selective uncertainty model.

V1 and v2 showed model-form error; v3 showed that exact point lookup still has
cross-run error.  V4 therefore predicts matched intervention *intervals*, not
cell-time points.  Two opened campaigns provide paired-process deltas and a
grouped cross-run residual.  Groups whose calibrated half-width would exceed
the effect/resolution scale abstain prospectively instead of emitting an
unbounded or falsely precise prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.nccl_primitive_identification_v1.matrix import (
    canonical_json_bytes,
    content_digest,
    load_manifest,
    validate_inventory,
)
from examples.nccl_primitive_interaction_v2.run_study import (
    AXES,
    _numeric_control,
    _paired_contrast,
    _percentile,
    _process_medians_from_jsonl,
    _rectangular_strata,
)

MODEL_SCHEMA = "simllm-nccl-primitive-uncertainty-model-v4"
SCORE_SCHEMA = "simllm-nccl-primitive-uncertainty-score-v4"


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
    """Publish one canonical object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _verify_digest(value: Mapping[str, Any], digest_name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(digest_name, None)
    if recorded != content_digest(unsigned):
        raise ValueError(f"invalid {digest_name}")


def _request_key(requested: Mapping[str, Any]) -> str:
    """Identify an executable control tuple independently of experiment role."""

    return content_digest(requested)


def _iqr(values: Sequence[float]) -> float:
    return _percentile(values, 0.75) - _percentile(values, 0.25)


def _resolution(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Use the same paired-IQR resolution floor as every earlier version."""

    return 2.0 * (_iqr(values_a) + _iqr(values_b))


def _ranked_quantile(values: Sequence[float], coverage: float) -> float:
    """Return the finite-sample upper ranked quantile.

    ``ceil((n + 1) * coverage)`` is deliberately more conservative than a
    linearly interpolated percentile.  The two directional residuals per
    intervention are dependent, so this is an empirical calibration rule, not
    a claim of exchangeable split-conformal coverage.
    """

    if not 0.0 < coverage < 1.0 or not values:
        raise ValueError("ranked quantile needs samples and coverage in (0, 1)")
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
    return ordered[rank - 1]


def _group_key(spec: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(spec["timer"]),
        str(spec["protocol"]),
        str(spec["simple_placement"]),
        str(spec["factor"]),
    )


def _group_record(key: Sequence[str]) -> dict[str, str]:
    return {
        "timer": key[0],
        "protocol": key[1],
        "simple_placement": key[2],
        "factor": key[3],
    }


def _contrast_specs(
    cells: Sequence[Mapping[str, Any]],
    *,
    timers: Sequence[str],
    full_sms: int,
) -> list[dict[str, Any]]:
    """Enumerate stable-id adjacent contrasts without depending on cell ids."""

    specs = []
    for timer in timers:
        for axis in AXES:
            groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for cell in cells:
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
                    body: dict[str, Any] = {
                        "timer": timer,
                        "protocol": cell_a["protocol"],
                        "simple_placement": cell_a["simple_placement"],
                        "factor": axis,
                        "level_a": cell_a["requested"][axis],
                        "level_b": cell_b["requested"][axis],
                        "request_a": cell_a["requested"],
                        "request_b": cell_b["requested"],
                    }
                    body["contrast_id"] = "contrast-" + content_digest(body)[:20]
                    specs.append(body)
    if len({spec["contrast_id"] for spec in specs}) != len(specs):
        raise AssertionError("contrast identity unexpectedly collided")
    return specs


def _training_run(
    *,
    label: str,
    evidence: Mapping[str, Any],
    inventory_path: Path,
    rows_path: Path,
    validation_path: Path,
    timers: Sequence[str],
    expected_processes: int,
    expected_repetitions: int,
) -> dict[str, Any]:
    """Load and fully bind one opened training campaign."""

    inventory = _read_json(inventory_path)
    validation = _read_json(validation_path)
    validate_inventory(inventory)
    if _sha256(inventory_path) != evidence["qualified_inventory_sha256"]:
        raise ValueError(f"{label} inventory file differs from the freeze")
    if inventory["inventory_digest"] != evidence["qualified_inventory_digest"]:
        raise ValueError(f"{label} inventory digest differs from the freeze")
    if _sha256(validation_path) != evidence["validation_sha256"]:
        raise ValueError(f"{label} validation file differs from the freeze")
    if validation.get("failures") or validation.get("void_scopes"):
        raise ValueError(f"{label} training validation is not fully valid")
    if validation.get("observations_sha256") != evidence["observations_sha256"]:
        raise ValueError(f"{label} validation targets another observation capture")

    cells = {cell["cell_id"]: cell for cell in inventory["cells"]}
    medians, observed_sha256 = _process_medians_from_jsonl(
        rows_path,
        cells=cells,
        timers=timers,
        expected_processes=expected_processes,
        expected_repetitions=expected_repetitions,
        expected_sha256=evidence["observations_sha256"],
    )
    by_request: dict[str, Mapping[str, Any]] = {}
    for cell in inventory["cells"]:
        key = _request_key(cell["requested"])
        if key in by_request:
            raise ValueError(f"{label} contains duplicate requested controls")
        by_request[key] = cell
    return {
        "label": label,
        "inventory": inventory,
        "cells": cells,
        "by_request": by_request,
        "medians": medians,
        "observations_sha256": observed_sha256,
        "validation_sha256": evidence["validation_sha256"],
    }


def freeze_model(
    *,
    manifest: Mapping[str, Any],
    holdout_plan: Mapping[str, Any],
    training_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit grouped uncertainty using only the two opened campaigns."""

    if len(training_runs) != 2:
        raise ValueError("v4 requires exactly two opened training campaigns")
    validate_inventory(holdout_plan)
    manifest_digest = content_digest(manifest)
    if holdout_plan.get("manifest_digest") != manifest_digest:
        raise ValueError("holdout plan targets another v4 manifest")

    # A planned cell is modellable only when its exact request exists in both
    # training runs.  This retains the unsupported rank-3 direct-read cells as
    # explicit unmodelled plan entries rather than silently deleting them.
    predicted_cells = []
    unmodelled_cells = []
    for cell in holdout_plan["cells"]:
        key = _request_key(cell["requested"])
        if all(key in run["by_request"] for run in training_runs):
            predicted_cells.append(cell)
        else:
            unmodelled_cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "reason": "exact_requested_control_absent_from_one_or_more_training_runs",
                }
            )
    if len(predicted_cells) != 112 or len(unmodelled_cells) != 16:
        raise ValueError("v4 cell coverage differs from the frozen contract")
    _rectangular_strata(
        predicted_cells,
        full_sms=int(manifest["model_spec"]["full_h100_sms"]),
    )

    specs = _contrast_specs(
        predicted_cells,
        timers=manifest["timing_boundaries"],
        full_sms=int(manifest["model_spec"]["full_h100_sms"]),
    )
    training_rows = []
    directional_residuals: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for spec in specs:
        per_run = []
        for run in training_runs:
            cell_a = run["by_request"][_request_key(spec["request_a"])]
            cell_b = run["by_request"][_request_key(spec["request_b"])]
            values_a = run["medians"][(cell_a["cell_id"], spec["timer"])]
            values_b = run["medians"][(cell_b["cell_id"], spec["timer"])]
            paired = [
                float(value_b) - float(value_a)
                for value_a, value_b in zip(values_a, values_b, strict=True)
            ]
            run_resolution = _resolution(values_a, values_b)
            per_run.append(
                {
                    "label": run["label"],
                    "cell_a": cell_a["cell_id"],
                    "cell_b": cell_b["cell_id"],
                    "paired_process_deltas_ns": paired,
                    "median_delta_ns": statistics.median(paired),
                    "resolution_ns": run_resolution,
                }
            )
        first, second = per_run
        disagreement = abs(first["median_delta_ns"] - second["median_delta_ns"])
        key = _group_key(spec)
        for source in per_run:
            denominator = max(
                abs(float(source["median_delta_ns"])),
                float(source["resolution_ns"]),
                1.0,
            )
            directional_residuals[key].append(disagreement / denominator)
        training_rows.append({"spec": spec, "per_run": per_run})

    group_rows = []
    group_quantiles = {}
    for key in sorted(directional_residuals):
        residuals = directional_residuals[key]
        quantile = _ranked_quantile(residuals, 0.90)
        group_quantiles[key] = quantile
        group_rows.append(
            {
                **_group_record(key),
                "directional_residual_count": len(residuals),
                "directional_residuals": residuals,
                "ranked_90_percent_quantile": quantile,
                "status": "predict" if quantile <= 1.0 else "abstain",
            }
        )

    predictions = []
    for row in training_rows:
        spec = row["spec"]
        per_run = row["per_run"]
        all_process_deltas = [
            value
            for run_row in per_run
            for value in run_row["paired_process_deltas_ns"]
        ]
        center = statistics.median(all_process_deltas)
        resolution_floor = max(float(run["resolution_ns"]) for run in per_run)
        scale = max(abs(center), resolution_floor, 1.0)
        quantile = group_quantiles[_group_key(spec)]
        half_width = max(resolution_floor, quantile * scale)
        normalized_half_width = half_width / scale
        status = "predict" if quantile <= 1.0 else "abstain"
        predictions.append(
            {
                **spec,
                "status": status,
                "predicted_delta_ns": center,
                "lower_ns": center - half_width,
                "upper_ns": center + half_width,
                "half_width_ns": half_width,
                "resolution_floor_ns": resolution_floor,
                "normalization_scale_ns": scale,
                "normalized_half_width": normalized_half_width,
                "group_quantile": quantile,
                "training_runs": per_run,
            }
        )

    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "manifest_digest": manifest_digest,
        "planned_inventory_digest": holdout_plan["inventory_digest"],
        "device_cohort_digest": manifest["device_cohort"]["cohort_digest"],
        "training": [
            {
                "label": run["label"],
                "qualified_inventory_digest": run["inventory"]["inventory_digest"],
                "observations_sha256": run["observations_sha256"],
                "validation_sha256": run["validation_sha256"],
                "cell_count": len(run["cells"]),
            }
            for run in training_runs
        ],
        "calibration_groups": group_rows,
        "contrast_predictions": predictions,
        "unmodelled_planned_cells": unmodelled_cells,
    }
    model["model_digest"] = content_digest(model)
    return model


def score_model(
    *,
    manifest: Mapping[str, Any],
    planned_inventory: Mapping[str, Any],
    qualified_inventory: Mapping[str, Any],
    rows_path: Path,
    validation: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Score only the intervals and abstentions frozen before v4 timing."""

    manifest_digest = content_digest(manifest)
    if model.get("manifest_digest") != manifest_digest:
        raise ValueError("model targets another v4 manifest")
    if model.get("planned_inventory_digest") != planned_inventory.get("inventory_digest"):
        raise ValueError("model targets another planned inventory")
    if model.get("device_cohort_digest") != manifest["device_cohort"]["cohort_digest"]:
        raise ValueError("model targets another device cohort")
    if qualified_inventory.get("manifest_digest") != manifest_digest:
        raise ValueError("qualified inventory targets another manifest")
    if validation.get("inventory_digest") != qualified_inventory.get("inventory_digest"):
        raise ValueError("validation targets another qualified inventory")
    if validation.get("failures") or validation.get("void_scopes"):
        raise ValueError("v4 validation is not fully valid")

    planned_by_id = {cell["cell_id"]: cell for cell in planned_inventory["cells"]}
    qualified_cells = list(qualified_inventory["cells"])
    if any(
        cell["cell_id"] not in planned_by_id or cell != planned_by_id[cell["cell_id"]]
        for cell in qualified_cells
    ):
        raise ValueError("qualified inventory is not an exact subset of the plan")
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
    request_to_id = {
        _request_key(cell["requested"]): cell["cell_id"] for cell in qualified_cells
    }
    frozen = {row["contrast_id"]: row for row in model["contrast_predictions"]}
    specs = _contrast_specs(
        qualified_cells,
        timers=manifest["timing_boundaries"],
        full_sms=full_sms,
    )
    if any(spec["contrast_id"] not in frozen for spec in specs):
        raise ValueError("qualified contrast lacks a frozen v4 prediction")

    scored = []
    for spec in specs:
        prediction = frozen[spec["contrast_id"]]
        cell_a = request_to_id[_request_key(spec["request_a"])]
        cell_b = request_to_id[_request_key(spec["request_b"])]
        measured = _paired_contrast(
            medians,
            cell_a=cell_a,
            cell_b=cell_b,
            timer=spec["timer"],
        )
        measured_delta = float(measured["paired_median_delta_ns"])
        point_error = abs(float(prediction["predicted_delta_ns"]) - measured_delta)
        point_bound = max(
            0.10 * abs(measured_delta),
            float(measured["resolution_threshold_ns"]),
        )
        resolved = bool(measured["resolved"])
        predicts = prediction["status"] == "predict"
        covered = (
            float(prediction["lower_ns"])
            <= measured_delta
            <= float(prediction["upper_ns"])
        )
        scored.append(
            {
                **{key: value for key, value in spec.items() if not key.startswith("request_")},
                "cell_a": cell_a,
                "cell_b": cell_b,
                **measured,
                "prediction_status": prediction["status"],
                "predicted_delta_ns": prediction["predicted_delta_ns"],
                "prediction_lower_ns": prediction["lower_ns"],
                "prediction_upper_ns": prediction["upper_ns"],
                "normalized_half_width": prediction["normalized_half_width"],
                "interval_coverage_applicable": resolved and predicts,
                "interval_covers_measured_delta": covered if resolved and predicts else None,
                "legacy_point_error_ns": point_error,
                "legacy_point_bound_ns": point_bound,
                "legacy_point_accepted": point_error <= point_bound if resolved else None,
            }
        )

    resolved_rows = [row for row in scored if row["resolved"]]
    predicted_resolved = [
        row for row in resolved_rows if row["prediction_status"] == "predict"
    ]
    covered = [row for row in predicted_resolved if row["interval_covers_measured_delta"]]
    retention = len(predicted_resolved) / len(resolved_rows) if resolved_rows else 0.0
    coverage = len(covered) / len(predicted_resolved) if predicted_resolved else 0.0
    timer_summaries = []
    for timer in manifest["timing_boundaries"]:
        rows = [row for row in predicted_resolved if row["timer"] == timer]
        timer_covered = [row for row in rows if row["interval_covers_measured_delta"]]
        timer_summaries.append(
            {
                "timer": timer,
                "predicted_resolved_interventions": len(rows),
                "covered_interventions": len(timer_covered),
                "coverage": len(timer_covered) / len(rows) if rows else 0.0,
            }
        )
    normalized_widths = [float(row["normalized_half_width"]) for row in predicted_resolved]
    contract = manifest["holdout_contract"]
    gates = {
        "has_minimum_resolved_interventions": len(resolved_rows)
        >= int(contract["minimum_resolved_interventions"]),
        "resolved_retention": retention >= float(contract["minimum_resolved_retention"]),
        "overall_interval_coverage": coverage
        >= float(contract["minimum_overall_interval_coverage"]),
        "per_timer_interval_coverage": all(
            row["predicted_resolved_interventions"] > 0
            and row["coverage"] >= float(contract["minimum_per_timer_interval_coverage"])
            for row in timer_summaries
        ),
        "maximum_normalized_half_width": bool(normalized_widths)
        and max(normalized_widths) <= float(contract["maximum_normalized_half_width"]),
    }
    passed = all(gates.values())
    summary = {
        "intervention_count": len(scored),
        "resolved_interventions": len(resolved_rows),
        "unresolved_interventions": len(scored) - len(resolved_rows),
        "predicted_resolved_interventions": len(predicted_resolved),
        "abstained_resolved_interventions": len(resolved_rows) - len(predicted_resolved),
        "resolved_retention": retention,
        "covered_predicted_resolved_interventions": len(covered),
        "interval_coverage": coverage,
        "per_timer": timer_summaries,
        "maximum_normalized_half_width": max(normalized_widths) if normalized_widths else None,
        "legacy_point_accepted_resolved_interventions": sum(
            row["legacy_point_accepted"] is True for row in resolved_rows
        ),
        "acceptance_gates": gates,
    }
    score: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "manifest_digest": manifest_digest,
        "planned_inventory_digest": planned_inventory["inventory_digest"],
        "qualified_inventory_digest": qualified_inventory["inventory_digest"],
        "observations_sha256": observations_sha256,
        "model_digest": model["model_digest"],
        "device_cohort_digest": model["device_cohort_digest"],
        "outcome": "PASS" if passed else "FAIL",
        "summary": summary,
        "scored_interventions": scored,
    }
    score["score_digest"] = content_digest(score)
    return score


def fit_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    evidence = manifest["training_evidence"]
    timers = manifest["timing_boundaries"]
    common = {
        "timers": timers,
        "expected_processes": int(manifest["ordinary_processes_per_cell"]),
        "expected_repetitions": int(manifest["recorded_iterations"]),
    }
    runs = [
        _training_run(
            label="v2",
            evidence=evidence["v2"],
            inventory_path=args.v2_inventory,
            rows_path=args.v2_rows,
            validation_path=args.v2_validation,
            **common,
        ),
        _training_run(
            label="v3",
            evidence=evidence["v3"],
            inventory_path=args.v3_inventory,
            rows_path=args.v3_rows,
            validation_path=args.v3_validation,
            **common,
        ),
    ]
    model = freeze_model(
        manifest=manifest,
        holdout_plan=_read_json(args.holdout_plan),
        training_runs=runs,
    )
    _write_json(args.output, model)
    print(
        json.dumps(
            {
                "status": "complete",
                "model_digest": model["model_digest"],
                "contrast_predictions": len(model["contrast_predictions"]),
                "calibration_groups": len(model["calibration_groups"]),
                "predict_groups": sum(
                    row["status"] == "predict" for row in model["calibration_groups"]
                ),
                "abstain_groups": sum(
                    row["status"] == "abstain" for row in model["calibration_groups"]
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def score_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    planned = _read_json(args.planned_inventory)
    qualified = _read_json(args.qualified_inventory)
    model = _read_json(args.model)
    validate_inventory(planned)
    validate_inventory(qualified)
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported v4 model schema")
    _verify_digest(model, "model_digest")
    score = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=args.rows,
        validation=_read_json(args.validation),
        model=model,
    )
    _write_json(args.output, score)
    print(
        json.dumps(
            {
                "status": "complete",
                **score["summary"],
                "outcome": score["outcome"],
                "score_digest": score["score_digest"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--holdout-plan", type=Path, required=True)
    for version in ("v2", "v3"):
        fit.add_argument(f"--{version}-inventory", type=Path, required=True)
        fit.add_argument(f"--{version}-rows", type=Path, required=True)
        fit.add_argument(f"--{version}-validation", type=Path, required=True)
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
