#!/usr/bin/env python3
"""Freeze and score the TRAF-94 v3 empirical replication table.

V2 proved that multilinear interpolation across sparse NCCL control points is
not accurate enough near runtime regime changes.  V3 consequently makes a
narrower claim: values measured at the exact v2 controls should reproduce in
new processes on the same node and pinned probe.  This module deliberately has
no interpolation path.  A control tuple absent from v2 is unmodellable rather
than silently approximated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from examples.nccl_primitive_identification_v1.matrix import (
    canonical_json_bytes,
    content_digest,
    load_manifest,
    validate_inventory,
)
from examples.nccl_primitive_interaction_v2.run_study import (
    _process_medians_from_jsonl,
    _rectangular_strata,
    _summary,
    score_model,
)

MODEL_SCHEMA = "simllm-nccl-primitive-replication-model-v3"
SCORE_SCHEMA = "simllm-nccl-primitive-replication-score-v3"


def _sha256(path: Path) -> str:
    """Hash a large artifact without retaining it in memory."""

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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a canonical identity-bearing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _request_key(cell: dict[str, Any]) -> str:
    """Identify a physical request independently of role and cell-id changes."""

    return content_digest(cell["requested"])


def _verify_digested(value: dict[str, Any], digest_name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(digest_name, None)
    if recorded != content_digest(unsigned):
        raise ValueError(f"invalid {digest_name}")


def freeze_model(
    *,
    manifest: dict[str, Any],
    training_inventory: dict[str, Any],
    training_rows: Path,
    training_validation: dict[str, Any],
    training_validation_sha256: str,
    v2_score: dict[str, Any],
    v2_score_sha256: str,
    holdout_plan: dict[str, Any],
) -> dict[str, Any]:
    """Freeze exact v2 medians as predictions for an independent rerun.

    Matching is by the complete requested-control object.  Cell ids differ in
    v3 because the role is new; matching by a subset of axes would accidentally
    turn this exact lookup into undocumented interpolation.
    """

    evidence = manifest["training_evidence"]
    validate_inventory(training_inventory)
    validate_inventory(holdout_plan)
    if training_inventory["inventory_digest"] != evidence["qualified_inventory_digest"]:
        raise ValueError("training inventory differs from the frozen v2 evidence")
    if training_validation_sha256 != evidence["validation_sha256"]:
        raise ValueError("training validation file differs from the freeze")
    if training_validation.get("observations_sha256") != evidence["observations_sha256"]:
        raise ValueError("training validation targets another observation capture")
    if training_validation.get("failures") or training_validation.get("void_scopes"):
        raise ValueError("v2 training evidence is not fully valid")
    if v2_score_sha256 != evidence["v2_score_sha256"]:
        raise ValueError("v2 score file differs from the freeze")
    _verify_digested(v2_score, "score_digest")
    if (
        v2_score.get("score_digest") != evidence["v2_score_digest"]
        or v2_score.get("model_digest") != evidence["v2_model_digest"]
        or v2_score.get("outcome") != evidence["v2_outcome"]
    ):
        raise ValueError("v2 score identity or outcome differs from the freeze")

    manifest_digest = content_digest(manifest)
    if holdout_plan.get("manifest_digest") != manifest_digest:
        raise ValueError("holdout plan targets another v3 manifest")

    training_by_request: dict[str, dict[str, Any]] = {}
    for cell in training_inventory["cells"]:
        key = _request_key(cell)
        if key in training_by_request:
            raise ValueError("v2 inventory contains duplicate physical requests")
        training_by_request[key] = cell

    # Reduce the opened v2 capture exactly once.  Five per-process medians are
    # retained so reviewers can see both the point prediction and its spread.
    training_cells = {cell["cell_id"]: cell for cell in training_inventory["cells"]}
    medians, observed_sha256 = _process_medians_from_jsonl(
        training_rows,
        cells=training_cells,
        timers=manifest["timing_boundaries"],
        expected_processes=int(manifest["ordinary_processes_per_cell"]),
        expected_repetitions=int(manifest["recorded_iterations"]),
        expected_sha256=evidence["observations_sha256"],
    )

    predictions = []
    unpredicted = []
    predicted_cells = []
    for cell in holdout_plan["cells"]:
        source = training_by_request.get(_request_key(cell))
        if source is None:
            unpredicted.append(
                {
                    "cell_id": cell["cell_id"],
                    "reason": "exact_requested_control_tuple_absent_from_v2_training",
                }
            )
            continue
        timer_predictions = []
        for timer in manifest["timing_boundaries"]:
            values = medians[(source["cell_id"], timer)]
            summary = _summary(values)
            timer_predictions.append(
                {
                    "timer": timer,
                    "predicted_ns": statistics.median(values),
                    "training_process_medians_ns": values,
                    "training_iqr_ns": summary["iqr_ns"],
                    "source_v2_cell_id": source["cell_id"],
                }
            )
        predictions.append({"cell_id": cell["cell_id"], "timers": timer_predictions})
        predicted_cells.append(cell)

    # This is both an implementation guard and a statement of scope: the exact
    # table covers the 112 qualified v2 controls, while the same 16 unsupported
    # rank-3 direct-read controls remain outside the model.
    if len(predictions) != 112 or len(unpredicted) != 16:
        raise ValueError("v3 exact-lookup coverage differs from the frozen contract")
    _rectangular_strata(
        predicted_cells,
        full_sms=int(manifest["model_spec"]["full_h100_sms"]),
    )

    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "manifest_digest": manifest_digest,
        "planned_inventory_digest": holdout_plan["inventory_digest"],
        "training": {
            "qualified_inventory_digest": training_inventory["inventory_digest"],
            "observations_sha256": observed_sha256,
            "validation_sha256": training_validation_sha256,
            "v2_score_digest": v2_score["score_digest"],
            "cell_count": len(training_cells),
        },
        "claim_scope": manifest["model_spec"]["claim_scope"],
        "holdout_predictions": predictions,
        "unpredicted_planned_cells": unpredicted,
    }
    model["model_digest"] = content_digest(model)
    return model


def fit_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    model = freeze_model(
        manifest=manifest,
        training_inventory=_read_json(args.training_inventory),
        training_rows=args.training_rows,
        training_validation=_read_json(args.training_validation),
        training_validation_sha256=_sha256(args.training_validation),
        v2_score=_read_json(args.v2_score),
        v2_score_sha256=_sha256(args.v2_score),
        holdout_plan=_read_json(args.holdout_plan),
    )
    _write_json(args.output, model)
    print(
        json.dumps(
            {
                "status": "complete",
                "model_digest": model["model_digest"],
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
    model = _read_json(args.model)
    validate_inventory(planned)
    validate_inventory(qualified)
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported v3 model schema")
    _verify_digested(model, "model_digest")

    # Reuse the already tested v2 contrast engine.  It consumes only frozen
    # predictions and ordinary rows; it has no fitting or lookup fallback.
    score = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=args.rows,
        validation=_read_json(args.validation),
        model=model,
    )
    score.pop("score_digest")
    score["schema"] = SCORE_SCHEMA
    score["claim_scope"] = model["claim_scope"]
    score["score_digest"] = content_digest(score)
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
    fit.add_argument("--training-inventory", type=Path, required=True)
    fit.add_argument("--training-rows", type=Path, required=True)
    fit.add_argument("--training-validation", type=Path, required=True)
    fit.add_argument("--v2-score", type=Path, required=True)
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
