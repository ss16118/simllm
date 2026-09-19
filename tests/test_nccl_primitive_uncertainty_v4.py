"""Contract tests for the frozen TRAF-94 v4 uncertainty model."""

import hashlib
import json
from pathlib import Path

from examples.nccl_primitive_identification_v1.matrix import (
    canonical_json_bytes,
    content_digest,
    load_manifest,
    validate_inventory,
)
from examples.nccl_primitive_uncertainty_v4.run_study import (
    _contrast_specs,
    score_model,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_primitive_uncertainty_v4"


def test_frozen_v4_artifacts_have_reviewed_identities_and_bounded_intervals():
    manifest = load_manifest(STUDY / "manifest.json")
    plan = json.loads((STUDY / "holdout-plan.json").read_text(encoding="utf-8"))
    model = json.loads((STUDY / "h100-v4-model.json").read_text(encoding="utf-8"))
    cohort = json.loads((STUDY / "device-cohort.json").read_text(encoding="utf-8"))

    validate_inventory(plan)
    assert content_digest(manifest) == (
        "e38e6b7c63823736e51b72a690c641f0e1199eb00a54508bcb4e950d48cd6620"
    )
    assert plan["inventory_digest"] == (
        "6c213b02ae9cb473aacaf91ca189860637d189b0e23a10de93d75cfab2439718"
    )
    assert plan["cell_count"] == 128

    unsigned_cohort = dict(cohort)
    assert unsigned_cohort.pop("cohort_digest") == content_digest(unsigned_cohort)
    assert cohort["cohort_digest"] == manifest["device_cohort"]["cohort_digest"]
    assert len(cohort["physical_devices"]) == cohort["maximum_ranks"] == 3

    unsigned_model = dict(model)
    assert unsigned_model.pop("model_digest") == content_digest(unsigned_model)
    assert model["model_digest"] == (
        "430d2752bcd5e3471d9aeabfa08764985e8d56dd013b1189490220f28f694895"
    )
    assert len(model["contrast_predictions"]) == 816
    assert len(model["calibration_groups"]) == 57
    assert sum(row["status"] == "predict" for row in model["calibration_groups"]) == 49
    assert sum(row["status"] == "abstain" for row in model["calibration_groups"]) == 8

    for row in model["contrast_predictions"]:
        assert row["lower_ns"] <= row["predicted_delta_ns"] <= row["upper_ns"]
        assert row["half_width_ns"] >= row["resolution_floor_ns"]
        if row["status"] == "predict":
            assert row["group_quantile"] <= 1.0
            assert row["normalized_half_width"] <= 1.0
        else:
            assert row["group_quantile"] > 1.0


def test_v4_scorer_applies_frozen_interval_retention_and_coverage(tmp_path):
    manifest = {
        "timing_boundaries": ["same_device_interval"],
        "ordinary_processes_per_cell": 2,
        "recorded_iterations": 2,
        "model_spec": {"full_h100_sms": 132},
        "device_cohort": {"cohort_digest": "cohort"},
        "holdout_contract": {
            "minimum_resolved_interventions": 1,
            "minimum_resolved_retention": 0.90,
            "minimum_overall_interval_coverage": 0.90,
            "minimum_per_timer_interval_coverage": 0.85,
            "maximum_normalized_half_width": 1.0,
        },
    }
    base = {
        "stage": "confirmation",
        "architecture": "h100",
        "protocol": "LL",
        "simple_placement": "source_default",
        "ranks": 2,
        "active_channels": 4,
        "available_sms": 32,
        "working_warps": 8,
        "useful_bytes_per_channel": 512,
        "delay_cycles": 768,
    }
    cells = []
    for cell_id, ranks in (("a", 2), ("b", 3)):
        cells.append(
            {
                "cell_id": cell_id,
                "protocol": "LL",
                "simple_placement": "source_default",
                "requested": {**base, "ranks": ranks},
            }
        )
    planned = {"inventory_digest": "planned", "cells": cells}
    qualified = {
        "inventory_digest": "qualified",
        "manifest_digest": content_digest(manifest),
        "cells": cells,
    }
    spec = _contrast_specs(
        cells, timers=manifest["timing_boundaries"], full_sms=132
    )[0]
    prediction = {
        **spec,
        "status": "predict",
        "predicted_delta_ns": 30.0,
        "lower_ns": 20.0,
        "upper_ns": 40.0,
        "normalized_half_width": 0.5,
    }
    model = {
        "manifest_digest": content_digest(manifest),
        "planned_inventory_digest": "planned",
        "device_cohort_digest": "cohort",
        "model_digest": "model",
        "contrast_predictions": [prediction],
    }

    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("wb") as stream:
        for cell, base_time in ((cells[0], 100), (cells[1], 130)):
            for process in range(2):
                for repetition in range(2):
                    stream.write(
                        canonical_json_bytes(
                            {
                                "cell_id": cell["cell_id"],
                                "requested": cell["requested"],
                                "timer": "same_device_interval",
                                "process_id": process,
                                "repetition": repetition,
                                "phase": "ordinary",
                                "qualification": "qualified",
                                "units": "ns",
                                "raw_duration": base_time + process,
                            }
                        )
                    )
    validation = {
        "inventory_digest": "qualified",
        "observations_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "failures": [],
        "void_scopes": [],
    }
    score = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=rows_path,
        validation=validation,
        model=model,
    )
    assert score["outcome"] == "PASS"
    assert score["summary"]["resolved_interventions"] == 1
    assert score["summary"]["predicted_resolved_interventions"] == 1
    assert score["summary"]["interval_coverage"] == 1.0
    assert all(score["summary"]["acceptance_gates"].values())
