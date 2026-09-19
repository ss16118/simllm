"""Contract tests for the frozen TRAF-94 v2 model and new holdout."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from examples.nccl_primitive_identification_v1.matrix import (
    canonical_json_bytes,
    capability_keys,
    content_digest,
    expand_manifest,
    load_manifest,
    validate_inventory,
)
from examples.nccl_primitive_interaction_v2.run_study import (
    AXES,
    _predict_from_stratum,
    score_model,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_primitive_interaction_v2"


def test_frozen_holdout_and_model_have_exact_reviewed_identities():
    manifest = load_manifest(STUDY / "manifest.json")
    cells = expand_manifest(manifest)
    plan = json.loads((STUDY / "holdout-plan.json").read_text(encoding="utf-8"))
    model = json.loads((STUDY / "h100-v2-model.json").read_text(encoding="utf-8"))

    validate_inventory(plan)
    assert len(cells) == plan["cell_count"] == 128
    assert len(capability_keys(cells)) == 32
    assert content_digest(manifest) == "2bf7455bf91bf6081b11ed82652e5258826cf59486c30e618cbcda2a7e662942"
    assert plan["inventory_digest"] == "0495e9583c5692ab229f3aa3bef93951a96a1a06ee7ddf40cc2ff6ce1a983cef"

    unsigned = dict(model)
    assert unsigned.pop("model_digest") == content_digest(unsigned)
    assert model["model_digest"] == "fba72bb7fe28e112cf4f2b73ca7381d5ea9ec9c55e29a1f4d1a4f6cfb1e15c0e"
    assert len(model["strata"]) == 12
    assert len(model["holdout_predictions"]) == 112
    assert len(model["unpredicted_planned_cells"]) == 16

    planned_by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    unpredicted = [
        planned_by_id[row["cell_id"]]
        for row in model["unpredicted_planned_cells"]
    ]
    assert all(cell["protocol"] == "SIMPLE" for cell in unpredicted)
    assert all(cell["simple_placement"] == "buffered_read" for cell in unpredicted)
    assert all(cell["requested"]["ranks"] == 3 for cell in unpredicted)

    # Every published prediction is a convex interpolation of named, measured
    # v1 anchors.  This also catches edited or incompletely serialized weights.
    for prediction in model["holdout_predictions"]:
        for timer in prediction["timers"]:
            assert sum(row["weight"] for row in timer["contributors"]) == pytest.approx(1.0)
            assert all(row["weight"] >= 0.0 for row in timer["contributors"])


def test_tensor_interpolation_retains_full_five_axis_interaction():
    anchors = []
    for coordinate in itertools.product((0.0, 1.0), repeat=len(AXES)):
        # A corner-only five-way interaction distinguishes tensor interpolation
        # from the additive v1 composition rule.
        value = 3.0 + sum(coordinate) + 10.0 * float(all(coordinate))
        anchors.append(
            {
                "cell_id": "corner-" + "".join(str(int(item)) for item in coordinate),
                "coordinate": list(coordinate),
                "summary": {"median_ns": value, "iqr_ns": 1.0},
            }
        )
    stratum = {
        "levels": {name: [0.0, 1.0] for name in AXES},
        "anchors": anchors,
    }

    predicted = _predict_from_stratum(stratum, [0.5] * len(AXES))
    assert predicted["predicted_ns"] == pytest.approx(3.0 + 2.5 + 10.0 / 32.0)
    assert predicted["propagated_anchor_iqr_ns"] == pytest.approx(1.0)
    assert len(predicted["contributors"]) == 32
    with pytest.raises(ValueError, match="outside"):
        _predict_from_stratum(stratum, [1.1, 0.5, 0.5, 0.5, 0.5])


def _score_fixture(tmp_path: Path):
    manifest = {
        "timing_boundaries": ["same_device_interval"],
        "ordinary_processes_per_cell": 2,
        "recorded_iterations": 2,
        "confirmation_error_bound": (
            "max(0.10*abs(measured_delta), frozen_repeat_spread_resolution)"
        ),
        "model_spec": {"full_h100_sms": 132},
    }
    base_request = {
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
        requested = {**base_request, "ranks": ranks}
        cells.append(
            {
                "cell_id": cell_id,
                "protocol": "LL",
                "simple_placement": "source_default",
                "requested": requested,
            }
        )
    planned = {"inventory_digest": "planned", "cells": cells}
    qualified = {
        "inventory_digest": "qualified",
        "manifest_digest": content_digest(manifest),
        "cells": copy.deepcopy(cells),
    }
    model = {
        "manifest_digest": content_digest(manifest),
        "planned_inventory_digest": "planned",
        "holdout_predictions": [
            {
                "cell_id": "a",
                "timers": [{"timer": "same_device_interval", "predicted_ns": 105.0}],
            },
            {
                "cell_id": "b",
                "timers": [{"timer": "same_device_interval", "predicted_ns": 135.0}],
            },
        ],
    }
    model["model_digest"] = content_digest(model)

    rows_path = tmp_path / "rows.jsonl"
    with rows_path.open("wb") as stream:
        for cell, base in ((cells[0], 100), (cells[1], 130)):
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
                                "raw_duration": base + 10 * process,
                            }
                        )
                    )
    validation = {
        "inventory_digest": "qualified",
        "observations_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "void_scopes": [],
        "failures": [],
    }
    return manifest, planned, qualified, rows_path, validation, model


def test_scorer_uses_only_the_predigested_prediction_and_frozen_rule(tmp_path):
    manifest, planned, qualified, rows_path, validation, model = _score_fixture(tmp_path)
    score = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=rows_path,
        validation=validation,
        model=model,
    )
    assert score["outcome"] == "PASS"
    assert score["summary"] == {
        "intervention_count": 1,
        "resolved_interventions": 1,
        "unresolved_interventions": 0,
        "accepted_resolved_interventions": 1,
        "rejected_resolved_interventions": 0,
        "has_resolved_interventions": True,
        "all_resolved_interventions_accepted": True,
        "acceptance_rule": manifest["confirmation_error_bound"],
    }

    # Changing a prediction is permitted here only to prove the scorer does not
    # refit it from observations; the changed model is re-digested and fails.
    wrong_model = copy.deepcopy(model)
    wrong_model["holdout_predictions"][1]["timers"][0]["predicted_ns"] = 50.0
    wrong_model.pop("model_digest")
    wrong_model["model_digest"] = content_digest(wrong_model)
    failed = score_model(
        manifest=manifest,
        planned_inventory=planned,
        qualified_inventory=qualified,
        rows_path=rows_path,
        validation=validation,
        model=wrong_model,
    )
    assert failed["outcome"] == "FAIL"
    assert failed["summary"]["rejected_resolved_interventions"] == 1
