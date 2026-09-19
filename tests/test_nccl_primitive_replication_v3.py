"""Contract tests for the frozen TRAF-94 v3 replication table."""

import json
import statistics
from pathlib import Path

from examples.nccl_primitive_identification_v1.matrix import (
    capability_keys,
    content_digest,
    expand_manifest,
    load_manifest,
    validate_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_primitive_replication_v3"


def test_frozen_replication_plan_and_table_have_reviewed_identities():
    manifest = load_manifest(STUDY / "manifest.json")
    cells = expand_manifest(manifest)
    plan = json.loads((STUDY / "holdout-plan.json").read_text(encoding="utf-8"))
    model = json.loads((STUDY / "h100-v3-model.json").read_text(encoding="utf-8"))

    validate_inventory(plan)
    assert content_digest(manifest) == (
        "175bc6541ed1e524d84f0cbdb39a2080691ad14da4a7e495ac40acc9dd7b3cd4"
    )
    assert plan["inventory_digest"] == (
        "1eb571ec187ccfe15428565f3efe6ecf8eeca59d4a4526d83f596816bc1d959d"
    )
    assert len(cells) == plan["cell_count"] == 128
    assert len(capability_keys(cells)) == 32

    unsigned = dict(model)
    assert unsigned.pop("model_digest") == content_digest(unsigned)
    assert model["model_digest"] == (
        "3135bad0b2b3d2da74323cbd7fceedefe292be0281fcadd1c4df3fbfff194b6e"
    )
    assert len(model["holdout_predictions"]) == 112
    assert len(model["unpredicted_planned_cells"]) == 16
    assert model["claim_scope"] == "same_node_same_probe_same_control_repeatability"


def test_every_prediction_is_an_exact_v2_process_median_lookup():
    model = json.loads((STUDY / "h100-v3-model.json").read_text(encoding="utf-8"))

    for cell in model["holdout_predictions"]:
        assert len(cell["timers"]) == 3
        for timer in cell["timers"]:
            values = timer["training_process_medians_ns"]
            assert len(values) == 5
            assert timer["predicted_ns"] == statistics.median(values)
            assert timer["source_v2_cell_id"].startswith("confirmation-")

    # The only absent lookup tuples are the exact source configurations that
    # failed v2 capability correctness.  V3 cannot invent values for them.
    plan = json.loads((STUDY / "holdout-plan.json").read_text(encoding="utf-8"))
    by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    absent = [by_id[row["cell_id"]] for row in model["unpredicted_planned_cells"]]
    assert all(cell["protocol"] == "SIMPLE" for cell in absent)
    assert all(cell["simple_placement"] == "buffered_read" for cell in absent)
    assert all(cell["requested"]["ranks"] == 3 for cell in absent)
