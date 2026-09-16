"""Contract tests for the TRAF-94 matrix, evidence gates, and statistics."""

import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_primitive_identification_v1"

from examples.nccl_primitive_identification_v1.analysis import (
    fit_identification,
    paired_contrast,
    predict_confirmation_cell,
    score_confirmation,
    score_confirmations,
)
from examples.nccl_primitive_identification_v1.matrix import (
    CAPABILITY_SCHEMA,
    OBSERVATION_SCHEMA,
    REQUIRED_SOURCE_COUNTS,
    capability_keys,
    completeness_failures,
    content_digest,
    expand_manifest,
    inventory_document,
    load_manifest,
    qualify_inventory,
    timing_scope_voids,
    validate_inventory,
    validate_observation,
)
from examples.nccl_primitive_identification_v1.probe import (
    _native_arguments,
    _qualification_failures,
)
from examples.nccl_primitive_identification_v1.publish_results import (
    build_result,
    render_markdown,
)
from examples.nccl_primitive_identification_v1.run_campaign import (
    _completed_indices,
    _compute_apps,
    _device_uuids,
    _wait_for_idle,
)
from examples.nccl_primitive_identification_v1.run_study import _schedule


@pytest.fixture
def manifest():
    return load_manifest(STUDY / "manifest.json")


def test_h100_expansion_is_sequential_deterministic_and_source_explicit(manifest):
    first = expand_manifest(manifest, architectures=("h100",))
    second = expand_manifest(copy.deepcopy(manifest), architectures=("h100",))

    assert first == second
    assert len(first) == 848
    # Capability identities include every support-sensitive control.  A single
    # easy representative must not qualify untested channel/warp/SM geometry.
    assert len(capability_keys(first)) == 388
    assert Counter(cell.stage for cell in first) == {
        "ready_publication": 96,
        "reuse": 160,
        "data_work": 112,
        "sharing": 288,
        "confirmation": 192,
    }
    assert {cell.family for cell in first if cell.stage == "confirmation"} == {"confirmation"}
    assert all(isinstance(cell.family, str) for cell in first)
    assert all(cell.requested["stage"] == cell.stage for cell in first)
    assert all(
        "useful_bytes" not in cell.requested
        for cell in first
        if "useful_bytes_per_channel" in cell.requested
    )
    assert {cell.simple_placement for cell in first if cell.protocol == "SIMPLE"} == {
        "buffered",
        "buffered_read",
    }
    assert {cell.simple_placement for cell in first if cell.protocol != "SIMPLE"} == {
        "source_default"
    }
    # A ready-publication cell must not accidentally inherit the reuse stage's
    # reservation axis.  This is the regression that a global Cartesian product
    # would trigger.
    assert all(
        "reservations" not in cell.requested for cell in first if cell.stage == "ready_publication"
    )


def test_manifest_and_inventory_digests_reject_semantic_edits(manifest):
    assert (
        content_digest(manifest)
        == "4427e867baae52901e0801822f687a53c99ffb8a6d1e0763e6a1aaeafdd809c2"
    )
    cells = expand_manifest(manifest, architectures=("h100",))
    inventory = inventory_document(manifest, cells, state="planned")
    validate_inventory(inventory)

    changed = copy.deepcopy(inventory)
    changed["cells"][0]["requested"]["useful_bytes"] = 123
    with pytest.raises(ValueError, match="digest"):
        validate_inventory(changed)


def _capability_rows(cells, *, failed_key=None, identity_digest="1" * 64):
    for request in capability_keys(cells):
        failed = request["capability_key"] == failed_key
        yield {
            "schema": CAPABILITY_SCHEMA,
            "capability_key": request["capability_key"],
            "qualified": not failed,
            "reason": "source_specialization_unsupported" if failed else "qualified",
            "identity_digest": identity_digest,
            "realized": {"working_warps": 5},
            "source_operation_counts": {
                name: int(name == "readiness_checks") for name in REQUIRED_SOURCE_COUNTS
            },
        }


def test_capability_freeze_retains_failure_and_excludes_affected_cells(manifest):
    cells = expand_manifest(manifest, architectures=("h100",))
    failed_key = capability_keys(cells)[0]["capability_key"]
    affected = sum(cell.capability_key == failed_key for cell in cells)
    qualified = qualify_inventory(
        manifest,
        cells,
        _capability_rows(cells, failed_key=failed_key),
    )

    validate_inventory(qualified)
    assert qualified["state"] == "capability_qualified"
    assert qualified["cell_count"] == len(cells) - affected
    failure = next(
        row for row in qualified["capability_outcomes"] if row["capability_key"] == failed_key
    )
    assert failure["qualified"] is False
    assert failure["reason"] == "source_specialization_unsupported"


def _small_contract(manifest):
    small = copy.deepcopy(manifest)
    small["architectures"] = ["h100"]
    small["protocols"] = ["LL", "LL128", "SIMPLE"]
    small["stages"] = [{"id": "data_work", "family": ["copy"], "useful_bytes": [16]}]
    small["ordinary_processes_per_cell"] = 2
    small["recorded_iterations"] = 2
    small["timing_boundaries"] = ["same_device_interval", "cuda_event"]
    return small


def _observation(manifest, inventory, cell, process, repetition, timer):
    return {
        "schema": OBSERVATION_SCHEMA,
        "manifest_digest": inventory["manifest_digest"],
        "cell_id": cell["cell_id"],
        "process_id": process,
        "repetition": repetition,
        "identity_digest": "a" * 64,
        "requested": cell["requested"],
        "realized": dict(cell["requested"]),
        "family": cell["family"],
        "phase": "ordinary",
        "timer": timer,
        "units": {
            "same_device_interval": "ns",
            "cuda_event": "us",
            "host_wall": "us",
        }[timer],
        "raw_duration": 12.5,
        "observed_local_delay": 0.0,
        "correctness": True,
        "qualification": "qualified",
        "reason": "",
    }


def test_observation_guards_and_full_completeness(manifest):
    small = _small_contract(manifest)
    cells = expand_manifest(small, architectures=("h100",))
    inventory = inventory_document(small, cells, state="capability_qualified")
    rows = [
        _observation(small, inventory, cell, process, repetition, timer)
        for cell in inventory["cells"]
        for process in range(2)
        for repetition in range(2)
        for timer in small["timing_boundaries"]
    ]
    assert completeness_failures(rows, small, inventory) == ()

    changed = copy.deepcopy(rows[0])
    changed["requested"]["useful_bytes"] = 17
    with pytest.raises(ValueError, match="changed after freeze"):
        validate_observation(changed, small, inventory)
    assert "missing 1 required rows" in completeness_failures(rows[1:], small, inventory)[0]


def test_timer_units_and_causal_scope_void_preserve_other_boundaries(manifest):
    small = _small_contract(manifest)
    expanded = expand_manifest(small, architectures=("h100",))[0]
    inventory = qualify_inventory(small, (expanded,), _capability_rows((expanded,)))
    cell = inventory["cells"][0]
    rows = [
        _observation(small, inventory, cell, 0, 0, timer)
        for timer in small["timing_boundaries"]
    ]
    by_timer = {row["timer"]: row for row in rows}
    by_timer["same_device_interval"]["raw_duration"] = 1_000_000_000.0
    by_timer["cuda_event"]["raw_duration"] = 10.0

    voids = timing_scope_voids(rows)
    assert voids == (
        {
            "cell_id": cell["cell_id"],
            "timer": "same_device_interval",
            "reason": "inner_same_device_interval_exceeds_outer_cuda_event",
            "violating_rows": 1,
            "process_ids": [0],
            "maximum_inner_ns": 1_000_000_000.0,
            "maximum_outer_ns": 10_000.0,
        },
    )
    assert {row["timer"] for row in rows} - {voids[0]["timer"]} == {"cuda_event"}

    wrong_unit = copy.deepcopy(by_timer["cuda_event"])
    wrong_unit["units"] = "ns"
    with pytest.raises(ValueError, match="requires unit 'us'"):
        validate_observation(wrong_unit, small, inventory)


def test_validate_cli_retains_rows_and_reports_complete_timer_void_scope(
    tmp_path, manifest
):
    small = _small_contract(manifest)
    # Keep the CLI fixture to one cell while retaining the frozen two-process,
    # two-repetition, two-timer completeness dimensions.
    expanded = expand_manifest(small, architectures=("h100",))[0]
    inventory = qualify_inventory(small, (expanded,), _capability_rows((expanded,)))
    cell = inventory["cells"][0]
    rows = [
        _observation(small, inventory, cell, process, repetition, timer)
        for process in range(small["ordinary_processes_per_cell"])
        for repetition in range(small["recorded_iterations"])
        for timer in small["timing_boundaries"]
    ]
    # One causal violation voids the entire cell/timer scope. The other seven
    # raw rows remain present, and the CUDA-event scope stays eligible.
    rows[0]["raw_duration"] = 1_000_000_000.0
    manifest_path = tmp_path / "manifest.json"
    inventory_path = tmp_path / "inventory.json"
    rows_path = tmp_path / "rows.jsonl"
    report_path = tmp_path / "validation.json"
    manifest_path.write_text(json.dumps(small), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "validate",
            "--manifest",
            str(manifest_path),
            "--inventory",
            str(inventory_path),
            "--rows",
            str(rows_path),
            "--output",
            str(report_path),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text())
    assert report["fatal_guards"] == "valid_outside_void_scopes"
    assert report["row_count"] == len(rows)
    assert report["failures"] == []
    assert report["void_scopes"] == [
        {
            "cell_id": cell["cell_id"],
            "timer": "same_device_interval",
            "reason": "inner_same_device_interval_exceeds_outer_cuda_event",
            "violating_rows": 1,
            "process_ids": [0],
            "maximum_inner_ns": 1_000_000_000.0,
            "maximum_outer_ns": 12_500.0,
        }
    ]


def test_paired_iqr_and_heldout_rules_use_process_summaries():
    rows = []
    for process in range(5):
        for repetition in range(3):
            for cell, duration in (("a", 100 + process), ("b", 130 + process)):
                rows.append(
                    {
                        "cell_id": cell,
                        "process_id": process,
                        "repetition": repetition,
                        "timer": "cuda_event",
                        "phase": "ordinary",
                        "qualification": "qualified",
                        "raw_duration": duration,
                    }
                )
    contrast = paired_contrast(
        rows,
        cell_a="a",
        cell_b="b",
        timer="cuda_event",
        expected_processes=5,
    )
    assert contrast["paired_median_delta"] == 30
    assert contrast["resolved"] is True
    assert contrast["interpretation"] == "separate_term"

    assert (
        score_confirmation(
            measured_delta=20,
            predicted_delta=18,
            frozen_repeat_spread_resolution=1,
        )["accepted"]
        is True
    )
    assert (
        score_confirmation(
            measured_delta=20,
            predicted_delta=17,
            frozen_repeat_spread_resolution=1,
        )["accepted"]
        is False
    )


def test_identification_fit_locks_before_confirmation_scoring():
    """Exercise the two-step fit/score boundary with a compact synthetic grid."""

    def cell(identity, stage, family, **controls):
        request = {
            "stage": stage,
            "protocol": "LL",
            "simple_placement": "source_default",
            "ranks": 2,
            "active_channels": 1,
            "available_sms": 16,
            "working_warps": 8,
            **controls,
        }
        return {
            "cell_id": identity,
            "stage": stage,
            "family": family,
            "protocol": "LL",
            "simple_placement": "source_default",
            "requested": request,
        }

    cells = []
    for delay in (0, 4096):
        cells.extend(
            [
                cell(
                    f"ready-{delay}-a",
                    "ready_publication",
                    "already_ready",
                    delay_cycles=delay,
                    useful_bytes=16,
                ),
                cell(
                    f"ready-{delay}-b",
                    "ready_publication",
                    "delayed_publication",
                    delay_cycles=delay,
                    useful_bytes=16,
                ),
            ]
        )
    for useful_bytes in (0, 1920):
        cells.append(
            cell(
                f"data-{useful_bytes}",
                "data_work",
                "sum",
                useful_bytes=useful_bytes,
                working_set="reused",
            )
        )
    for channels in (1, 8, 32):
        cells.append(
            cell(
                f"sharing-{channels}",
                "sharing",
                "sum",
                active_channels=channels,
                useful_bytes_per_channel=1920,
            )
        )
    confirmation = [
        cell(
            f"confirmation-{delay}",
            "confirmation",
            "confirmation",
            active_channels=2,
            useful_bytes_per_channel=1924,
            delay_cycles=delay,
        )
        for delay in (512, 2048)
    ]
    inventory = {
        "manifest_digest": "1" * 64,
        "inventory_digest": "2" * 64,
        "cells": cells + confirmation,
    }
    manifest = {
        "ordinary_processes_per_cell": 5,
        "timing_boundaries": ["same_device_interval"],
        "resolved_contrast": "abs(paired_median_delta) > 2*(IQR_A+IQR_B)",
        "confirmation_error_bound": (
            "max(0.10*abs(measured_delta), frozen_repeat_spread_resolution)"
        ),
    }

    durations = {}
    for item in cells:
        if item["stage"] == "ready_publication":
            delay = item["requested"]["delay_cycles"]
            durations[item["cell_id"]] = 1_000 + (
                2 * delay if item["family"] == "delayed_publication" else 0
            )
        elif item["stage"] == "data_work":
            durations[item["cell_id"]] = 2_000 + item["requested"]["useful_bytes"]
        else:
            durations[item["cell_id"]] = 4_000 + 1_000 * item["requested"]["active_channels"]

    def synthetic_rows(selected, values):
        result = []
        for item in selected:
            for process in range(5):
                for repetition in range(3):
                    result.append(
                        {
                            "cell_id": item["cell_id"],
                            "process_id": process,
                            "repetition": repetition,
                            "timer": "same_device_interval",
                            "units": "ns",
                            "raw_duration": values[item["cell_id"]] + process,
                            "phase": "ordinary",
                            "qualification": "qualified",
                        }
                    )
        return result

    identification_rows = synthetic_rows(cells, durations)
    fit = fit_identification(
        identification_rows,
        inventory=inventory,
        manifest=manifest,
        observations_sha256="3" * 64,
    )
    assert fit["confirmation_rows_used"] == 0
    assert fit["fit_scope"] == "identification_cells_only"
    assert fit["publication_response"]

    predictions = {
        item["cell_id"]: predict_confirmation_cell(
            fit, item, timer="same_device_interval"
        )["predicted_ns"]
        for item in confirmation
    }
    all_rows = identification_rows + synthetic_rows(confirmation, predictions)
    score = score_confirmations(
        all_rows,
        inventory=inventory,
        manifest=manifest,
        fit=fit,
        observations_sha256="3" * 64,
    )
    assert score["summary"]["resolved_interventions"] == 1
    assert score["summary"]["accepted_resolved_interventions"] == 1
    assert score["summary"]["all_resolved_interventions_accepted"] is True


def test_compact_result_publisher_cross_binds_fit_score_and_raw_evidence():
    manifest_digest = "1" * 64
    inventory_digest = "2" * 64
    observations_sha256 = "3" * 64
    fit = {
        "schema": "simllm-nccl-primitive-identification-fit-v1",
        "manifest_digest": manifest_digest,
        "inventory_digest": inventory_digest,
        "observations_sha256": observations_sha256,
        "fit_scope": "identification_cells_only",
        "confirmation_rows_used": 0,
        "void_scopes": [],
        "anchors": [
            {
                "cell_id": "data",
                "stage": "data_work",
                "timer": "cuda_event",
                "protocol": "LL",
                "simple_placement": "source_default",
                "family": "sum",
                "requested": {"useful_bytes": 1920, "working_set": "reused"},
                "summary": {
                    "process_medians_ns": [100, 101, 102, 103, 104],
                    "median_ns": 102,
                    "q1_ns": 101,
                    "q3_ns": 103,
                    "iqr_ns": 2,
                },
            }
        ],
        "paired_contrasts": [
            {
                "stage": "data_work",
                "timer": "cuda_event",
                "protocol": "LL",
                "simple_placement": "source_default",
                "factor": "reduction",
                "paired_median_delta": 7,
                "resolved": True,
            }
        ],
        "publication_response": [],
        "separability": {
            "contrast_count": 1,
            "resolved_separate_terms": 1,
            "joint_intervals": 0,
            "rule": "frozen",
        },
        "confirmation_model": {"kind": "test"},
    }
    fit["fit_digest"] = content_digest(fit)
    scored = {
        "timer": "cuda_event",
        "protocol": "LL",
        "simple_placement": "source_default",
        "factor": "delay_cycles",
        "acceptance_applicable": True,
        "accepted_resolved_delta": True,
        "absolute_error": 1,
        "acceptance_bound": 2,
    }
    score = {
        "schema": "simllm-nccl-primitive-confirmation-score-v1",
        "manifest_digest": manifest_digest,
        "inventory_digest": inventory_digest,
        "observations_sha256": observations_sha256,
        "fit_digest": fit["fit_digest"],
        "void_scopes": [],
        "scored_interventions": [scored],
        "summary": {
            "intervention_count": 1,
            "resolved_interventions": 1,
            "unresolved_interventions": 0,
            "accepted_resolved_interventions": 1,
            "rejected_resolved_interventions": 0,
            "has_resolved_interventions": True,
            "all_resolved_interventions_accepted": True,
            "acceptance_rule": "frozen",
        },
    }
    score["score_digest"] = content_digest(score)
    validation = {
        "schema": "simllm-nccl-primitive-validation-v1",
        "manifest_digest": manifest_digest,
        "inventory_digest": inventory_digest,
        "observations_sha256": observations_sha256,
        "row_count": 300,
        "cell_count": 1,
        "fatal_guards": "valid",
        "void_scopes": [],
        "failures": [],
    }
    merged = {
        "schema": "simllm-nccl-primitive-merged-artifacts-v1",
        "manifest_digest": manifest_digest,
        "inventory_digest": inventory_digest,
        "observations_sha256": observations_sha256,
        "row_count": 300,
        "work_manifests": [{"work_index": 0}],
    }

    result = build_result(
        validation=validation,
        fit=fit,
        score=score,
        merged_artifacts=merged,
        input_hashes={name: name[0] * 64 for name in ("validation", "fit", "score")},
    )
    assert result["outcome"] == "PASS"
    assert result["identification"]["published_component_anchors"][0]["median_ns"] == 102
    assert result["confirmation"]["groups"][0]["worst_error_over_bound"] == 0.5
    assert "Outcome: **PASS**" in render_markdown(result)
    unsigned = dict(result)
    assert unsigned.pop("result_digest") == content_digest(unsigned)

    changed = copy.deepcopy(score)
    changed["observations_sha256"] = "4" * 64
    changed.pop("score_digest")
    changed["score_digest"] = content_digest(changed)
    with pytest.raises(ValueError, match="does not match validation"):
        build_result(
            validation=validation,
            fit=fit,
            score=changed,
            merged_artifacts=merged,
            input_hashes={},
        )


def test_cli_writes_h100_plan_and_minimal_capability_plan(tmp_path):
    inventory = tmp_path / "planned.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "plan",
            "--architecture-extension",
            str(STUDY / "h100-extension.json"),
            "--output",
            str(inventory),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(inventory.read_text())["cell_count"] == 848

    capabilities = tmp_path / "capabilities.jsonl"
    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "capability-plan",
            "--inventory",
            str(inventory),
            "--output",
            str(capabilities),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    requests = [json.loads(line) for line in capabilities.read_text().splitlines()]
    assert len(requests) == 388
    assert len({row["capability_key"] for row in requests}) == 388
    assert all(isinstance(row["requested"], dict) for row in requests)
    assert all(row["requested"]["stage"] == row["stage"] for row in requests)
    # Group-equivalent delay cells use the strongest member as their pilot.
    assert max(row["requested"].get("delay_cycles", 0) for row in requests) == 4096


def test_native_request_carries_stage_and_empty_direct_read_uses_placement_evidence(
    manifest,
):
    cell = next(
        cell
        for cell in expand_manifest(manifest, architectures=("h100",))
        if cell.stage == "ready_publication"
        and cell.protocol == "SIMPLE"
        and cell.simple_placement == "buffered_read"
        and cell.requested["useful_bytes"] == 0
        and cell.family == "already_ready"
    )
    arguments = _native_arguments(
        {"native_executable": "/tmp/not-run"},
        family=cell.family,
        requested=cell.requested,
        diagnostic=True,
        warmups=2,
        iterations=1,
    )
    assert arguments[arguments.index("--stage") + 1] == "ready_publication"

    sharing = next(
        candidate
        for candidate in expand_manifest(manifest, architectures=("h100",))
        if candidate.stage == "sharing"
        and candidate.protocol == "LL"
        and candidate.requested["active_channels"] == 8
    )
    sharing_arguments = _native_arguments(
        {"native_executable": "/tmp/not-run"},
        family=sharing.family,
        requested=sharing.requested,
        diagnostic=True,
        warmups=2,
        iterations=1,
    )
    assert sharing_arguments[sharing_arguments.index("--useful-bytes") + 1] == str(
        sharing.requested["useful_bytes_per_channel"] * 8
    )

    working_threads = 32 * cell.requested["working_warps"]
    native = {
        "correctness": True,
        "canary_ok": True,
        "unsupported_site": 0,
        "samples": {
            "same_device_interval": [1],
            "observed_local_delay": [0],
        },
        "counts": {
            name: int(
                name
                in {
                    "readiness_checks",
                    "successful_observations",
                    "consumed_steps",
                    "head_publications",
                }
            )
            for name in REQUIRED_SOURCE_COUNTS
        }
        | {"unsuccessful_checks": 0},
        "realized": {
            "protocols": [3, 3],
            "working_warps": [cell.requested["working_warps"]] * 2,
            "block_threads": [working_threads + 32, working_threads],
            "active_channels": [1, 1],
            "channel_masks": [1, 1],
            "placement_masks": [1 << 2, 1 << 2],
            "recv_peers": [[-1] * 32, [0] + [-1] * 31],
            "send_peers": [[1] + [-1] * 31, [-1] * 32],
            "granted_sms": [132, 132],
            "resident_sms": [1, 1],
            "input_pointer_distinct": [1, 1],
            "input_pointer_span_bytes": [0, 0],
        },
    }
    # Empty work must prove DirectRead from source flags while all payload-byte
    # counters remain zero; demanding remote_read_bytes > 0 would contradict
    # the empty/nonempty intervention itself.
    assert _qualification_failures(cell.family, cell.requested, native, require_counts=True) == []


def test_schedule_keeps_matched_families_adjacent_and_counterbalanced(manifest):
    small = _small_contract(manifest)
    small["stages"] = [
        {
            "id": "data_work",
            "family": ["copy", "sum"],
            "useful_bytes": [16, 32],
        }
    ]
    cells = expand_manifest(small, architectures=("h100",))
    inventory = inventory_document(small, cells, state="capability_qualified")
    first = _schedule(inventory, small)
    assert first == _schedule(inventory, copy.deepcopy(small))

    for process in range(small["ordinary_processes_per_cell"]):
        work = [row for row in first if row["process_id"] == process]
        assert len(work) == len(cells)
        assert len({row["cell_id"] for row in work}) == len(cells)
        position = {row["cell_id"]: row["order_position"] for row in work}
        for cell in cells:
            partner = next(
                candidate
                for candidate in cells
                if candidate.family != cell.family
                and candidate.protocol == cell.protocol
                and candidate.simple_placement == cell.simple_placement
                and candidate.requested == cell.requested
            )
            assert abs(position[cell.cell_id] - position[partner.cell_id]) == 1
            assert work[position[cell.cell_id]]["pair_order"] == (
                "AB" if process % 2 == 0 else "BA"
            )


def test_campaign_resolves_physical_devices_before_cuda_visibility(monkeypatch):
    """The idle gate must monitor host UUIDs, not remapped CUDA ordinals."""

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=("nvidia-smi",),
            returncode=0,
            stdout="0, GPU-a\n2, GPU-c\n4, GPU-e\n5, GPU-f\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _device_uuids((0, 2, 4, 5)) == frozenset(
        {"GPU-a", "GPU-c", "GPU-e", "GPU-f"}
    )
    with pytest.raises(ValueError, match="do not exist"):
        _device_uuids((0, 7))


def test_campaign_compute_app_filter_ignores_unselected_gpus(monkeypatch):
    """Unrelated contexts are reported only when they touch selected GPUs."""

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=("nvidia-smi",),
            returncode=0,
            stdout=(
                "GPU-selected, 41, python, 1024\n"
                "GPU-other, 42, another process, 2048\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _compute_apps(frozenset({"GPU-selected"})) == [
        {
            "gpu_uuid": "GPU-selected",
            "pid": 41,
            "process_name": "python",
            "used_gpu_memory_mib": 1024,
        }
    ]


def test_campaign_idle_gate_requires_consecutive_empty_samples(monkeypatch):
    """A single scheduling gap must not be mistaken for a stable idle node."""

    samples = iter(
        [
            [{"gpu_uuid": "GPU-a", "pid": 1}],
            [],
            [{"gpu_uuid": "GPU-a", "pid": 2}],
            [],
            [],
            [],
        ]
    )
    sleeps = []
    monkeypatch.setattr(
        "examples.nccl_primitive_identification_v1.run_campaign._compute_apps",
        lambda _uuids: next(samples),
    )
    monkeypatch.setattr(
        "examples.nccl_primitive_identification_v1.run_campaign.time.sleep",
        sleeps.append,
    )

    _wait_for_idle(
        monitored_uuids=frozenset({"GPU-a"}),
        consecutive_samples=3,
        poll_seconds=0.25,
    )

    # Six samples require five waits; the final qualifying sample exits without
    # adding an unnecessary delay before the campaign launches its work item.
    assert sleeps == [0.25] * 5


def test_campaign_resume_accepts_only_hash_verified_work(tmp_path):
    schedule = {"work": [{"work_index": 0}, {"work_index": 1}]}
    work = tmp_path / "work-000000"
    work.mkdir()
    payload = b"evidence\n"
    (work / "observations.jsonl").write_bytes(payload)
    artifacts = {
        "schema": "simllm-nccl-primitive-artifact-manifest-v1",
        "artifacts": [
            {
                "path": "observations.jsonl",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    artifacts_bytes = (json.dumps(artifacts, indent=2, sort_keys=True) + "\n").encode()
    (work / "artifacts.json").write_bytes(artifacts_bytes)
    (work / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(artifacts_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert _completed_indices(schedule, tmp_path) == {0}
    (work / "observations.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after completion"):
        _completed_indices(schedule, tmp_path)

    # Restore the hashed payload, then prove that an explicit post-run
    # contention disposition overrides otherwise complete immutable evidence.
    (work / "observations.jsonl").write_bytes(payload)
    (work / "CONTAMINATED.json").write_text(
        json.dumps(
            {
                "schema": "simllm-nccl-primitive-contamination-v1",
                "status": "void_external_context_at_post_run_boundary",
                "work_index": 0,
                "applications": [{"gpu_uuid": "GPU-a", "pid": 99}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="external-contention marker"):
        _completed_indices(schedule, tmp_path)


def test_mock_probe_exercises_process_artifacts_collection_and_validation(tmp_path, manifest):
    """Exercise orchestration only; the mock deliberately makes no GPU claim."""

    small = _small_contract(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(small), encoding="utf-8")
    cells = expand_manifest(small, architectures=("h100",))
    mock_identity = {
        "schema": "simllm-nccl-primitive-probe-identity-v1",
        "source_commit": "7b83616df3ae082a1f32bb74c27458bfe8153a13",
        "architecture": "h100",
        "build": {
            "kind": "test-only-mock",
            "executable_sha256": "0" * 64,
            "nccl_library_sha256": "1" * 64,
        },
        "device": {
            "kind": "test-only-mock",
            "uuids": ["mock-gpu-0", "mock-gpu-1"],
            "driver": "mock",
            "topology_digest": "2" * 64,
        },
        "source_files": [{"path": "mock", "sha256": "3" * 64}],
    }
    mock_identity["identity_digest"] = content_digest(mock_identity)
    inventory = qualify_inventory(
        small,
        cells,
        _capability_rows(cells, identity_digest=mock_identity["identity_digest"]),
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    schedule_path = tmp_path / "schedule.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "schedule",
            "--manifest",
            str(manifest_path),
            "--inventory",
            str(inventory_path),
            "--output",
            str(schedule_path),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    schedule = json.loads(schedule_path.read_text())

    probe = tmp_path / "mock-probe"
    probe.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import sys

def digest(value):
    data = (json.dumps(value, sort_keys=True, separators=(\",\", \":\")) + \"\\n\").encode()
    return hashlib.sha256(data).hexdigest()

identity = {
    \"schema\": \"simllm-nccl-primitive-probe-identity-v1\",
    \"source_commit\": \"7b83616df3ae082a1f32bb74c27458bfe8153a13\",
    \"architecture\": \"h100\",
    \"build\": {
        \"kind\": \"test-only-mock\",
        \"executable_sha256\": \"0\" * 64,
        \"nccl_library_sha256\": \"1\" * 64,
    },
    \"device\": {
        \"kind\": \"test-only-mock\",
        \"uuids\": [\"mock-gpu-0\", \"mock-gpu-1\"],
        \"driver\": \"mock\",
        \"topology_digest\": \"2\" * 64,
    },
    \"source_files\": [{\"path\": \"mock\", \"sha256\": \"3\" * 64}],
}
identity[\"identity_digest\"] = digest(identity)
if sys.argv[1] == \"--describe\":
    print(json.dumps(identity))
elif sys.argv[1] == \"--run\":
    request = json.load(open(sys.argv[2], encoding=\"utf-8\"))
    cell = request[\"cell\"]
    with open(sys.argv[3], \"w\", encoding=\"utf-8\") as output:
        for repetition in range(request[\"recorded_iterations\"]):
            for timer in request[\"timing_boundaries\"]:
                row = {
                    \"schema\": \"simllm-nccl-primitive-observation-v1\",
                    \"manifest_digest\": request[\"manifest_digest\"],
                    \"cell_id\": cell[\"cell_id\"],
                    \"process_id\": request[\"process_id\"],
                    \"repetition\": repetition,
                    \"identity_digest\": identity[\"identity_digest\"],
                    \"requested\": cell[\"requested\"],
                    \"realized\": dict(cell[\"requested\"]),
                    \"family\": cell[\"family\"],
                    \"phase\": \"ordinary\",
                    \"timer\": timer,
                    \"units\": {
                        \"same_device_interval\": \"ns\",
                        \"cuda_event\": \"us\",
                        \"host_wall\": \"us\",
                    }[timer],
                    \"raw_duration\": 10 + repetition,
                    \"observed_local_delay\": 0,
                    \"correctness\": True,
                    \"qualification\": \"qualified\",
                    \"reason\": \"\",
                }
                output.write(json.dumps(row, sort_keys=True) + \"\\n\")
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    work_root = tmp_path / "work"
    for work in schedule["work"]:
        completed = subprocess.run(
            (
                sys.executable,
                str(STUDY / "run_study.py"),
                "run-one",
                "--manifest",
                str(manifest_path),
                "--inventory",
                str(inventory_path),
                "--schedule",
                str(schedule_path),
                "--probe",
                str(probe),
                "--output",
                str(work_root),
                "--work-index",
                str(work["work_index"]),
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    observations = tmp_path / "observations.jsonl"
    artifacts = tmp_path / "merged-artifacts.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "collect",
            "--manifest",
            str(manifest_path),
            "--inventory",
            str(inventory_path),
            "--schedule",
            str(schedule_path),
            "--work-root",
            str(work_root),
            "--output",
            str(observations),
            "--artifact-manifest",
            str(artifacts),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(observations.read_text().splitlines()) == 32

    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "run_study.py"),
            "validate",
            "--manifest",
            str(manifest_path),
            "--inventory",
            str(inventory_path),
            "--rows",
            str(observations),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["fatal_guards"] == "valid"
