"""Contract tests for the TRAF-94 matrix, evidence gates, and statistics."""

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_primitive_identification_v1"

from examples.nccl_primitive_identification_v1.analysis import (
    paired_contrast,
    score_confirmation,
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
    validate_inventory,
    validate_observation,
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
    assert len(capability_keys(first)) == 32
    assert Counter(cell.stage for cell in first) == {
        "ready_publication": 96,
        "reuse": 160,
        "data_work": 112,
        "sharing": 288,
        "confirmation": 192,
    }
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
    assert all("reservations" not in cell.requested for cell in first if cell.stage == "ready_publication")


def test_manifest_and_inventory_digests_reject_semantic_edits(manifest):
    assert content_digest(manifest) == "dcb481d70a436b2b82f8091cefceff0d5363ab026eb55886cd1ba470cccb9823"
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
                name: int(name == "readiness_checks")
                for name in REQUIRED_SOURCE_COUNTS
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
        row for row in qualified["capability_outcomes"]
        if row["capability_key"] == failed_key
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
        "units": "ns",
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


def test_paired_iqr_and_heldout_rules_use_process_summaries():
    rows = []
    for process in range(5):
        for repetition in range(3):
            for cell, duration in (("a", 100 + process), ("b", 130 + process)):
                rows.append({
                    "cell_id": cell,
                    "process_id": process,
                    "repetition": repetition,
                    "timer": "cuda_event",
                    "phase": "ordinary",
                    "qualification": "qualified",
                    "raw_duration": duration,
                })
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

    assert score_confirmation(
        measured_delta=20,
        predicted_delta=18,
        frozen_repeat_spread_resolution=1,
    )["accepted"] is True
    assert score_confirmation(
        measured_delta=20,
        predicted_delta=17,
        frozen_repeat_spread_resolution=1,
    )["accepted"] is False


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
    assert len(requests) == 32
    assert len({row["capability_key"] for row in requests}) == 32


def test_schedule_keeps_matched_families_adjacent_and_counterbalanced(manifest):
    small = _small_contract(manifest)
    small["stages"] = [{
        "id": "data_work",
        "family": ["copy", "sum"],
        "useful_bytes": [16, 32],
    }]
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
                candidate for candidate in cells
                if candidate.family != cell.family
                and candidate.protocol == cell.protocol
                and candidate.simple_placement == cell.simple_placement
                and candidate.requested == cell.requested
            )
            assert abs(position[cell.cell_id] - position[partner.cell_id]) == 1
            assert work[position[cell.cell_id]]["pair_order"] == (
                "AB" if process % 2 == 0 else "BA"
            )


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
                    \"units\": \"ns\",
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
