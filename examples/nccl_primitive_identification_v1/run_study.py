"""Plan, qualify, execute, and validate the frozen TRAF-94 experiment.

The runner delegates device work to a separately built source-faithful probe.
That boundary is deliberate: Python owns reproducibility, scheduling, evidence,
and fatal guards, while the pinned NCCL/CUDA build owns the measured work.  See
``PROBE_CONTRACT.md`` before implementing or selecting a probe binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.analysis import (
    fit_identification,
    score_confirmations,
)
from examples.nccl_primitive_identification_v1.matrix import (
    CAPABILITY_SCHEMA,
    canonical_json_bytes,
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

DEFAULT_MANIFEST = HERE / "manifest.json"
PROBE_IDENTITY_SCHEMA = "simllm-nccl-primitive-probe-identity-v1"
PROCESS_SCHEMA = "simllm-nccl-primitive-process-v1"
ARTIFACT_SCHEMA = "simllm-nccl-primitive-artifact-manifest-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(dict(row)).decode("utf-8"))


def _load_inventory(path: Path, *, required_state: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_inventory(value)
    if required_state is not None and value["state"] != required_state:
        raise ValueError(
            f"inventory state is {value['state']!r}; {required_state!r} is required"
        )
    return value


def _manifest_architectures(
    manifest: Mapping[str, Any], extension_path: Path | None
) -> tuple[str, ...]:
    if extension_path is None:
        return tuple(manifest["architectures"])
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if extension.get("schema") != "simllm-nccl-primitive-architecture-extension-v1":
        raise ValueError("unsupported architecture extension schema")
    if extension.get("base_manifest_digest") != content_digest(manifest):
        raise ValueError("architecture extension targets a different base manifest")
    if extension.get("source_commit") != manifest["source_commit"]:
        raise ValueError("architecture extension changes the pinned source commit")
    architectures = extension.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("architecture extension must contain architectures")
    selected = tuple(str(value["id"]) for value in architectures)
    if len(set(selected)) != len(selected):
        raise ValueError("architecture extension contains duplicate ids")
    return selected


def plan_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    architectures = _manifest_architectures(manifest, args.architecture_extension)
    cells = expand_manifest(manifest, architectures=architectures)
    inventory = inventory_document(manifest, cells, state="planned")
    _write_json(args.output, inventory)
    print(json.dumps({
        "state": inventory["state"],
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "cell_count": inventory["cell_count"],
        "capability_count": len(capability_keys(cells)),
        "architectures": list(architectures),
    }, indent=2))


def capability_plan_command(args: argparse.Namespace) -> None:
    inventory = _load_inventory(args.inventory, required_state="planned")
    # Recreate just enough typed cell state to use the same grouping rule as
    # expansion.  The capability key itself is already frozen in each row, so
    # this operation cannot silently regroup the inventory.
    unique: dict[str, dict[str, Any]] = {}
    for cell in inventory["cells"]:
        key = cell["capability_key"]
        candidate = {
            "schema": "simllm-nccl-primitive-capability-request-v1",
            "manifest_digest": inventory["manifest_digest"],
            "inventory_digest": inventory["inventory_digest"],
            "source_commit": inventory["source_commit"],
            "capability_key": key,
            **{
                name: cell[name]
                for name in (
                    "architecture",
                    "protocol",
                    "simple_placement",
                    "stage",
                    "family",
                )
            },
            # Grouping avoids repeating a diagnostic for every timing value,
            # but the device probe still needs concrete controls to launch.
            # The first cell is deterministic because inventory expansion is
            # ordered, so it is a reproducible representative pilot.
            "requested": cell["requested"],
        }
        existing = unique.get(key)
        # Within a support-equivalent group, use the strongest deterministic
        # diagnostic: longest delay, most useful work, and rotating allocation.
        # This prevents a zero-delay/one-word pilot from qualifying the harder
        # members while avoiding redundant timing-grid diagnostics.
        def pilot_score(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
            requested = row["requested"]
            return (
                int(requested.get("delay_cycles", 0)),
                int(requested.get("useful_bytes_per_channel", requested.get("useful_bytes", 0))),
                int(requested.get("reservations", 1)),
                int(requested.get("working_set") == "rotating_64MiB"),
            )

        if existing is None or pilot_score(candidate) > pilot_score(existing):
            unique[key] = candidate
    _write_jsonl(args.output, (unique[key] for key in sorted(unique)))
    print(json.dumps({"capability_count": len(unique), "output": str(args.output)}))


def qualify_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    planned = _load_inventory(args.inventory, required_state="planned")
    architectures = tuple(dict.fromkeys(cell["architecture"] for cell in planned["cells"]))
    expected = inventory_document(
        manifest,
        expand_manifest(manifest, architectures=architectures),
        state="planned",
    )
    # Equality here protects more than the manifest digest: it verifies that
    # the checked-in expansion algorithm still generates the exact planned
    # cells supplied to the capability probe.
    if expected != planned:
        raise ValueError("planned inventory is not the deterministic manifest expansion")
    rows = _read_jsonl(args.capabilities)
    typed_cells = expand_manifest(manifest, architectures=architectures)
    qualified = qualify_inventory(manifest, typed_cells, rows)
    _write_json(args.output, qualified)
    print(json.dumps({
        "state": qualified["state"],
        "inventory_digest": qualified["inventory_digest"],
        "qualified_cells": qualified["cell_count"],
        "capability_outcomes": len(qualified["capability_outcomes"]),
    }, indent=2))


def _probe_identity(probe: Path, timeout: int) -> dict[str, Any]:
    completed = subprocess.run(
        (str(probe), "--describe"),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"probe identity command exited {completed.returncode}: {completed.stderr.strip()}"
        )
    identity = json.loads(completed.stdout)
    if identity.get("schema") != PROBE_IDENTITY_SCHEMA:
        raise ValueError("probe returned an unsupported identity schema")
    for field in ("source_commit", "architecture", "build", "device", "source_files"):
        if field not in identity:
            raise ValueError(f"probe identity is missing {field!r}")
    if len(str(identity["source_commit"])) != 40:
        raise ValueError("probe source_commit is not a full Git object id")
    if not isinstance(identity["build"], dict) or not identity["build"]:
        raise ValueError("probe build identity must be a non-empty object")
    if not isinstance(identity["device"], dict) or not identity["device"]:
        raise ValueError("probe device identity must be a non-empty object")
    if not isinstance(identity["source_files"], list) or not identity["source_files"]:
        raise ValueError("probe source file identity must be a non-empty list")
    for digest_field in ("executable_sha256", "nccl_library_sha256"):
        digest = identity["build"].get(digest_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"probe build identity is missing {digest_field}")
    for device_field in ("uuids", "driver", "topology_digest"):
        if not identity["device"].get(device_field):
            raise ValueError(f"probe device identity is missing {device_field}")
    recorded = identity.get("identity_digest")
    unsigned = dict(identity)
    unsigned.pop("identity_digest", None)
    if recorded != content_digest(unsigned):
        raise ValueError("probe identity digest does not match its content")
    return identity


def run_capabilities_command(args: argparse.Namespace) -> None:
    requests = _read_jsonl(args.capability_plan)
    identity = _probe_identity(args.probe, args.timeout)
    build_record = json.loads(args.build_record.read_text(encoding="utf-8"))
    unsigned_build = dict(build_record)
    recorded_build_digest = unsigned_build.pop("build_record_digest", None)
    if recorded_build_digest != content_digest(unsigned_build):
        raise ValueError("build record digest does not match its content")
    if (
        identity["source_commit"] != build_record.get("source_commit")
        or identity["build"]["nccl_library_sha256"] != build_record.get("library_sha256")
        or identity["source_files"] != build_record.get("source_files")
    ):
        raise ValueError("probe identity is not bound to the supplied build record")
    output_rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for request in requests:
        key = request["capability_key"]
        request_path = args.output / f"{key}.request.json"
        stdout_path = args.output / f"{key}.stdout"
        stderr_path = args.output / f"{key}.stderr"
        _write_json(request_path, request | {"probe_identity": identity})
        if identity["source_commit"] != request["source_commit"]:
            # Preserve one explicit outcome per specialization.  Exiting at the
            # first mismatch would make the remaining cells look not attempted
            # rather than uniformly unqualified by source identity.
            completed = None
            stdout = ""
            stderr = "probe source commit differs from frozen request\n"
            returncode = None
        else:
            completed = subprocess.run(
                (str(args.probe), "--capability", str(request_path)),
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        if completed is None:
            row = {
                "schema": CAPABILITY_SCHEMA,
                "capability_key": key,
                "qualified": False,
                "reason": "source_commit_mismatch",
                "identity_digest": identity["identity_digest"],
                "source_operation_counts": {},
                "realized": {},
            }
        elif returncode:
            row = {
                "schema": CAPABILITY_SCHEMA,
                "capability_key": key,
                "qualified": False,
                "reason": f"probe_exit_{returncode}",
                "identity_digest": identity["identity_digest"],
                "source_operation_counts": {},
                "realized": {},
            }
        else:
            row = json.loads(stdout)
            if row.get("schema") != CAPABILITY_SCHEMA or row.get("capability_key") != key:
                raise ValueError(f"probe returned the wrong capability identity for {key}")
            if row.get("identity_digest") != identity["identity_digest"]:
                raise ValueError(f"probe changed identity during capability {key}")
            if not isinstance(row.get("source_operation_counts"), dict):
                raise ValueError(f"capability {key} omitted source-operation counts")
        output_rows.append(row)
    _write_jsonl(args.results, output_rows)
    print(json.dumps({
        "capabilities": len(output_rows),
        "qualified": sum(bool(row.get("qualified")) for row in output_rows),
        "results": str(args.results),
    }, indent=2))


def _schedule(inventory: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create five independent, deterministic, counterbalanced process orders."""

    processes = int(manifest["ordinary_processes_per_cell"])
    seed = int(manifest["ordering_seed"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in inventory["cells"]:
        # Ready-publication, copy/reduction, and sharing stages contain matched
        # family interventions.  Family is excluded from the grouping identity
        # so their A and B cells stay adjacent.  Other stages have one family;
        # grouping by the full cell id avoids inventing a pairing rule that the
        # frozen manifest did not declare.
        if cell["stage"] in {"ready_publication", "data_work", "sharing"}:
            identity = {
                name: value
                for name, value in cell.items()
                if name not in {"cell_id", "ordinal", "family", "capability_key"}
            }
            group_key = content_digest(identity)
        else:
            group_key = cell["cell_id"]
        groups.setdefault(group_key, []).append(cell)

    schedule = []
    for process_id in range(processes):
        # Shuffle matched groups rather than individual cells; otherwise A and
        # B can drift far apart and cease to be a useful paired intervention.
        # Odd processes reverse each group's canonical order, yielding AB/BA
        # counterbalancing while preserving one execution of every cell.
        group_keys = list(groups)
        random.Random(seed + process_id).shuffle(group_keys)
        order = []
        for group_key in group_keys:
            members = sorted(groups[group_key], key=lambda cell: cell["family"])
            if process_id % 2:
                members.reverse()
            order.extend(members)
        for position, cell in enumerate(order):
            schedule.append({
                "work_index": len(schedule),
                "process_id": process_id,
                "order_position": position,
                "pair_order": "AB" if process_id % 2 == 0 else "BA",
                "cell_id": cell["cell_id"],
            })
    return schedule


def schedule_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    if inventory["manifest_digest"] != content_digest(manifest):
        raise ValueError("qualified inventory targets a different manifest")
    if not inventory["cells"]:
        raise ValueError("ordinary timing cannot schedule an empty qualified inventory")
    schedule = {
        "schema": "simllm-nccl-primitive-schedule-v1",
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "ordering_seed": manifest["ordering_seed"],
        "pair_order": manifest["pair_order"],
        "work": _schedule(inventory, manifest),
    }
    schedule["schedule_digest"] = content_digest(schedule)
    _write_json(args.output, schedule)
    print(json.dumps({
        "work_count": len(schedule["work"]),
        "schedule_digest": schedule["schedule_digest"],
        "output": str(args.output),
    }, indent=2))


def _load_schedule(path: Path) -> dict[str, Any]:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if schedule.get("schema") != "simllm-nccl-primitive-schedule-v1":
        raise ValueError("unsupported schedule schema")
    unsigned = dict(schedule)
    recorded = unsigned.pop("schedule_digest", None)
    if recorded != content_digest(unsigned):
        raise ValueError("schedule digest does not match its content")
    return schedule


def run_one_command(args: argparse.Namespace) -> None:
    """Run one process/cell work item, suitable for a scheduler array task."""

    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    schedule = _load_schedule(args.schedule)
    if (
        schedule["manifest_digest"] != inventory["manifest_digest"]
        or schedule["inventory_digest"] != inventory["inventory_digest"]
    ):
        raise ValueError("schedule and qualified inventory identities differ")
    if args.work_index < 0 or args.work_index >= len(schedule["work"]):
        raise ValueError("work index is outside the frozen schedule")
    work = schedule["work"][args.work_index]
    cell = next(cell for cell in inventory["cells"] if cell["cell_id"] == work["cell_id"])
    identity = _probe_identity(args.probe, args.timeout)
    if identity.get("source_commit") != inventory["source_commit"]:
        raise ValueError("probe was not built from the pinned NCCL source commit")
    if identity.get("architecture") != cell["architecture"]:
        raise ValueError("probe device architecture does not match the cell")
    outcomes = {
        row["capability_key"]: row for row in inventory["capability_outcomes"]
    }
    capability = outcomes[cell["capability_key"]]
    if identity["identity_digest"] != capability["identity_digest"]:
        raise ValueError("ordinary probe identity differs from the capability pilot")

    workdir = args.output / f"work-{args.work_index:06d}"
    if workdir.exists():
        raise FileExistsError(f"refusing to overwrite existing work directory {workdir}")
    workdir.mkdir(parents=True)
    request = {
        "schema": PROCESS_SCHEMA,
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "schedule_digest": schedule["schedule_digest"],
        "cell": cell,
        "process_id": work["process_id"],
        "order_position": work["order_position"],
        "pair_order": work["pair_order"],
        "warmup_iterations": manifest["warmup_iterations"],
        "recorded_iterations": manifest["recorded_iterations"],
        "timing_boundaries": manifest["timing_boundaries"],
        "diagnostic_separate_from_timing": manifest["diagnostic_separate_from_timing"],
        "probe_identity": identity,
    }
    request_path = workdir / "request.json"
    rows_path = workdir / "observations.jsonl"
    _write_json(request_path, request)
    completed = subprocess.run(
        (str(args.probe), "--run", str(request_path), str(rows_path)),
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
        env={key: value for key, value in os.environ.items() if not key.startswith("NCCL_")},
    )
    (workdir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (workdir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        _write_json(workdir / "FAILED.json", {
            "status": "probe_failed",
            "returncode": completed.returncode,
        })
        raise RuntimeError(f"probe exited {completed.returncode}; evidence was retained")

    rows = _read_jsonl(rows_path)
    observed = set()
    for row in rows:
        validate_observation(row, manifest, inventory)
        if (
            row["cell_id"] != cell["cell_id"]
            or row["process_id"] != work["process_id"]
            or row["identity_digest"] != identity["identity_digest"]
        ):
            raise ValueError("probe output escaped its assigned process/cell identity")
        if row["phase"] != "ordinary" or row["qualification"] != "qualified":
            raise ValueError("ordinary probe work produced a non-qualified timing row")
        key = (row["repetition"], row["timer"])
        if key in observed:
            raise ValueError("probe produced duplicate repetition/timer rows")
        observed.add(key)
    expected_rows = manifest["recorded_iterations"] * len(manifest["timing_boundaries"])
    if len(rows) != expected_rows:
        raise ValueError(f"probe produced {len(rows)} rows; expected {expected_rows}")
    _write_artifact_manifest(workdir)
    print(json.dumps({
        "status": "complete",
        "work_index": args.work_index,
        "rows": len(rows),
        "workdir": str(workdir),
    }, indent=2))


def _write_artifact_manifest(workdir: Path) -> None:
    artifacts = []
    for path in sorted(workdir.iterdir()):
        if not path.is_file() or path.name in {"artifacts.json", "COMPLETE.json"}:
            continue
        data = path.read_bytes()
        artifacts.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema": ARTIFACT_SCHEMA, "artifacts": artifacts}
    _write_json(workdir / "artifacts.json", manifest)
    _write_json(workdir / "COMPLETE.json", {
        "schema": ARTIFACT_SCHEMA,
        "status": "complete",
        "manifest_sha256": hashlib.sha256((workdir / "artifacts.json").read_bytes()).hexdigest(),
    })


def _verify_work_artifacts(workdir: Path) -> None:
    """Verify a work directory before any of its rows enter the merged file."""

    contamination_path = workdir / "CONTAMINATED.json"
    if contamination_path.exists():
        raise ValueError(f"{workdir} has an unresolved external-contention marker")
    complete_path = workdir / "COMPLETE.json"
    artifacts_path = workdir / "artifacts.json"
    if not complete_path.is_file() or not artifacts_path.is_file():
        raise ValueError(f"{workdir} is not complete")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise ValueError(f"{workdir} has a non-complete terminal marker")
    if complete.get("manifest_sha256") != hashlib.sha256(artifacts_path.read_bytes()).hexdigest():
        raise ValueError(f"{workdir} artifact manifest hash changed")
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    if artifacts.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"{workdir} has an unsupported artifact manifest")
    for artifact in artifacts.get("artifacts", ()):
        path = workdir / artifact["path"]
        if not path.is_file():
            raise ValueError(f"{workdir} is missing artifact {artifact['path']}")
        data = path.read_bytes()
        if len(data) != artifact["bytes"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError(f"{workdir}/{artifact['path']} changed after completion")


def collect_command(args: argparse.Namespace) -> None:
    """Merge a complete schedule while rechecking every identity and hash."""

    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    schedule = _load_schedule(args.schedule)
    if (
        schedule["manifest_digest"] != inventory["manifest_digest"]
        or schedule["inventory_digest"] != inventory["inventory_digest"]
    ):
        raise ValueError("schedule and qualified inventory identities differ")
    expected_per_work = manifest["recorded_iterations"] * len(manifest["timing_boundaries"])
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    work_manifests = []
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as merged:
            for work in schedule["work"]:
                workdir = args.work_root / f"work-{work['work_index']:06d}"
                _verify_work_artifacts(workdir)
                request = json.loads((workdir / "request.json").read_text(encoding="utf-8"))
                if (
                    request.get("schedule_digest") != schedule["schedule_digest"]
                    or request.get("cell", {}).get("cell_id") != work["cell_id"]
                    or request.get("process_id") != work["process_id"]
                ):
                    raise ValueError(f"{workdir} request differs from the frozen schedule")
                rows = _read_jsonl(workdir / "observations.jsonl")
                if len(rows) != expected_per_work:
                    raise ValueError(f"{workdir} has {len(rows)} rows, expected {expected_per_work}")
                identities = set()
                for row in rows:
                    validate_observation(row, manifest, inventory)
                    if row["cell_id"] != work["cell_id"] or row["process_id"] != work["process_id"]:
                        raise ValueError(f"{workdir} contains rows for another work item")
                    key = (row["repetition"], row["timer"])
                    if key in identities:
                        raise ValueError(f"{workdir} contains duplicate repetition/timer rows")
                    identities.add(key)
                    merged.write(canonical_json_bytes(row).decode("utf-8"))
                    row_count += 1
                work_manifests.append({
                    "work_index": work["work_index"],
                    "complete_sha256": hashlib.sha256(
                        (workdir / "COMPLETE.json").read_bytes()
                    ).hexdigest(),
                })
        temporary.replace(args.output)
    except BaseException:
        # A partial file is useful only as transient debugging state and could
        # be mistaken for a complete campaign.  Remove just this explicitly
        # named temporary path; individual work evidence remains untouched.
        temporary.unlink(missing_ok=True)
        raise

    artifact_manifest = {
        "schema": "simllm-nccl-primitive-merged-artifacts-v1",
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "schedule_digest": schedule["schedule_digest"],
        "row_count": row_count,
        "observations_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "work_manifests": work_manifests,
    }
    _write_json(args.artifact_manifest, artifact_manifest)
    print(json.dumps({
        "status": "complete",
        "work_count": len(schedule["work"]),
        "row_count": row_count,
        "observations": str(args.output),
        "artifact_manifest": str(args.artifact_manifest),
    }, indent=2))


def validate_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    rows = _read_jsonl(args.rows)
    failures = completeness_failures(rows, manifest, inventory)
    void_scopes = timing_scope_voids(rows)
    report = {
        "schema": "simllm-nccl-primitive-validation-v1",
        "manifest_digest": inventory["manifest_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "observations_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "cell_count": inventory["cell_count"],
        "fatal_guards": (
            "invalid"
            if failures
            else "valid_outside_void_scopes"
            if void_scopes
            else "valid"
        ),
        "void_scopes": list(void_scopes),
        "failures": list(failures),
    }
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


def _complete_rows(
    *, rows_path: Path, manifest: Mapping[str, Any], inventory: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Load one complete campaign and return rows plus their immutable hash."""

    rows = _read_jsonl(rows_path)
    failures = completeness_failures(rows, manifest, inventory)
    if failures:
        raise ValueError("campaign rows failed completeness: " + "; ".join(failures))
    return rows, hashlib.sha256(rows_path.read_bytes()).hexdigest()


def fit_command(args: argparse.Namespace) -> None:
    """Lock identification parameters without consuming confirmation values."""

    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    rows, observations_sha256 = _complete_rows(
        rows_path=args.rows, manifest=manifest, inventory=inventory
    )
    fit = fit_identification(
        rows,
        inventory=inventory,
        manifest=manifest,
        observations_sha256=observations_sha256,
    )
    _write_json(args.output, fit)
    print(
        json.dumps(
            {
                "status": "complete",
                "fit_digest": fit["fit_digest"],
                "identification_anchors": len(fit["anchors"]),
                "paired_contrasts": len(fit["paired_contrasts"]),
                "confirmation_rows_used": fit["confirmation_rows_used"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


def score_command(args: argparse.Namespace) -> None:
    """Open held-out cells only after loading a self-digested locked fit."""

    manifest = load_manifest(args.manifest)
    inventory = _load_inventory(args.inventory, required_state="capability_qualified")
    rows, observations_sha256 = _complete_rows(
        rows_path=args.rows, manifest=manifest, inventory=inventory
    )
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    score = score_confirmations(
        rows,
        inventory=inventory,
        manifest=manifest,
        fit=fit,
        observations_sha256=observations_sha256,
    )
    _write_json(args.output, score)
    print(
        json.dumps(
            {
                "status": "complete",
                "score_digest": score["score_digest"],
                **score["summary"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="expand one architecture inventory")
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--architecture-extension", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(function=plan_command)

    capability_plan = subparsers.add_parser("capability-plan")
    capability_plan.add_argument("--inventory", type=Path, required=True)
    capability_plan.add_argument("--output", type=Path, required=True)
    capability_plan.set_defaults(function=capability_plan_command)

    run_capabilities = subparsers.add_parser("run-capabilities")
    run_capabilities.add_argument("--capability-plan", type=Path, required=True)
    run_capabilities.add_argument("--probe", type=Path, required=True)
    run_capabilities.add_argument("--build-record", type=Path, required=True)
    run_capabilities.add_argument("--output", type=Path, required=True)
    run_capabilities.add_argument("--results", type=Path, required=True)
    run_capabilities.add_argument("--timeout", type=int, default=300)
    run_capabilities.set_defaults(function=run_capabilities_command)

    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    qualify.add_argument("--inventory", type=Path, required=True)
    qualify.add_argument("--capabilities", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.set_defaults(function=qualify_command)

    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    schedule.add_argument("--inventory", type=Path, required=True)
    schedule.add_argument("--output", type=Path, required=True)
    schedule.set_defaults(function=schedule_command)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run_one.add_argument("--inventory", type=Path, required=True)
    run_one.add_argument("--schedule", type=Path, required=True)
    run_one.add_argument("--probe", type=Path, required=True)
    run_one.add_argument("--output", type=Path, required=True)
    run_one.add_argument("--work-index", type=int, required=True)
    run_one.add_argument("--timeout", type=int, default=1800)
    run_one.set_defaults(function=run_one_command)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    collect.add_argument("--inventory", type=Path, required=True)
    collect.add_argument("--schedule", type=Path, required=True)
    collect.add_argument("--work-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--artifact-manifest", type=Path, required=True)
    collect.set_defaults(function=collect_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument("--rows", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    validate.set_defaults(function=validate_command)

    fit = subparsers.add_parser(
        "fit", help="fit and lock parameters from identification cells only"
    )
    fit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fit.add_argument("--inventory", type=Path, required=True)
    fit.add_argument("--rows", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(function=fit_command)

    score = subparsers.add_parser(
        "score", help="score held-out confirmation deltas against a locked fit"
    )
    score.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    score.add_argument("--inventory", type=Path, required=True)
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--fit", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(function=score_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
