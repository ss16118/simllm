"""Deterministic matrix and evidence rules for the TRAF-94 experiment.

This module intentionally contains no CUDA or subprocess code.  Keeping the
experiment contract pure makes it possible to review and test the inventory,
qualification, and completeness rules without needing a particular GPU node.
The hardware runner imports these functions and is not allowed to weaken them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "simllm-nccl-primitive-identification-plan-v1"
INVENTORY_SCHEMA = "simllm-nccl-primitive-inventory-v1"
OBSERVATION_SCHEMA = "simllm-nccl-primitive-observation-v1"
CAPABILITY_SCHEMA = "simllm-nccl-primitive-capability-v1"
QUALIFICATIONS = frozenset(("qualified", "unqualified", "void"))
REQUIRED_SOURCE_COUNTS = frozenset((
    "readiness_checks",
    "successful_observations",
    "tail_publications",
    "head_publications",
    "source_load_bytes",
    "shared_staging_bytes",
    "remote_read_bytes",
    "remote_write_bytes",
    "consumed_steps",
))


def canonical_json_bytes(value: object) -> bytes:
    """Encode identity-bearing JSON independently of whitespace and key order."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def content_digest(value: object) -> str:
    """Return the lowercase SHA-256 used by all generated TRAF-94 identities."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject ambiguous plans before their ambiguity can enter a frozen digest."""

    if manifest.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported primitive manifest schema: {manifest.get('schema')!r}")
    for name in (
        "source_commit",
        "architectures",
        "protocols",
        "simple_placements",
        "baseline",
        "working_warps",
        "baseline_working_warps",
        "stages",
        "required_row_fields",
    ):
        if name not in manifest:
            raise ValueError(f"primitive manifest is missing {name!r}")
    if len(str(manifest["source_commit"])) != 40:
        raise ValueError("source_commit must be a full 40-character Git object id")
    if set(manifest["protocols"]) != {"LL", "LL128", "SIMPLE"}:
        raise ValueError("v1 requires the LL, LL128, and SIMPLE protocols")
    if set(manifest["simple_placements"]) != {"buffered", "buffered_read"}:
        raise ValueError("v1 requires both Simple buffer placements")
    stage_ids = [stage.get("id") for stage in manifest["stages"]]
    if not stage_ids or len(stage_ids) != len(set(stage_ids)):
        raise ValueError("stage ids must be present and unique")
    for name in (
        "ordinary_processes_per_cell",
        "warmup_iterations",
        "recorded_iterations",
        "ordering_seed",
    ):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ExperimentCell:
    """One fully specified cell from exactly one sequential manifest stage."""

    cell_id: str
    ordinal: int
    stage: str
    role: str
    family: str
    architecture: str
    protocol: str
    simple_placement: str
    requested: dict[str, Any]

    @property
    def capability_key(self) -> str:
        """Group controls that rely on the same source specialization.

        Capability is established once per primitive family and source branch,
        not once for every byte count or delay.  The latter are ordinary timing
        controls and must not turn the diagnostic pilot into the full campaign.
        """

        key = {
            "architecture": self.architecture,
            "protocol": self.protocol,
            "simple_placement": self.simple_placement,
            "stage": self.stage,
            "family": self.family,
        }
        return "cap-" + content_digest(key)[:16]


def _protocol_placements(manifest: Mapping[str, Any], protocol: str) -> tuple[str, ...]:
    # LL and LL128 have only their source-defined write placement.  Expanding
    # both Simple placement labels for them would create duplicate cells and
    # falsely suggest that a receiver-side read specialization was exercised.
    if protocol == "SIMPLE":
        return tuple(str(value) for value in manifest["simple_placements"])
    return ("source_default",)


def _stage_axes(
    manifest: Mapping[str, Any], stage: Mapping[str, Any], protocol: str
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    axes: list[tuple[str, tuple[Any, ...]]] = []
    family = stage.get("family", ("confirmation",))
    axes.append(("family", tuple(family) if isinstance(family, list) else (family,)))
    for name, value in stage.items():
        if name in {"id", "family", "role"}:
            continue
        if name == "working_warps" and value == "protocol_grid":
            value = manifest["working_warps"][protocol]
        # Every list named by a stage is an axis for that stage only.  Scalars
        # are held controls.  This is the crucial distinction from taking a
        # Cartesian product over every axis in the complete manifest.
        values = tuple(value) if isinstance(value, list) else (value,)
        if not values:
            raise ValueError(f"stage {stage['id']!r} has an empty {name!r} axis")
        axes.append((str(name), values))
    return tuple(axes)


def expand_manifest(
    manifest: Mapping[str, Any], *, architectures: Sequence[str] | None = None
) -> tuple[ExperimentCell, ...]:
    """Expand each stage independently and assign content-derived stable ids."""

    validate_manifest(manifest)
    selected_architectures = tuple(architectures or manifest["architectures"])
    if not selected_architectures or len(set(selected_architectures)) != len(
        selected_architectures
    ):
        raise ValueError("architectures must be a non-empty unique sequence")

    drafts: list[dict[str, Any]] = []
    for architecture, protocol in itertools.product(
        selected_architectures, manifest["protocols"]
    ):
        for placement in _protocol_placements(manifest, protocol):
            for stage in manifest["stages"]:
                axes = _stage_axes(manifest, stage, protocol)
                names = tuple(name for name, _ in axes)
                for values in itertools.product(*(values for _, values in axes)):
                    selected = dict(zip(names, values, strict=True))
                    requested = {
                        **manifest["baseline"],
                        "dtype": manifest["dtype"],
                        "architecture": architecture,
                        "protocol": protocol,
                        "simple_placement": placement,
                        "working_warps": manifest["baseline_working_warps"][protocol],
                        **{name: value for name, value in selected.items() if name != "family"},
                    }
                    drafts.append({
                        "stage": stage["id"],
                        "role": stage.get("role", "identification"),
                        "family": selected["family"],
                        "architecture": architecture,
                        "protocol": protocol,
                        "simple_placement": placement,
                        "requested": requested,
                    })

    cells = []
    for ordinal, draft in enumerate(drafts):
        # The readable ordinal is convenient during a long campaign; the hash
        # prevents a reordered or edited cell from masquerading as the old id.
        cell_id = f"{draft['stage']}-{ordinal:06d}-{content_digest(draft)[:12]}"
        cells.append(ExperimentCell(cell_id=cell_id, ordinal=ordinal, **draft))
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("content-derived cell ids unexpectedly collided")
    return tuple(cells)


def inventory_document(
    manifest: Mapping[str, Any], cells: Sequence[ExperimentCell], *, state: str
) -> dict[str, Any]:
    """Build a self-digesting planned or capability-qualified inventory."""

    if state not in {"planned", "capability_qualified"}:
        raise ValueError("inventory state must be planned or capability_qualified")
    body: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "state": state,
        "manifest_digest": content_digest(manifest),
        "source_commit": manifest["source_commit"],
        "cell_count": len(cells),
        "cells": [asdict(cell) | {"capability_key": cell.capability_key} for cell in cells],
    }
    body["inventory_digest"] = content_digest(body)
    return body


def validate_inventory(document: Mapping[str, Any]) -> None:
    if document.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unsupported inventory schema")
    unsigned = dict(document)
    recorded = unsigned.pop("inventory_digest", None)
    if recorded != content_digest(unsigned):
        raise ValueError("inventory digest does not match its content")
    cells = document.get("cells")
    if not isinstance(cells, list) or document.get("cell_count") != len(cells):
        raise ValueError("inventory cell_count does not match cells")
    ids = [cell.get("cell_id") for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("inventory contains duplicate cell ids")
    if document.get("state") == "capability_qualified":
        outcomes = document.get("capability_outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("qualified inventory is missing capability outcomes")
        by_key = {row.get("capability_key"): row for row in outcomes}
        if len(by_key) != len(outcomes):
            raise ValueError("qualified inventory has duplicate capability outcomes")
        if any(
            not by_key.get(cell.get("capability_key"), {}).get("qualified")
            for cell in cells
        ):
            raise ValueError("qualified inventory contains a cell without a passing outcome")


def capability_keys(cells: Sequence[ExperimentCell]) -> tuple[dict[str, str], ...]:
    """Return the minimal deterministic pilot inventory."""

    unique: dict[str, dict[str, str]] = {}
    for cell in cells:
        unique.setdefault(cell.capability_key, {
            "capability_key": cell.capability_key,
            "architecture": cell.architecture,
            "protocol": cell.protocol,
            "simple_placement": cell.simple_placement,
            "stage": cell.stage,
            "family": cell.family,
        })
    return tuple(unique[key] for key in sorted(unique))


def qualify_inventory(
    manifest: Mapping[str, Any],
    cells: Sequence[ExperimentCell],
    capability_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze only cells whose exact specialization passed the diagnostic."""

    key_descriptors = {
        row["capability_key"]: row for row in capability_keys(cells)
    }
    expected = set(key_descriptors)
    observed: dict[str, Mapping[str, Any]] = {}
    for row in capability_rows:
        if row.get("schema") != CAPABILITY_SCHEMA:
            raise ValueError("unsupported capability row schema")
        key = str(row.get("capability_key", ""))
        if key not in expected or key in observed:
            raise ValueError(f"unexpected or duplicate capability key {key!r}")
        if not isinstance(row.get("qualified"), bool):
            raise TypeError(f"capability {key!r} lacks a boolean qualified result")
        if not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError(f"capability {key!r} lacks an explicit reason")
        identity_digest = row.get("identity_digest")
        if not isinstance(identity_digest, str) or len(identity_digest) != 64:
            raise ValueError(f"capability {key!r} lacks a full identity digest")
        if not isinstance(row.get("source_operation_counts"), dict):
            raise TypeError(f"capability {key!r} lacks source-operation counts")
        if not isinstance(row.get("realized"), dict):
            raise TypeError(f"capability {key!r} lacks realized controls")
        if row["qualified"] and not row["realized"]:
            raise ValueError(f"qualified capability {key!r} has empty realized controls")
        if row["qualified"]:
            missing_counts = REQUIRED_SOURCE_COUNTS - set(row["source_operation_counts"])
            if missing_counts:
                raise ValueError(
                    f"qualified capability {key!r} is missing source counts "
                    f"{sorted(missing_counts)}"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in row["source_operation_counts"].values()
            ):
                raise ValueError(
                    f"qualified capability {key!r} has invalid source counts"
                )
        observed[key] = row
    missing = expected - set(observed)
    if missing:
        raise ValueError(f"capability results are missing {len(missing)} keys")

    identities_by_architecture: dict[str, set[str]] = {}
    for key, row in observed.items():
        if row["qualified"]:
            architecture = key_descriptors[key]["architecture"]
            identities_by_architecture.setdefault(architecture, set()).add(
                row["identity_digest"]
            )
    inconsistent = {
        architecture: identities
        for architecture, identities in identities_by_architecture.items()
        if len(identities) != 1
    }
    if inconsistent:
        raise ValueError("qualified capabilities use multiple identities within an architecture")

    qualified = [cell for cell in cells if observed[cell.capability_key]["qualified"]]
    result = inventory_document(manifest, qualified, state="capability_qualified")
    # Retain every failure rather than making unsupported source branches
    # disappear.  A reviewer can therefore distinguish "not run" from "failed
    # capability" without consulting transient runner logs.
    result["capability_outcomes"] = [dict(observed[key]) for key in sorted(observed)]
    unsigned = dict(result)
    unsigned.pop("inventory_digest")
    result["inventory_digest"] = content_digest(unsigned)
    return result


def validate_observation(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    """Validate one raw row without inferring or repairing missing evidence."""

    required = set(manifest["required_row_fields"])
    missing = required - set(row)
    if missing:
        raise ValueError(f"observation is missing required fields: {sorted(missing)}")
    if row["schema"] != OBSERVATION_SCHEMA:
        raise ValueError("unsupported observation schema")
    if row["manifest_digest"] != inventory["manifest_digest"]:
        raise ValueError("observation manifest digest does not match inventory")
    cells = {cell["cell_id"]: cell for cell in inventory["cells"]}
    cell = cells.get(row["cell_id"])
    if cell is None:
        raise ValueError("observation cell is outside the frozen inventory")
    if row["requested"] != cell["requested"] or row["family"] != cell["family"]:
        raise ValueError("observation requested controls or family changed after freeze")
    if not isinstance(row["realized"], dict) or not row["realized"]:
        raise ValueError("observation must record non-empty realized controls")
    unrealized = set(row["requested"]) - set(row["realized"])
    if unrealized:
        raise ValueError(f"observation omitted realized controls: {sorted(unrealized)}")
    if row["timer"] not in manifest["timing_boundaries"]:
        raise ValueError("observation uses an undeclared timer boundary")
    if row["units"] not in {"cycles", "ns", "us", "ps"}:
        raise ValueError("observation uses unsupported units")
    for name in ("raw_duration", "observed_local_delay"):
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not isinstance(row["correctness"], bool):
        raise TypeError("correctness must be boolean")
    if row["qualification"] not in QUALIFICATIONS:
        raise ValueError("observation qualification is invalid")
    if not isinstance(row["reason"], str):
        raise TypeError("observation reason must be a string")
    if row["qualification"] != "qualified" and not row["reason"]:
        raise ValueError("non-qualified observations require an explicit reason")
    if row["qualification"] == "qualified" and not row["correctness"]:
        raise ValueError("an incorrect observation cannot be qualified")
    if not isinstance(row["identity_digest"], str) or len(row["identity_digest"]) != 64:
        raise ValueError("identity_digest must be a full SHA-256")
    process_id = row["process_id"]
    repetition = row["repetition"]
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id < 0:
        raise ValueError("process_id must be a non-negative integer")
    if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
        raise ValueError("repetition must be a non-negative integer")
    if process_id >= manifest["ordinary_processes_per_cell"]:
        raise ValueError("process_id is outside the frozen process inventory")
    if repetition >= manifest["recorded_iterations"]:
        raise ValueError("repetition is outside the frozen iteration inventory")


def completeness_failures(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every ordinary-campaign completeness failure, deterministically."""

    failures: list[str] = []
    for index, row in enumerate(rows):
        try:
            validate_observation(row, manifest, inventory)
        except ValueError as error:
            failures.append(f"row {index}: {error}")

    qualified = [row for row in rows if row.get("qualification") == "qualified"]
    processes = int(manifest["ordinary_processes_per_cell"])
    repetitions = int(manifest["recorded_iterations"])
    timers = tuple(manifest["timing_boundaries"])
    expected = {
        (cell["cell_id"], process, repetition, timer)
        for cell in inventory["cells"]
        for process in range(processes)
        for repetition in range(repetitions)
        for timer in timers
    }
    observed = [
        (row.get("cell_id"), row.get("process_id"), row.get("repetition"), row.get("timer"))
        for row in qualified
        if row.get("phase") == "ordinary"
    ]
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        failures.append("ordinary campaign contains duplicate cell/process/repetition/timer rows")
    missing = expected - observed_set
    extra = observed_set - expected
    if missing:
        failures.append(f"ordinary campaign is missing {len(missing)} required rows")
    if extra:
        failures.append(f"ordinary campaign contains {len(extra)} unexpected rows")
    return tuple(failures)
