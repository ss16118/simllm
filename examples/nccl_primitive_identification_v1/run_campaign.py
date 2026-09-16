#!/usr/bin/env python3
"""Run the frozen TRAF-94 schedule serially and resume from hashed artifacts.

This driver deliberately does not parallelize campaign work items.  Even
disjoint rank sets can share the node's NVSwitch fabric, so overlapping two
TRAF-94 probes would turn an uncontended primitive experiment into an
undocumented contention experiment.  ``run_study.py run-one`` remains the
authority for constructing and validating each work directory; this file only
supplies durable resumption, selected-device idle gating, and campaign-level
progress records.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.run_study import (
    _load_schedule,
    _verify_work_artifacts,
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Replace a small status record without exposing a partially written file."""

    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _device_uuids(indices: tuple[int, ...]) -> frozenset[str]:
    """Resolve physical CUDA indices before visibility remaps their ordinals."""

    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    by_index = {}
    for line in completed.stdout.splitlines():
        index, uuid = (field.strip() for field in line.split(",", 1))
        by_index[int(index)] = uuid
    missing = set(indices) - set(by_index)
    if missing:
        raise ValueError(f"selected CUDA devices do not exist: {sorted(missing)}")
    return frozenset(by_index[index] for index in indices)


def _compute_apps(monitored_uuids: frozenset[str]) -> list[dict[str, Any]]:
    """Return compute processes resident on the selected measurement GPUs.

    A process with a context on any selected GPU invalidates the idle boundary.
    The campaign operator must separately record any workload confined to other
    GPUs; such a workload is acceptable only when it cannot issue peer traffic
    through the selected GPU set.
    """

    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    applications = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi compute row: {line!r}")
        uuid, pid, process_name, used_memory = fields
        if uuid in monitored_uuids:
            applications.append(
                {
                    "gpu_uuid": uuid,
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_gpu_memory_mib": int(used_memory),
                }
            )
    return applications


def _wait_for_idle(
    *,
    monitored_uuids: frozenset[str],
    consecutive_samples: int,
    poll_seconds: float,
) -> None:
    """Wait for a stable selected-device idle boundary before work starts."""

    idle_samples = 0
    last_report: tuple[tuple[str, int], ...] | None = None
    while idle_samples < consecutive_samples:
        applications = _compute_apps(monitored_uuids)
        signature = tuple((row["gpu_uuid"], row["pid"]) for row in applications)
        if applications:
            idle_samples = 0
            if signature != last_report:
                print(
                    json.dumps(
                        {"status": "waiting_for_idle_gpu_node", "applications": applications},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        else:
            idle_samples += 1
            if idle_samples == 1:
                print(
                    json.dumps(
                        {
                            "status": "idle_stability_check",
                            "required_samples": consecutive_samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        last_report = signature
        if idle_samples < consecutive_samples:
            time.sleep(poll_seconds)


def _completed_indices(schedule: dict[str, Any], output: Path) -> set[int]:
    """Verify, then return, resumable work rather than trusting directory names."""

    completed = set()
    for work in schedule["work"]:
        index = int(work["work_index"])
        workdir = output / f"work-{index:06d}"
        if not workdir.exists():
            continue
        # An incomplete or modified directory is evidence that needs diagnosis;
        # never overwrite it or silently count it as resumable progress.
        _verify_work_artifacts(workdir)
        completed.add(index)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--visible-devices",
        default="0,1,2,3",
        help="Physical CUDA devices exposed to each two/four-rank native probe.",
    )
    args = parser.parse_args()
    if args.idle_samples < 1 or args.idle_poll_seconds <= 0:
        raise ValueError("idle sampling controls must be positive")

    schedule = _load_schedule(args.schedule)
    visible_indices = tuple(int(value) for value in args.visible_devices.split(","))
    if len(visible_indices) < 4 or len(set(visible_indices)) != len(visible_indices):
        raise ValueError("the frozen campaign needs at least four unique CUDA devices")
    monitored_uuids = _device_uuids(visible_indices)
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output.parent / "campaign-status.json"
    completed = _completed_indices(schedule, args.output)
    total = len(schedule["work"])
    _write_json_atomic(
        status_path,
        {"status": "waiting_for_idle", "completed": len(completed), "total": total},
    )
    _wait_for_idle(
        monitored_uuids=monitored_uuids,
        consecutive_samples=args.idle_samples,
        poll_seconds=args.idle_poll_seconds,
    )

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    for work in schedule["work"]:
        index = int(work["work_index"])
        if index in completed:
            continue
        # At a work boundary this driver has no GPU child.  Any reported
        # compute application is therefore external and the next timing must
        # wait for a fresh, stable idle boundary.
        if _compute_apps(monitored_uuids):
            _write_json_atomic(
                status_path,
                {
                    "status": "waiting_for_idle",
                    "completed": len(completed),
                    "total": total,
                    "next_work_index": index,
                },
            )
            _wait_for_idle(
                monitored_uuids=monitored_uuids,
                consecutive_samples=args.idle_samples,
                poll_seconds=args.idle_poll_seconds,
            )

        _write_json_atomic(
            status_path,
            {
                "status": "running",
                "completed": len(completed),
                "total": total,
                "work_index": index,
                "cell_id": work["cell_id"],
                "process_id": work["process_id"],
            },
        )
        command = (
            sys.executable,
            str(HERE / "run_study.py"),
            "run-one",
            "--manifest",
            str(args.manifest),
            "--inventory",
            str(args.inventory),
            "--schedule",
            str(args.schedule),
            "--probe",
            str(args.probe),
            "--output",
            str(args.output),
            "--work-index",
            str(index),
            "--timeout",
            str(args.timeout),
        )
        completed_process = subprocess.run(command, env=environment, check=False)
        if completed_process.returncode:
            _write_json_atomic(
                status_path,
                {
                    "status": "failed",
                    "completed": len(completed),
                    "total": total,
                    "work_index": index,
                    "returncode": completed_process.returncode,
                },
            )
            raise SystemExit(completed_process.returncode)
        workdir = args.output / f"work-{index:06d}"
        # Query only after the timed child and its CUDA contexts have exited.
        # Polling NVML during an ordinary sample would itself be an observer and
        # violate the frozen design.  A context present at this first post-run
        # boundary might have started during the sample, so the completed work
        # is conservatively marked and cannot be resumed or collected.  The
        # operator retains it for audit, moves the directory aside, and reruns
        # that frozen work index after another stable-idle boundary.
        post_run_applications = _compute_apps(monitored_uuids)
        if post_run_applications:
            contamination = {
                "schema": "simllm-nccl-primitive-contamination-v1",
                "status": "void_external_context_at_post_run_boundary",
                "work_index": index,
                "applications": post_run_applications,
            }
            _write_json_atomic(workdir / "CONTAMINATED.json", contamination)
            _write_json_atomic(
                status_path,
                {
                    "status": "contaminated",
                    "completed": len(completed),
                    "total": total,
                    "work_index": index,
                    "applications": post_run_applications,
                },
            )
            raise SystemExit(86)
        _verify_work_artifacts(workdir)
        completed.add(index)
        _write_json_atomic(
            status_path,
            {
                "status": "running",
                "completed": len(completed),
                "total": total,
                "last_work_index": index,
            },
        )

    _write_json_atomic(
        status_path,
        {"status": "complete", "completed": len(completed), "total": total},
    )
    print(json.dumps({"status": "complete", "work_count": total}, indent=2))


if __name__ == "__main__":
    main()
