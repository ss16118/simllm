"""Capture a pre-pilot environment audit without claiming probe capability.

This command is intentionally weaker than ``run_study.py run-capabilities``.
It finds cheap blockers (wrong source, dirty checkout, wrong GPU generation,
missing peer topology, or an unbound library) before anyone spends time
building or launching the diagnostic probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.matrix import content_digest


def _command(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command {arguments!r} exited {completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(source: Path) -> dict[str, Any]:
    head = _command("git", "rev-parse", "HEAD", cwd=source).strip()
    status = _command("git", "status", "--porcelain=v1", cwd=source).splitlines()
    files = []
    # These files contain the generic primitive, protocol specializations, and
    # collective call sites relevant to the frozen families.  The future probe
    # may add more files, but it may not omit these from its final build record.
    for relative in (
        "src/device/primitives.h",
        "src/device/prims_ll.h",
        "src/device/prims_ll128.h",
        "src/device/prims_simple.h",
        "src/device/reduce_kernel.h",
        "src/device/sendrecv.h",
        "src/device/all_reduce.h",
        "src/device/common.h",
        "src/enqueue/enqueue.cc",
        "src/include/comm.h",
        "src/include/device.h",
        "src/include/traf94.h",
        "src/init.cc",
        "src/nccl.h.in",
        "src/transport/p2p.cc",
    ):
        path = source / relative
        files.append({
            "path": relative,
            "present": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        })
    return {"path": str(source.resolve()), "head": head, "dirty": status, "files": files}


def _device_identity() -> tuple[list[dict[str, str]], str]:
    fields = "index,name,uuid,compute_cap,driver_version"
    output = _command("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader")
    devices = []
    for line in output.splitlines():
        index, name, uuid, capability, driver = (part.strip() for part in line.split(","))
        devices.append({
            "index": index,
            "name": name,
            "uuid": uuid,
            "compute_capability": capability,
            "driver": driver,
        })
    topology = _command("nvidia-smi", "topo", "-m")
    return devices, topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    extension = json.loads(args.extension.read_text(encoding="utf-8"))
    if len(extension.get("architectures", ())) != 1:
        raise ValueError("environment audit requires exactly one architecture extension")
    expected = extension["architectures"][0]
    source = _source_identity(args.source)
    devices, topology = _device_identity()
    library = args.library.resolve(strict=True)
    nvcc = _command("nvcc", "--version").strip()

    build_record = (
        json.loads(args.build_record.read_text(encoding="utf-8"))
        if args.build_record
        else None
    )
    reasons = []
    if source["head"] != extension["source_commit"]:
        reasons.append("source_commit_mismatch")
    # A clean checkout is required for the upstream build.  The experimental
    # probe build is allowed to contain only the versioned instrumentation
    # patch recorded below; arbitrary dirty state remains fatal.
    if source["dirty"] and build_record is None:
        reasons.append("source_checkout_dirty")
    if any(not row["present"] for row in source["files"]):
        reasons.append("required_source_file_missing")
    if len(devices) < int(expected["minimum_gpu_count"]):
        reasons.append("insufficient_gpu_count")
    if any(device["name"] != expected["name_pattern"] for device in devices):
        reasons.append("device_name_mismatch")
    if any(
        device["compute_capability"] != expected["compute_capability"]
        for device in devices
    ):
        reasons.append("compute_capability_mismatch")

    # For an N-GPU all-peer node, every GPU row has N-1 NV-prefixed peer cells.
    # This is a conservative textual check of nvidia-smi's own topology report;
    # the capability probe must still verify actual peer access and transfers.
    gpu_rows = [line.split() for line in topology.splitlines() if line.startswith("GPU")]
    if len(gpu_rows) != len(devices) or any(
        sum(value.startswith("NV") for value in row[1 : 1 + len(devices)]) != len(devices) - 1
        for row in gpu_rows
    ):
        reasons.append("all_peer_nvlink_topology_not_observed")

    if args.build_record:
        assert build_record is not None
        unsigned = dict(build_record)
        recorded_digest = unsigned.pop("build_record_digest", None)
        if (
            build_record.get("schema") != "simllm-nccl-primitive-build-record-v1"
            or recorded_digest != content_digest(unsigned)
        ):
            reasons.append("build_record_digest_invalid")
        if build_record.get("source_commit") != extension["source_commit"]:
            reasons.append("build_record_source_mismatch")
        if build_record.get("library_sha256") != _sha256(library):
            reasons.append("build_record_library_mismatch")
        if source["dirty"]:
            if not build_record.get("instrumentation_patch_sha256"):
                reasons.append("dirty_source_without_instrumentation_patch")
            if build_record.get("instrumentation_dirty_paths") != source["dirty"]:
                reasons.append("instrumented_dirty_paths_changed")
            if build_record.get("source_files") != source["files"]:
                reasons.append("instrumented_source_files_changed")
    else:
        # A library filename or package version cannot prove which source tree
        # produced its bytes.  Requiring a build record closes that provenance
        # gap before the more expensive source-conformance pilot.
        reasons.append("source_build_binding_unverified")

    report = {
        "schema": "simllm-nccl-primitive-environment-audit-v1",
        "architecture": expected["id"],
        "qualified_for_capability_build": not reasons,
        "reasons": reasons or ["qualified"],
        "source": source,
        "library": {"path": str(library), "sha256": _sha256(library)},
        "build_record": build_record,
        "devices": devices,
        "topology": topology,
        "topology_sha256": hashlib.sha256(topology.encode("utf-8")).hexdigest(),
        "nvcc": nvcc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "qualified_for_capability_build": report["qualified_for_capability_build"],
        "reasons": report["reasons"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
