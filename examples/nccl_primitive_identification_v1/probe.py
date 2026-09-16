#!/usr/bin/env python3
"""Contract adapter for the TRAF-94 native CUDA/NCCL probe.

The adapter validates frozen JSON requests, constructs a sanitized and explicit
NCCL environment, checks diagnostic evidence, and emits the exact schemas used
by ``run_study.py``.  It never manufactures a realized control: values that
come from the device are checked before a capability or ordinary row qualifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.matrix import (
    CAPABILITY_SCHEMA,
    OBSERVATION_SCHEMA,
    REQUIRED_SOURCE_COUNTS,
    canonical_json_bytes,
    content_digest,
)

IDENTITY_SCHEMA = "simllm-nccl-primitive-probe-identity-v1"
NATIVE_SCHEMA = "simllm-nccl-primitive-native-v1"
CONFIG_ENV = "TRAF94_PROBE_CONFIG"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _command(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
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


def _config() -> dict[str, Any]:
    path = Path(os.environ.get(CONFIG_ENV, HERE / "probe-build.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "simllm-nccl-primitive-probe-build-v1":
        raise ValueError("unsupported probe build configuration")
    unsigned = dict(value)
    recorded = unsigned.pop("config_digest", None)
    if recorded != content_digest(unsigned):
        raise ValueError("probe build configuration digest mismatch")
    value["_path"] = str(path.resolve())
    return value


def _device_identity() -> dict[str, Any]:
    rows = _command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,driver_version",
        "--format=csv,noheader",
    ).splitlines()
    devices = []
    for row in rows:
        index, name, uuid, capability, driver = (field.strip() for field in row.split(","))
        devices.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "compute_capability": capability,
                "driver": driver,
            }
        )
    topology = _command("nvidia-smi", "topo", "-m")
    return {
        "uuids": [row["uuid"] for row in devices],
        "driver": devices[0]["driver"],
        "devices": devices,
        "topology_digest": hashlib.sha256(topology.encode("utf-8")).hexdigest(),
    }


def identity(config: Mapping[str, Any]) -> dict[str, Any]:
    native = Path(config["native_executable"]).resolve(strict=True)
    library = Path(config["nccl_library"]).resolve(strict=True)
    if _sha256(native) != config["native_executable_sha256"]:
        raise ValueError("native probe changed after its build configuration")
    if _sha256(library) != config["library_sha256"]:
        raise ValueError("configured NCCL library changed after probe build")
    linked = _command("ldd", str(native))
    resolved_nccl = []
    for line in linked.splitlines():
        if "libnccl.so" not in line or "=>" not in line:
            continue
        candidate = line.split("=>", 1)[1].strip().split()[0]
        if candidate != "not":
            resolved_nccl.append(Path(candidate).resolve())
    if resolved_nccl != [library]:
        raise ValueError("native probe does not resolve to the recorded NCCL library")
    result: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "source_commit": config["source_commit"],
        "architecture": config["architecture"],
        "build": {
            "compiler": config["compiler"],
            "cuda": config["cuda"],
            "flags": config["flags"],
            "executable_sha256": _sha256(native),
            "wrapper_sha256": _sha256(Path(__file__).resolve()),
            "nccl_library_sha256": _sha256(library),
            "instrumentation_patch_sha256": config["instrumentation_patch_sha256"],
        },
        "device": _device_identity(),
        "source_files": config["source_files"],
    }
    result["identity_digest"] = content_digest(result)
    return result


def _threads(protocol: str, working_warps: int) -> int:
    # NCCL_NTHREADS names collective working threads.  The planner adds
    # Simple's synchronization warp itself; the patched P2P scheduler likewise
    # uses requestedWorkingWarps and adds that warp only on the sender rank.
    return 32 * working_warps


def _useful_bytes(requested: Mapping[str, Any]) -> int:
    """Resolve the mutually exclusive total/per-channel payload controls."""

    return (
        int(requested["useful_bytes_per_channel"]) * int(requested["active_channels"])
        if "useful_bytes_per_channel" in requested
        else int(requested["useful_bytes"])
    )


def _native_environment(requested: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    protocol = str(requested["protocol"])
    threads = _threads(protocol, int(requested["working_warps"]))
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
    library_dir = str(Path(config["nccl_library"]).resolve().parent)
    prior_library_path = environment.get("LD_LIBRARY_PATH")
    environment.update(
        LD_LIBRARY_PATH=(
            library_dir if not prior_library_path else library_dir + os.pathsep + prior_library_path
        ),
        NCCL_ALGO="Ring",
        NCCL_PROTO=protocol,
        NCCL_MIN_NCHANNELS=str(requested["active_channels"]),
        NCCL_MAX_NCHANNELS=str(requested["active_channels"]),
        NCCL_MIN_CTAS=str(requested["active_channels"]),
        NCCL_MAX_CTAS=str(requested["active_channels"]),
        NCCL_MIN_P2P_NCHANNELS=str(requested["active_channels"]),
        NCCL_MAX_P2P_NCHANNELS=str(requested["active_channels"]),
        NCCL_THREAD_THRESHOLDS="0 0 0 0 0 0",
        NCCL_P2P_READ_ENABLE=("1" if requested["simple_placement"] == "buffered_read" else "0"),
    )
    environment["NCCL_LL128_NTHREADS" if protocol == "LL128" else "NCCL_NTHREADS"] = str(threads)
    return environment


def _native_arguments(
    config: Mapping[str, Any],
    *,
    family: str,
    requested: Mapping[str, Any],
    diagnostic: bool,
    warmups: int,
    iterations: int,
) -> list[str]:
    useful_bytes = _useful_bytes(requested)
    available = requested["available_sms"]
    return [
        str(Path(config["native_executable"]).resolve()),
        "--stage",
        str(requested["stage"]),
        "--protocol",
        str(requested["protocol"]),
        "--family",
        family,
        "--placement",
        str(requested["simple_placement"]),
        "--working-set",
        str(requested.get("working_set", "reused")),
        "--ranks",
        str(requested["ranks"]),
        "--channels",
        str(requested["active_channels"]),
        "--working-warps",
        str(requested["working_warps"]),
        "--available-sms",
        "0" if available == "full" else str(available),
        "--reservations",
        str(requested.get("reservations", 1)),
        "--useful-bytes",
        str(useful_bytes),
        "--delay-cycles",
        str(requested.get("delay_cycles", 0)),
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
        "--diagnostic",
        "1" if diagnostic else "0",
    ]


def _run_native(
    config: Mapping[str, Any],
    *,
    family: str,
    requested: Mapping[str, Any],
    diagnostic: bool,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        _native_arguments(
            config,
            family=family,
            requested=requested,
            diagnostic=diagnostic,
            warmups=warmups,
            iterations=iterations,
        ),
        env=_native_environment(requested, config),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native probe exited {completed.returncode}: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if value.get("schema") != NATIVE_SCHEMA:
        raise ValueError("native probe returned an unsupported schema")
    return value


def _realized(requested: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(requested)
    device = native["realized"]
    placement_names = {
        1 << 0: "source_default",
        1 << 1: "buffered",
        1 << 2: "buffered_read",
    }
    observed_placements = [
        placement_names.get(mask, f"unknown_mask_{mask}") for mask in device["placement_masks"]
    ]
    result.update(
        actual_protocol_ids=device["protocols"],
        actual_block_threads=device["block_threads"],
        actual_working_warps=device["working_warps"],
        actual_channel_masks=device["channel_masks"],
        actual_active_channels=device["active_channels"],
        actual_placement_masks=device["placement_masks"],
        actual_recv_peers=device["recv_peers"],
        actual_send_peers=device["send_peers"],
        primitive_resident_sms=device["resident_sms"],
        actual_granted_sms=device["granted_sms"],
        input_pointer_distinct=device["input_pointer_distinct"],
        input_pointer_span_bytes=device["input_pointer_span_bytes"],
        allocation_bytes=device["allocation_bytes"],
        buffer_placement=(
            observed_placements[0] if len(set(observed_placements)) == 1 else observed_placements
        ),
    )
    return result


def _peer_mapping_realized(requested: Mapping[str, Any], realized: Mapping[str, Any]) -> bool:
    """Validate source-recorded peers for every channel that actually ran."""

    ranks = int(requested["ranks"])
    recv = realized.get("recv_peers")
    send = realized.get("send_peers")
    masks = realized.get("channel_masks")
    if (
        not isinstance(recv, list)
        or not isinstance(send, list)
        or not isinstance(masks, list)
        or len(recv) != ranks
        or len(send) != ranks
        or len(masks) != ranks
        or any(not isinstance(row, list) or len(row) != 32 for row in recv + send)
    ):
        return False
    p2p_stage = requested["stage"] in {"ready_publication", "reuse"}
    for channel in range(32):
        active = [bool(int(masks[rank]) & (1 << channel)) for rank in range(ranks)]
        if len(set(active)) != 1:
            return False
        if not active[0]:
            continue
        if p2p_stage:
            if ranks != 2 or [recv[0][channel], send[0][channel]] != [-1, 1]:
                return False
            if [recv[1][channel], send[1][channel]] != [0, -1]:
                return False
            continue
        # Each Ring channel must be one directed permutation: no self edges,
        # every peer is in range, and every send is the receiver's inverse.
        for rank in range(ranks):
            recv_peer = recv[rank][channel]
            send_peer = send[rank][channel]
            if (
                recv_peer < 0
                or recv_peer >= ranks
                or send_peer < 0
                or send_peer >= ranks
                or recv_peer == rank
                or send_peer == rank
                or recv[send_peer][channel] != rank
                or send[recv_peer][channel] != rank
            ):
                return False
    return True


def _qualification_failures(
    family: str,
    requested: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    require_counts: bool,
) -> list[str]:
    failures = []
    counts = native["counts"]
    realized = native["realized"]
    protocol_id = {"LL": 1, "LL128": 2, "SIMPLE": 3}[requested["protocol"]]
    ranks = int(requested["ranks"])
    if not native.get("correctness"):
        failures.append("incorrect_output")
    if not native.get("canary_ok") or native.get("unsupported_site"):
        failures.append("instrumentation_canary_or_site_failure")
    if realized["protocols"] != [protocol_id] * ranks:
        failures.append("protocol_not_realized")
    if any(value != int(requested["working_warps"]) for value in realized["working_warps"]):
        failures.append("working_warps_not_realized")
    p2p_stage = requested["stage"] in {"ready_publication", "reuse"}
    working_threads = 32 * int(requested["working_warps"])
    if requested["protocol"] == "SIMPLE":
        expected_blocks = (
            [working_threads + 32, working_threads] if p2p_stage else [working_threads + 32] * ranks
        )
    else:
        expected_blocks = [working_threads] * ranks
    if realized["block_threads"] != expected_blocks:
        failures.append("block_threads_not_realized")
    if any(value != int(requested["active_channels"]) for value in realized["active_channels"]):
        failures.append("active_channels_not_realized")
    resident_sms = realized.get("resident_sms")
    granted_sms = realized.get("granted_sms")
    if (
        not isinstance(resident_sms, list)
        or not isinstance(granted_sms, list)
        or len(resident_sms) != ranks
        or len(granted_sms) != ranks
        or any(
            resident <= 0 or resident > granted
            for resident, granted in zip(resident_sms, granted_sms, strict=True)
        )
    ):
        failures.append("primitive_residency_not_observed")
    if not _peer_mapping_realized(requested, realized):
        failures.append("peer_mapping_not_realized")
    expected_reuse = requested.get("working_set", "reused") == "reused"
    pointer_distinct = realized.get("input_pointer_distinct")
    pointer_span = realized.get("input_pointer_span_bytes")
    if expected_reuse:
        if pointer_distinct != [1] * ranks or pointer_span != [0] * ranks:
            failures.append("reused_working_set_not_realized")
    elif (
        not isinstance(pointer_distinct, list)
        or not isinstance(pointer_span, list)
        or len(pointer_distinct) != ranks
        or len(pointer_span) != ranks
        or any(distinct <= 1 for distinct in pointer_distinct)
        or any(span <= 0 for span in pointer_span)
    ):
        failures.append("rotating_working_set_not_realized")
    if any(value <= 0 for value in native["samples"]["same_device_interval"]):
        failures.append("consumer_device_interval_not_observed")
    # The native result must contain the grant once green contexts are in use;
    # absence is explicit failed capability, never an inferred grant.
    if (
        requested["available_sms"] != "full"
        and realized.get("granted_sms") != [int(requested["available_sms"])] * ranks
    ):
        failures.append("sm_grant_not_realized")
    if require_counts and counts.get("readiness_checks", 0) == 0:
        failures.append("no_readiness_observation")
    if require_counts and counts.get("successful_observations", 0) == 0:
        failures.append("no_successful_readiness_observation")
    if require_counts and counts.get("consumed_steps", 0) == 0:
        failures.append("no_consumed_step")
    if require_counts and counts.get("head_publications", 0) == 0:
        failures.append("no_head_publication")
    if require_counts and family == "already_ready" and counts.get("unsuccessful_checks", 0) != 0:
        failures.append("already_ready_polled_unsuccessfully")
    requested_delay = int(requested.get("delay_cycles", 0))
    if family in {"delayed_publication", "delayed_consumption", "confirmation"}:
        observed = max(native["samples"]["observed_local_delay"])
        if observed < requested_delay:
            failures.append("device_delay_shorter_than_requested")
    useful_bytes = _useful_bytes(requested)
    operations = int(requested.get("reservations", 1))
    expected_source_bytes = useful_bytes * (operations if p2p_stage else ranks)
    if require_counts and counts.get("source_load_bytes") != expected_source_bytes:
        failures.append("source_bytes_not_conserved")
    if require_counts and useful_bytes > 0 and counts.get("tail_publications", 0) == 0:
        failures.append("no_tail_publication")
    if (
        require_counts
        and family == "delayed_consumption"
        and counts.get("consumed_steps", 0) < operations
    ):
        failures.append("fifo_reservations_not_consumed")
    expected_placement = {
        "source_default": 1 << 0,
        "buffered": 1 << 1,
        "buffered_read": 1 << 2,
    }[requested["simple_placement"]]
    if realized.get("placement_masks") != [expected_placement] * ranks:
        failures.append("buffer_placement_not_realized")
    if require_counts and requested["protocol"] == "SIMPLE" and useful_bytes != 0:
        if requested["simple_placement"] == "buffered_read":
            if counts.get("remote_read_bytes", 0) == 0 or counts.get("remote_write_bytes", 0) != 0:
                failures.append("simple_buffered_read_not_realized")
        elif requested["simple_placement"] == "buffered" and (
            counts.get("shared_staging_bytes", 0) == 0
            or counts.get("remote_read_bytes", 0) != 0
            or counts.get("remote_write_bytes", 0) != 0
        ):
            failures.append("simple_buffered_staging_not_observed")
    if (
        require_counts
        and useful_bytes == 0
        and any(
            counts.get(name, 0)
            for name in (
                "source_load_bytes",
                "shared_staging_bytes",
                "remote_read_bytes",
                "remote_write_bytes",
            )
        )
    ):
        failures.append("empty_case_moved_payload")
    return failures


def capability(path: Path, config: Mapping[str, Any], probe_identity: Mapping[str, Any]) -> None:
    request = json.loads(path.read_text(encoding="utf-8"))
    requested = request["requested"]
    try:
        native = _run_native(
            config,
            family=request["family"],
            requested=requested,
            diagnostic=True,
            warmups=2,
            iterations=1,
        )
        failures = _qualification_failures(
            request["family"], requested, native, require_counts=True
        )
        counts = {name: int(native["counts"].get(name, 0)) for name in REQUIRED_SOURCE_COUNTS}
        row = {
            "schema": CAPABILITY_SCHEMA,
            "capability_key": request["capability_key"],
            "qualified": not failures,
            "reason": "qualified" if not failures else ";".join(failures),
            "identity_digest": probe_identity["identity_digest"],
            "realized": _realized(requested, native),
            "source_operation_counts": counts,
        }
    # The capability boundary deliberately turns any native/configuration
    # failure into retained unqualified evidence instead of dropping the key.
    except Exception as error:  # noqa: BLE001
        row = {
            "schema": CAPABILITY_SCHEMA,
            "capability_key": request["capability_key"],
            "qualified": False,
            "reason": f"native_probe_failure:{type(error).__name__}:{error}",
            "identity_digest": probe_identity["identity_digest"],
            "realized": {},
            "source_operation_counts": {},
        }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))


def run(
    path: Path, output: Path, config: Mapping[str, Any], probe_identity: Mapping[str, Any]
) -> None:
    request = json.loads(path.read_text(encoding="utf-8"))
    cell = request["cell"]
    native = _run_native(
        config,
        family=cell["family"],
        requested=cell["requested"],
        diagnostic=False,
        warmups=int(request["warmup_iterations"]),
        iterations=int(request["recorded_iterations"]),
    )
    failures = _qualification_failures(
        cell["family"], cell["requested"], native, require_counts=False
    )
    if failures:
        raise RuntimeError("ordinary realized-control failure: " + ";".join(failures))
    realized = _realized(cell["requested"], native)
    timer_units = {
        "same_device_interval": "ns",
        "cuda_event": "us",
        "host_wall": "us",
    }
    rows = []
    for repetition in range(int(request["recorded_iterations"])):
        for timer in request["timing_boundaries"]:
            rows.append(
                {
                    "schema": OBSERVATION_SCHEMA,
                    "manifest_digest": request["manifest_digest"],
                    "cell_id": cell["cell_id"],
                    "process_id": request["process_id"],
                    "repetition": repetition,
                    "identity_digest": probe_identity["identity_digest"],
                    "requested": cell["requested"],
                    "realized": realized,
                    "family": cell["family"],
                    "phase": "ordinary",
                    "timer": timer,
                    "units": timer_units[timer],
                    "raw_duration": native["samples"][timer][repetition],
                    "observed_local_delay": native["samples"]["observed_local_delay"][repetition],
                    "correctness": True,
                    "qualification": "qualified",
                    "reason": "",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--capability", type=Path)
    modes.add_argument("--run", nargs=2, metavar=("REQUEST", "OUTPUT"))
    args = parser.parse_args()
    config = _config()
    probe_identity = identity(config)
    if args.describe:
        print(json.dumps(probe_identity, sort_keys=True, separators=(",", ":")))
    elif args.capability:
        capability(args.capability, config, probe_identity)
    else:
        run(Path(args.run[0]), Path(args.run[1]), config, probe_identity)


if __name__ == "__main__":
    main()
