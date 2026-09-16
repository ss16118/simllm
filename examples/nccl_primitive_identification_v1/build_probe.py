"""Build and bind the TRAF-94 native probe to one recorded NCCL library."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.audit_environment import _sha256
from examples.nccl_primitive_identification_v1.matrix import content_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="h100")
    parser.add_argument("--probe-source", type=Path, default=HERE / "primitive_probe.cu")
    args = parser.parse_args()

    record = json.loads(args.build_record.read_text(encoding="utf-8"))
    unsigned = dict(record)
    recorded_digest = unsigned.pop("build_record_digest", None)
    if recorded_digest != content_digest(unsigned):
        raise ValueError("build record digest mismatch")
    patch_digest = record.get("instrumentation_patch_sha256")
    if not patch_digest:
        raise ValueError("native probe requires the recorded instrumentation patch")
    library = Path(record["library_path"]).resolve(strict=True)
    if _sha256(library) != record["library_sha256"]:
        raise ValueError("NCCL library bytes changed after their build record")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = output_dir / "primitive_probe_native"
    flags = [
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    command = [
        "nvcc",
        *flags,
        f"-I{args.source / 'build/include'}",
        f"-I{args.source / 'src/include'}",
        str(args.probe_source.resolve(strict=True)),
        f"-L{library.parent}",
        "-lnccl",
        "-lcuda",
        f"-Xlinker=-rpath={library.parent}",
        "-o",
        str(executable),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"nvcc exited {completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )

    config = {
        "schema": "simllm-nccl-primitive-probe-build-v1",
        "architecture": args.architecture,
        "source_commit": record["source_commit"],
        "source_files": record["source_files"],
        "instrumentation_patch_sha256": patch_digest,
        "nccl_library": str(library),
        "library_sha256": record["library_sha256"],
        "native_executable": str(executable),
        "native_executable_sha256": _sha256(executable),
        "probe_source": str(args.probe_source.resolve()),
        "probe_source_sha256": _sha256(args.probe_source),
        "compiler": record["cxx"],
        "cuda": record["nvcc"],
        "flags": flags,
        "build_record_digest": record["build_record_digest"],
    }
    config["config_digest"] = content_digest(config)
    config_path = output_dir / "probe-build.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "native_executable": str(executable),
                "native_executable_sha256": config["native_executable_sha256"],
                "config": str(config_path),
                "config_digest": config["config_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
