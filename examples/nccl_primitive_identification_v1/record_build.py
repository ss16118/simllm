"""Create the immutable source-to-library record required before probe work."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.nccl_primitive_identification_v1.audit_environment import (
    _command,
    _sha256,
    _source_identity,
)
from examples.nccl_primitive_identification_v1.matrix import content_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--build-command",
        required=True,
        help="Exact command used to produce the library; stored as evidence, not executed.",
    )
    parser.add_argument(
        "--instrumentation-patch",
        type=Path,
        help=(
            "Versioned patch intentionally applied to the pinned checkout. "
            "Without this option any dirty source remains fatal."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = _source_identity(args.source)
    if source["head"] != manifest["source_commit"]:
        raise ValueError("source checkout is not at the manifest's pinned commit")
    patch_digest = None
    if source["dirty"]:
        if args.instrumentation_patch is None:
            raise ValueError("source checkout has uncommitted changes")
        patch = args.instrumentation_patch.resolve(strict=True)
        patch_digest = _sha256(patch)
        # Reverse applicability proves that every hunk in the declared patch is
        # present in this checkout.  Source-file hashes below bind the complete
        # resulting bytes, so later extra edits cannot reuse this record.
        reverse = subprocess.run(
            ("git", "apply", "--reverse", "--check", str(patch)),
            cwd=args.source,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if reverse.returncode:
            raise ValueError(
                "instrumentation patch is not exactly applicable in reverse: "
                + reverse.stderr.strip()
            )
    elif args.instrumentation_patch is not None:
        raise ValueError("instrumentation patch was supplied to a clean checkout")
    library = args.library.resolve(strict=True)
    # ldd records the actual shared-library dependencies of these bytes.  It is
    # evidence for reproducing the build, but the output is not interpreted as
    # proof that a later probe loaded this library; the probe must report that
    # independently in --describe mode.
    dependencies = subprocess.run(
        ("ldd", str(library)),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    record = {
        "schema": "simllm-nccl-primitive-build-record-v1",
        "source_commit": source["head"],
        "source_files": source["files"],
        "library_path": str(library),
        "library_sha256": _sha256(library),
        "build_command": args.build_command,
        "nvcc": _command("nvcc", "--version").strip(),
        "cxx": _command("c++", "--version").splitlines()[0],
        "dependencies": dependencies.stdout.splitlines(),
        "dependencies_returncode": dependencies.returncode,
    }
    if patch_digest is not None:
        record["instrumentation_patch_sha256"] = patch_digest
        record["instrumentation_dirty_paths"] = source["dirty"]
    record["build_record_digest"] = content_digest(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_commit": record["source_commit"],
        "library_sha256": record["library_sha256"],
        "build_record_digest": record["build_record_digest"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
