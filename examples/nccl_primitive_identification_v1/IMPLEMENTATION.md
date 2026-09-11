# TRAF-94 runner implementation and handoff

## Components

- `manifest.json` is the frozen architecture-independent experiment contract.
- `h100-extension.json` names this node's architecture explicitly without
  changing or relabeling the A100/GH200 design.
- `matrix.py` owns deterministic stage-by-stage expansion, canonical digests,
  capability grouping, row guards, and full-campaign completeness.
- `analysis.py` owns the five-process paired-IQR rule and held-out prediction
  bound. It intentionally works on process medians rather than treating all
  iterations as independent samples.
- `record_build.py` binds a clean pinned checkout, the exact build command,
  compiler versions, source-file hashes, and output library bytes into one
  self-digested provenance record.
- `audit_environment.py` performs a cheap pre-pilot source/build/device/topology
  audit. Passing it permits a capability build; it is not primitive capability.
- `run_study.py` writes planned and qualified inventories, builds the process
  schedule, invokes one external probe work item, retains stdout/stderr, hashes
  artifacts, and validates merged raw rows.
- `PROBE_CONTRACT.md` defines the required source-faithful CUDA/NCCL executable.

## State machine

```text
manifest + architecture extension
              |
              v
       planned inventory
              |
              v
      capability requests ----> diagnostic probe rows
                                      |
                                      v
                         capability-qualified inventory
                                      |
                                      v
                          frozen five-process schedule
                                      |
                                      v
                        ordinary per-work observations
                                      |
                                      v
                    completeness + contrasts + holdouts
```

Ordinary timing requires an inventory whose state is exactly
`capability_qualified`. The runner verifies its self-digest, the schedule digest,
the manifest digest, the cell identity, the probe identity, the pinned NCCL
commit, and the architecture before launching work.

## Typical commands

```bash
python examples/nccl_primitive_identification_v1/record_build.py \
  --source /path/to/clean/pinned/nccl \
  --library /path/to/clean/pinned/nccl/build/lib/libnccl.so.2.31.2 \
  --build-command 'make -j8 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE=...' \
  --output /capture/nccl-build-record.json

python examples/nccl_primitive_identification_v1/audit_environment.py \
  --extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --source /path/to/clean/pinned/nccl \
  --library /path/to/clean/pinned/nccl/build/lib/libnccl.so.2.31.2 \
  --build-record /capture/nccl-build-record.json \
  --output /capture/environment-audit.json

python examples/nccl_primitive_identification_v1/run_study.py plan \
  --architecture-extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --output /capture/planned-inventory.json

python examples/nccl_primitive_identification_v1/run_study.py capability-plan \
  --inventory /capture/planned-inventory.json \
  --output /capture/capability-requests.jsonl

python examples/nccl_primitive_identification_v1/run_study.py run-capabilities \
  --capability-plan /capture/capability-requests.jsonl \
  --probe /path/to/source-faithful-probe \
  --build-record /capture/nccl-build-record.json \
  --output /capture/capability-artifacts \
  --results /capture/capabilities.jsonl

python examples/nccl_primitive_identification_v1/run_study.py qualify \
  --inventory /capture/planned-inventory.json \
  --capabilities /capture/capabilities.jsonl \
  --output /capture/qualified-inventory.json

python examples/nccl_primitive_identification_v1/run_study.py schedule \
  --inventory /capture/qualified-inventory.json \
  --output /capture/schedule.json

python examples/nccl_primitive_identification_v1/run_study.py run-one \
  --inventory /capture/qualified-inventory.json \
  --schedule /capture/schedule.json \
  --probe /path/to/source-faithful-probe \
  --output /capture/work \
  --work-index 0

python examples/nccl_primitive_identification_v1/run_study.py collect \
  --inventory /capture/qualified-inventory.json \
  --schedule /capture/schedule.json \
  --work-root /capture/work \
  --output /capture/observations.jsonl \
  --artifact-manifest /capture/merged-artifacts.json

python examples/nccl_primitive_identification_v1/run_study.py validate \
  --inventory /capture/qualified-inventory.json \
  --rows /capture/observations.jsonl \
  --output /capture/validation.json
```

Each `run-one --work-index N` command is independent and can be assigned to a
scheduler array. Bulk raw rows belong outside Git. Commit only the frozen
qualified inventory, compact analysis, and a content-hash artifact manifest.
Collection re-hashes each completed work directory and refuses partial or
changed evidence before it writes the merged JSONL file.

## Current H100 gate

The development node has eight H100 80GB HBM3 GPUs with all GPU pairs reported
as NV18 and CUDA 13.0. The convenient installed NCCL is 2.30.7 and the existing
working source checkout is at another commit with local modifications, so both
are disqualified.

As a build prerequisite check, a fresh temporary checkout of pinned commit
`7b83616df3ae082a1f32bb74c27458bfe8153a13` compiled NCCL 2.31.2 for
`sm_90` successfully. The library SHA-256 is
`57161bd381053afad3fab8a717caafe472e6dbbe24e751828930d281bd2f50b9`, and
the build-record self-digest is
`f801de5f55ef02167fe63816a8ae406fe5ca6f9309a3bb550d1ccab0b71aadd2`.
The clean environment audit passes the *capability-build* gate.

No primitive capability or ordinary timing result is claimed yet. The remaining
gate is the source-faithful device probe in `PROBE_CONTRACT.md`, including the
controlled publication/consumption hooks and diagnostic counters. Do not bypass
it by hashing only the runner, by using the system library, or by substituting a
custom CUDA copy kernel.
