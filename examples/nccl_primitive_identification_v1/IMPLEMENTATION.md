# TRAF-94 runner implementation and handoff

## Components

- `manifest.json` is the frozen architecture-independent experiment contract.
- `h100-extension.json` names this node's architecture explicitly without
  changing or relabeling the A100/GH200 design.
- `matrix.py` owns deterministic stage-by-stage expansion, canonical digests,
  capability grouping, row guards, full-campaign completeness, and the causal
  inner-device-within-outer-CUDA-event timer disposition.
- `analysis.py` owns the five-process paired-IQR rule and held-out prediction
  bound, the identification-only component profile, and the locked held-out
  scorer. It intentionally works on process medians rather than treating all
  iterations as independent samples. `ANALYSIS_DESIGN.md` states the candidate
  equation, source-transfer assumptions, uncertainty rule, and claim boundary.
- `record_build.py` binds either a clean pinned checkout or its exactly named
  versioned instrumentation patch, the build command, compiler versions,
  source-file hashes, and output library bytes into one self-digested record.
- `audit_environment.py` performs a cheap pre-pilot source/build/device/topology
  audit. Passing it permits a capability build; it is not primitive capability.
- `run_study.py` writes planned and qualified inventories, builds the process
  schedule, invokes one external probe work item, retains stdout/stderr, hashes
  artifacts, and validates merged raw rows.
- `run_campaign.py` is the node-local serial campaign supervisor. It resumes
  only hash-verified `COMPLETE.json` work, resolves physical GPU UUIDs before
  applying `CUDA_VISIBLE_DEVICES`, and requires a stable idle boundary on the
  selected GPUs before launching each item. It intentionally never overlaps
  two measurements because nominally disjoint rank sets still share NVSwitch.
  Its first post-run action rechecks selected-GPU contexts. A newly visible
  external context writes `CONTAMINATED.json`, stops the campaign, and makes
  both resume and collection reject that directory until the retained attempt
  is moved aside and the same frozen index is rerun.
- `PROBE_CONTRACT.md` defines the required source-faithful CUDA/NCCL executable.
- `nccl-2.31.2-traf94.patch` is the reviewable instrumentation/intervention
  patch for the pinned source, and `SOURCE_PROBE_DESIGN.md` maps each frozen
  family to the exact native execution path.
- `primitive_probe.cu`, `build_probe.py`, and `probe.py` provide the native
  executable, provenance-bound build, and strict contract adapter.

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
  --instrumentation-patch examples/nccl_primitive_identification_v1/nccl-2.31.2-traf94.patch \
  --output /capture/nccl-build-record.json

python examples/nccl_primitive_identification_v1/audit_environment.py \
  --extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --source /path/to/clean/pinned/nccl \
  --library /path/to/clean/pinned/nccl/build/lib/libnccl.so.2.31.2 \
  --build-record /capture/nccl-build-record.json \
  --output /capture/environment-audit.json

python examples/nccl_primitive_identification_v1/build_probe.py \
  --source /path/to/instrumented/pinned/nccl \
  --build-record /capture/nccl-build-record.json \
  --output-dir /capture/probe-build

export TRAF94_PROBE_CONFIG=/capture/probe-build/probe-build.json

python examples/nccl_primitive_identification_v1/run_study.py plan \
  --architecture-extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --output /capture/planned-inventory.json

python examples/nccl_primitive_identification_v1/run_study.py capability-plan \
  --inventory /capture/planned-inventory.json \
  --output /capture/capability-requests.jsonl

python examples/nccl_primitive_identification_v1/run_study.py run-capabilities \
  --capability-plan /capture/capability-requests.jsonl \
  --probe examples/nccl_primitive_identification_v1/probe.py \
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
  --probe examples/nccl_primitive_identification_v1/probe.py \
  --output /capture/work \
  --work-index 0

# Node-local bulk execution is serial and safely resumable. The four physical
# indices may be changed at a work boundary only after rechecking probe identity
# and recording the device-set transition in the external campaign ledger.
python examples/nccl_primitive_identification_v1/run_campaign.py \
  --inventory /capture/qualified-inventory.json \
  --schedule /capture/schedule.json \
  --probe examples/nccl_primitive_identification_v1/probe.py \
  --output /capture/work \
  --visible-devices 0,1,2,3

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

python examples/nccl_primitive_identification_v1/run_study.py fit \
  --inventory /capture/qualified-inventory.json \
  --rows /capture/observations.jsonl \
  --output /capture/identified-parameters.json

python examples/nccl_primitive_identification_v1/run_study.py score \
  --inventory /capture/qualified-inventory.json \
  --rows /capture/observations.jsonl \
  --fit /capture/identified-parameters.json \
  --output /capture/confirmation-score.json
```

Each `run-one --work-index N` command is independent and can be assigned to a
scheduler array only when the scheduler provides exclusive node/fabric access;
do not overlap items on a shared NVSwitch node. Bulk raw rows belong outside
Git. Commit only the frozen
qualified inventory, compact analysis, and a content-hash artifact manifest.
Collection re-hashes each completed work directory and refuses partial or
changed evidence before it writes the merged JSONL file.

Validation preserves raw rows even when a timer boundary is invalid. It emits
`valid_outside_void_scopes` only when campaign completeness and all other row
guards hold, names the complete `(cell, timer)` scope, and binds the report to
the observation-file SHA-256. Analysis must remove every row in a named void
scope; it may not discard only the repetitions that exposed the violation.

The campaign does not poll NVML while an ordinary probe is timing. Such a poll
would be an observer inside the sample and would require its own perturbation
study. Stable pre-run idle samples plus the immediate post-run check instead
bracket every accepted work item. A post-run context is treated conservatively
as possible overlap even though it may have started just after probe exit.

## H100 implementation status

The source-faithful probe is implemented against the pinned NCCL commit and has
run on the eight-H100 NVL8 development node. Representative diagnostics cover
LL, LL128, Simple buffered and Simple DirectRead, empty and nonempty work,
publication/consumption delays, and 8/32-channel sharing. These smoke results
established that the implementation could reach its intended source branches.

The complete 388-request capability sweep is now frozen. It qualified 334
capability classes and 768 of 848 planned cells. The 80 excluded cells comprise
48 unsupported Simple 17-worker-warp geometries, 24 held-out four-rank Simple
DirectRead cells whose native reduction failed correctness, and 8 zero-payload
LL/LL128 delayed-publication cells with no observable producer payload
publication. `h100-capability-freeze.json` records every excluded capability
key and all build, inventory, result, schedule, and smoke-artifact hashes.

One ordinary schedule item also completed the full 20-warmup/100-iteration
path and emitted 300 qualified timer rows. This is an execution-path check, not
the five-process campaign. The 3,840-item ordinary schedule, parameter fit, and
held-out confirmation remain pending and must use the frozen inventory without
adding back an excluded cell.

The exact final build-record, environment-audit, capability inventory, and
schedule hashes are in the progress ledger and capability-freeze record; bulk
captures remain in the external capture directory. Do not reuse the earlier
clean-library SHA after applying instrumentation, and do not substitute the
system library or a stand-alone CUDA copy kernel.
