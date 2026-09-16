# TRAF-93 and TRAF-94 progress

This document is the working ledger for the Simple buffered-read implementation
and the standardized NCCL primitive experiment. It records what is implemented,
what was verified, and what remains open. Hardware results are not claimed until
the corresponding frozen campaign is complete and its guards pass.

## Scope and order

1. **TRAF-93:** implement `buffered_read` for Simple on the retained GPU/NVLink
   calendar while preserving the existing LL, LL128 and source-off paths.
2. **TRAF-94:** implement the manifest-driven primitive runner, capability and
   source-conformance diagnostics, raw result schema, and analysis gates.
3. Run hardware measurements only after the runner inventory and capability
   outcomes have been frozen as required by the existing v1 experiment design.

The normative inputs are:

- `examples/nccl_simple_read_v1/expectations.md` and `expectations.json`;
- `docs/design/nccl-primitive-identification-v1.md`;
- `examples/nccl_primitive_identification_v1/manifest.json`.

## TRAF-93 checklist

- [x] Read the frozen expectations and inspect the retained packet and NCCL
  runtime boundaries.
- [x] Confirm that component packetization already represents a peer read as a
  zero-payload request followed by payload-bearing response packets.
- [x] Admit peer reads on the retained packet calendar without weakening the
  existing peer-write preflight.
- [x] Add a one-shot response gate so owner-memory service becomes eligible
  only after request visibility and responses become eligible only after that
  service completes.
- [x] Extend `NcclRingProgram` and `NcclExecutionConfig` with explicit
  `buffered_read`, applying it only to Simple payload movement.
- [x] Emit read-specific source bindings that preserve requester, buffer owner,
  connection sequence, stripe, request and response identity.
- [x] Preserve empty-slice control work without inventing payload reads.
- [x] Reject placement changes on an existing communicator and unsupported
  registered/direct/proxy branches before admission.
- [x] Add exact isolated-delay, memory-rate, FIFO-reuse, source-bypass and
  collective/graph tests from the frozen expectations.
- [x] Add and run the `examples/nccl_simple_read_v1` mechanism study.

## TRAF-94 checklist

- [x] Define the expanded-cell representation and deterministic manifest digest.
- [x] Implement sequential stage expansion rather than a blind Cartesian
  product.
- [x] Implement required result-row validation and explicit qualification
  reasons.
- [x] Implement already-ready, delayed-publication, delayed-consumption,
  empty/nonempty, copy/reduction and working-set primitive probes.
- [x] Add channel/warp/SM resource-sharing probes after the one-channel pilot.
- [x] Keep diagnostic source-operation counts separate from ordinary timing in
  the runner/probe contract, and emit them only from diagnostic native runs.
- [x] Capture requested and realized controls, source/build/device identity,
  local timer boundaries, correctness and process exits.
- [x] Implement five-process completeness, paired-IQR resolution and held-out
  prediction checks.
- [x] Freeze the capability-qualified expanded H100 inventory before ordinary
  timing. H100 is an extension; it is not silently labeled A100 or GH200.
- [ ] Run and publish the hardware campaign.

## Verification log

| Date | Branch | Evidence | Result |
|---|---|---|---|
| 2026-09-11 | `traf-93-simple-read` | Repository and frozen-design inspection | Packetizer read support exists; retained read admission and source-memory response gating are missing. |
| 2026-09-11 | `traf-93-simple-read` | Focused retained packet/NCCL tests | 221 tests passed while the implementation was being integrated. |
| 2026-09-11 | `traf-93-simple-read` | Expanded packet/NCCL regression set | 368 passed; one assertion expected the old validation wording and was corrected without changing behavior. |
| 2026-09-11 | `traf-93-simple-read` | Final combined packet/NCCL regression (`PYTHONPATH=. .venv/bin/pytest -q ...`) | 370 passed in 14.89 seconds. |
| 2026-09-11 | `traf-93-simple-read` | `nccl_simple_read_v1/run_study.py` | All fatal guards valid: 9 isolated-read, 6 remote-memory, 288 collective and 32 graph-metric rows. |
| 2026-09-11 | `traf-93-simple-read` | Ruff over every changed Python file; `git diff --check` | Passed. |
| 2026-09-11 | `traf-94-primitive-measurements` | H100 planned expansion | 848 sequential timing cells and 32 source-specialization capability pilots; planned inventory digest `8dc3ac3ca36bfa42b70c52ea21d601fc88273d7a9b24b54de632321189bbb1b4`. |
| 2026-09-11 | `traf-94-primitive-measurements` | Local H100 environment audit | Not qualified: working NCCL checkout is at the wrong commit and dirty; installed NCCL 2.30.7 lacks a verified binding to the pinned 2.31.2 source. No timing launched. |
| 2026-09-11 | `traf-94-primitive-measurements` | Matrix, qualification, schedule, mock process, artifact collection, completeness, paired-IQR and holdout tests | 8 passed in 1.05 seconds. Mock evidence is test-only and carries no hardware claim. |
| 2026-09-11 | `traf-94-primitive-measurements` | Repository-wide pytest attempt | Interrupted during a long study section after 2,144 passed and 14 skipped. One failure was a generic `run_study` module-name collision introduced by the new test and is fixed; its exact regression test now passes. The other was an environment-dependent system-`pip` wheel build and passes when the virtual-environment `pip` is first on `PATH`. |
| 2026-09-11 | `traf-94-primitive-measurements` | Combined TRAF-93/TRAF-94 and discovered-regression set with venv `pip` on `PATH` | 380 passed in 17.41 seconds; Ruff and `git diff --check` passed. |
| 2026-09-11 | `traf-94-primitive-measurements` | Clean pinned NCCL build for `sm_90` | NCCL 2.31.2 built successfully from `7b83616df3ae082a1f32bb74c27458bfe8153a13`; library SHA-256 `57161bd381053afad3fab8a717caafe472e6dbbe24e751828930d281bd2f50b9`. |
| 2026-09-11 | `traf-94-primitive-measurements` | Build-record plus clean H100 environment audit | Capability-build gate passed; build-record digest `f801de5f55ef02167fe63816a8ae406fe5ca6f9309a3bb550d1ccab0b71aadd2`. This is not primitive capability or timing evidence. |
| 2026-09-16 | `traf-94-primitive-measurements` | Refreshed `upstream/main` and compared all TRAF-94 paths | Upstream has no newer probe or hardware campaign to reuse; the source-faithful executable remains the missing implementation. |
| 2026-09-16 | `traf-94-primitive-measurements` | `SOURCE_PROBE_DESIGN.md` | Froze the implementation mapping from every manifest family to an actual pinned-NCCL source site, including ordinary/diagnostic separation and fatal qualification guards. |
| 2026-09-16 | `traf-94-primitive-measurements` | Expanded-inventory type audit | Found and fixed the confirmation-stage default being emitted as a one-element JSON list instead of the scalar family `confirmation`; the H100 inventory must be regenerated and re-frozen before timing. |
| 2026-09-16 | `traf-94-primitive-measurements` | Capability-request audit | Added the deterministic representative cell controls to each grouped capability request; the earlier request named a source family but did not contain enough values to launch or verify a real device pilot. |
| 2026-09-16 | `traf-94-primitive-measurements` | Capability grouping audit | Replaced 32 overly broad family-only pilots with 388 support-sensitive pilots covering rank/channel/warp/SM geometry, FIFO reservations, empty work, and rotating allocation. The strongest delay/size member represents only support-equivalent timing cells. |
| 2026-09-16 | `traf-94-primitive-measurements` | Pinned NCCL source probe implementation | Added a versioned patch over NCCL `7b83616d`: LL/LL128/Simple source counters and interventions, explicit P2P protocol/warp controls, Simple buffered/DirectRead selection, same-device `%globaltimer` intervals, and realized protocol/channel/block/worker/placement/SM evidence. |
| 2026-09-16 | `traf-94-primitive-measurements` | Native harness and strict adapter | Added `primitive_probe.cu`, `build_probe.py`, and `probe.py`; the harness uses public NCCL P2P for readiness/reuse and segmented Ring AllReduce for data/sharing/confirmation, exportable `ncclMemAlloc` buffers, CUDA green contexts, three timer boundaries, and post-timing exact correctness. |
| 2026-09-16 | `traf-94-primitive-measurements` | H100 source-path smoke matrix | LL, LL128, Simple buffered and Simple DirectRead passed already-ready, delayed-publication, delayed-consumption, empty/nonempty, sum, 8/32-channel sharing, and observed-delay checks. A wider four-rank Simple DirectRead sum is correctly retained as unsupported because the forced pinned-source path produces an incorrect result; buffered Simple and LL four-rank confirmation pass. |
| 2026-09-16 | `traf-94-primitive-measurements` | Final instrumented NCCL/probe provenance | NCCL 2.31.2 `sm_90` library SHA-256 `fe444e4a9ebe18e99ba31e10bff50ee2b4ae1c3c7de3775b20baac484c43dae0`; instrumentation patch SHA-256 `c53bb397ad92f7bec46cc5b6d94c34070b51c87d2db865a8c1afc840c2d6547f`; build-record digest `3beb1d0935c9a685173fb398f097784c8111634f6abc21ce03c20fcf74b0ef12`; probe identity `8612974c171d7de4d39fbecef383d4ecf8402754c1a368d95fb911b5ad8423e8`; environment gate passed. |
| 2026-09-16 | `traf-94-primitive-measurements` | Full H100 capability sweep and inventory freeze | All 388 capability requests completed: 334 qualified and 54 explicitly rejected. The resulting 768-cell inventory digest is `a0915efc4442aa290aaf82f8de48b3327e9c62d3ff94d4fdd8563fbdb2eaefa7`; the exact hashes and rejected keys are in `h100-capability-freeze.json`. |
| 2026-09-16 | `traf-94-primitive-measurements` | Ordinary-path smoke work item | Work index 0 completed 20 warmups and 100 measured repetitions, emitting 300 qualified rows across the three required timers; observation SHA-256 `1940c6b100447ef3e0be8bdf9e7ce0e434ec7e79a3146e0e9fd8dc0b34141b72`. This proves the normal runner path, not completion of the 3,840-item campaign. |
| 2026-09-16 | `traf-94-primitive-measurements` | Focused TRAF-93/TRAF-94 regression suite | 242 passed in 10.57 seconds; Ruff and `git diff --check` passed. |
| 2026-09-17 | `traf-94-primitive-measurements` | Resumable ordinary-campaign supervisor | Added selected-physical-GPU UUID resolution, stable-idle gating, hash-verified resume, atomic status records, and strict serial execution so two probes cannot contend on the shared NVSwitch. Four focused supervisor tests plus the existing TRAF-94 tests pass: 13 passed in 1.39 seconds; Ruff and `git diff --check` passed. |
| 2026-09-17 | `traf-94-primitive-measurements` | In-progress frozen ordinary campaign | 467 of 3,840 schedule items have hash-complete work directories and zero have `FAILED.json`. The live supervisor is waiting at work index 467 because external jobs occupy the selected GPUs; it will resume only after the configured stable-idle boundary. This is progress evidence, not a parameter or completion claim. |
| 2026-09-17 | `traf-94-primitive-measurements` | Inner-timer fatal-guard audit at work boundary 467 | Found 700 rows across seven empty already-ready LL/LL128 cell scopes whose source-local start was absent, producing a node-uptime-sized interval near `1.78959e18` ns. Every other completed inner interval was contained by its matching CUDA event. The campaign was stopped safely, an exact-unit and causal-nesting guard was added, raw evidence was retained, and the affected complete `(cell, same_device_interval)` scopes are explicitly void rather than selectively dropped. The unchanged probe identity and frozen schedule then resumed from all 467 hash-verified work items. Fourteen focused tests, Ruff, and `git diff --check` pass. |
| 2026-09-17 | `traf-94-primitive-measurements` | Identification/confirmation analysis lock | Added a self-digested identification-only nonparametric component profile, explicit resolved-versus-joint contrast ledger, conservative anchor-IQR propagation, source-constrained Ring transfer, and a separately invoked held-out scorer that rejects another fit, inventory, or observation hash. The design and equation are in `ANALYSIS_DESIGN.md`. A compact synthetic end-to-end case proves that confirmation rows do not enter the fit and that a resolved held-out delta is scored only after the lock. Thirty focused/documentation tests, Ruff, and `git diff --check` pass. |
| 2026-09-17 | `traf-94-primitive-measurements` | External-contention boundary audit | The existing supervisor established stable idle before every launch and queried selected GPUs again at the next work boundary. Parsing that evidence found 16 completed jobs followed by a newly visible external context; overlap is not established, but isolation is not provable for those jobs. Added an explicit immediate post-run check and `CONTAMINATED.json` rejection without placing an NVML observer inside timed work. The 16 conservative-risk jobs and the original pre-campaign smoke item are retained in external quarantine and rerun under the strengthened bracket. |
| 2026-09-17 | `traf-94-primitive-measurements` | Compact result publication path | Added `publish_results.py` to verify and cross-bind the merged-artifact, validation, fit, score, inventory, observation, and self-digest identities. It publishes only the load-bearing parameter surfaces plus grouped separability and held-out outcomes, while leaving bulk rows and the full per-process fit external. A mutation test proves that another observation hash is rejected. Thirty-one focused/documentation tests, Ruff, and `git diff --check` pass. |
| 2026-09-17 | `traf-94-primitive-measurements` | Broader NCCL/NVLink/NVSwitch regression | The explicit integration set covering primitive identification, channel FIFO/steps, NCCL stack and demo, DGX NVLink, htsim NVLink, native NVSwitch, and documentation completed with 337 passed in 87.89 seconds. |

## Commands

Commands and exact results will be added here as implementation gates run. Raw
hardware captures must remain outside Git; only compact summaries and their
content-hash manifest belong in the repository.

TRAF-94 planning/audit commands run on 2026-09-11:

```bash
python examples/nccl_primitive_identification_v1/audit_environment.py \
  --extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --source /home/siyshen/workspace/nccl \
  --library /lib/x86_64-linux-gnu/libnccl.so \
  --output /tmp/traf94-h100-environment-audit.json

python examples/nccl_primitive_identification_v1/run_study.py plan \
  --architecture-extension examples/nccl_primitive_identification_v1/h100-extension.json \
  --output /tmp/traf94-h100-planned.json

python examples/nccl_primitive_identification_v1/run_study.py capability-plan \
  --inventory /tmp/traf94-h100-planned.json \
  --output /tmp/traf94-h100-capabilities.jsonl
```

Those commands document the superseded 2026-09-11 planning pass. The completed
2026-09-16 source-probe pass expanded 848 cells into 388 support-sensitive
capability requests and froze 768 qualified cells. Its compact record is
`examples/nccl_primitive_identification_v1/h100-capability-freeze.json`; raw
requests, native stdout/stderr, qualified inventory, schedule, and observation
rows remain in the external capture area named by their SHA-256 hashes.

The remaining hardware work is deliberately separate: execute all 3,840 frozen
ordinary schedule items, collect the five process records per cell, fit only the
identification stages, and score the held-out confirmation stage. Until those
steps pass, TRAF-94 has a complete runner and frozen capability inventory but no
published parameter or accuracy claim.

The active H100 campaign uses the serial `run_campaign.py` supervisor. Its
external status, work directories, launch ledger, and log are retained under
`/home/siyshen/workspace/traf94-campaign-final5`; that host-local path is an
operator handoff locator, not a portable published artifact. The eventual
merged-artifact manifest will bind the portable result to every completed work
marker and to the observation-file hash.

## Handoff notes for future engineers and agents

- The retained packet engine is the single clock owner. Request-visibility
  callbacks may enqueue GPU work on that calendar but must never advance it
  recursively.
- A read response is intentionally ineligible until
  `release_read_response(extent_id)` is called. The call is one-shot and is
  legal only after the request is consumer-visible. Do not replace this gate
  with a duration added to packet transport: doing so loses owner-memory
  contention and causal attribution.
- For a Simple read, the resident block belongs to the requester while the
  memory cursor belongs to the source/buffer owner. `memory_rank` represents
  precisely this split; it does not imply a resident block on the owner.
- `buffered_read` is communicator placement, not a fourth protocol. LL and
  LL128 still execute writes on that communicator; only Simple payload motion
  changes direction.
- The critical-path packet-only reporter cannot describe an external GPU
  service gate and therefore rejects this path before admission. Extend its
  evidence model before relaxing that guard.
- TRAF-94 must not turn an unqualified H100 run into a result merely because
  this checkout happens to be on H100 hardware. Freeze the realized source,
  build, device, and expanded-cell inventory first.
