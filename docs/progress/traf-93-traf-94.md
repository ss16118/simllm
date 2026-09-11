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
- [ ] Implement already-ready, delayed-publication, delayed-consumption,
  empty/nonempty, copy/reduction and working-set primitive probes.
- [ ] Add channel/warp/SM resource-sharing probes after the one-channel pilot.
- [x] Keep diagnostic source-operation counts separate from ordinary timing in
  the runner/probe contract. The device probe still has to emit the counts.
- [x] Capture requested and realized controls, source/build/device identity,
  local timer boundaries, correctness and process exits.
- [x] Implement five-process completeness, paired-IQR resolution and held-out
  prediction checks.
- [ ] Freeze the capability-qualified expanded H100 inventory before ordinary
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

The planned inventory file's byte hash is
`798da79003759346505140aa2b9716043a24d433a622ba9e5801c9857aa41851`;
its canonical self-digest is the value recorded in the verification table.
The 32-line capability-request file's byte hash is
`2293ac3c677af45d8dae3eaadd1d13fc95dd9b6f04d69067530f6a11794fac5b`.

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
