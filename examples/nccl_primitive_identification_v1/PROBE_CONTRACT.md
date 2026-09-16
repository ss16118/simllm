# TRAF-94 source-faithful probe contract

This document defines the boundary between the manifest runner and the CUDA/NCCL
executable. A probe is not qualified merely because it emits syntactically valid
JSON. It must execute the pinned NCCL primitive specialization and preserve the
source operations named in the frozen design.

## Why the boundary exists

Python is responsible for deterministic expansion, process order, artifact
identity, fatal guards, completeness, and statistics. The probe is responsible
for device-local interventions and measurements. This separation prevents the
orchestrator from approximating a device delay with a CPU sleep or manufacturing
realized geometry from requested environment variables.

The runner strips inherited `NCCL_*` variables from ordinary timing processes.
Any NCCL control used by the probe must be in its build/identity record or the
frozen request.

## Command interface

The runner invokes a probe in three modes.

### `probe --describe`

Print one JSON object to stdout and nothing else. Required shape:

```json
{
  "schema": "simllm-nccl-primitive-probe-identity-v1",
  "source_commit": "40 hexadecimal characters",
  "architecture": "h100",
  "build": {
    "compiler": "...",
    "cuda": "...",
    "flags": ["..."],
    "executable_sha256": "...",
    "nccl_library_sha256": "..."
  },
  "device": {
    "uuids": ["..."],
    "driver": "...",
    "topology_digest": "..."
  },
  "source_files": [{"path": "...", "sha256": "..."}],
  "identity_digest": "SHA-256 of the canonical object without this field"
}
```

The source identity must include the primitive, memory-order, reduction, and
launch-selection sources actually compiled. The executable and loaded NCCL
library hashes are distinct because a correct source checkout can still load an
unrelated system library. The runner compares this identity with the
self-digested record produced by `record_build.py`; ordinary timing must also use
the exact identity digest that passed the capability pilot.

### `probe --capability REQUEST.json`

Run a diagnostic-only source-conformance pilot for the requested specialization
and print one `simllm-nccl-primitive-capability-v1` object. It includes the
unchanged `capability_key`, `qualified` boolean, explicit `reason`,
`identity_digest`, nonempty `realized` controls when qualified, and
`source_operation_counts`.

Diagnostic counters must cover the relevant readiness checks, successful
observations, tail/head publications, source loads, shared staging, remote
reads/writes, and consumed steps. A family may report a zero for an operation
that is source-prescribed but not executed in that case; it may not omit the
counter. Diagnostics do not supply ordinary timing samples.

### `probe --run REQUEST.json OUTPUT.jsonl`

Run one fresh process for one frozen cell. Perform exactly 20 untimed warmups and
100 recorded iterations unless the request says otherwise. Emit one ordinary
observation per requested timer boundary per iteration. Allocation,
communicator construction, initialization, and correctness checks stay outside
the device-service interval.

Every row follows `manifest.json.required_row_fields` and uses schema
`simllm-nccl-primitive-observation-v1`. `requested` must equal the cell's frozen
request byte-for-byte. `realized` records observed channels, work warps, block
threads, SM grant/residency evidence, bytes, peer mapping, and buffer placement.
Qualified rows use an empty reason; other rows retain an explicit reason.

## Device intervention requirements

- Already-ready establishes data and matching sequence publication with a
  device dependency before the consumer timer begins. Its diagnostic has zero
  unsuccessful checks.
- Delayed publication uses a device-side consumer-start signal. The producer
  performs the requested local `clock64()` delay and records the observed local
  delay. Delay zero follows the identical signal path. Device-service interval
  boundaries use same-GPU `%globaltimer` nanoseconds so blocks on different SMs
  cannot create an invalid SM-local clock subtraction.
- Delayed consumption waits only after readiness and before consumption/head
  return. It must keep the live FIFO reservation unavailable to the producer.
- Empty cases execute source-prescribed control steps but perform no payload
  movement. A convenient zero-byte API call that bypasses the primitive is not
  equivalent. Simple placement is proven by the executing connection flags in
  this case, not by requiring a nonzero payload-byte counter.
- Copy and float32 sum retain the same geometry. Sum records input count and
  source loads so memory work is not mislabeled arithmetic.
- Rotating working sets are initialized and warmed before timing. The reuse and
  rotation cases retain identical useful bytes and primitive operations.
- Restricted-SM cases record the granted resource and observed kernel
  residency. A separate SM sampler is capability evidence only, not proof of
  primitive residency.

All consumer intervals are timed locally. CUDA events and host wall intervals
are separate rows. Cross-GPU timestamp subtraction is forbidden.

## Failure behavior

Unsupported specialization, source mismatch, unrealized control, incorrect
output, missing counter, process timeout, and incomplete rows are retained as
unqualified or void evidence. They are never silently dropped or imputed. A
failed work directory has no `COMPLETE.json`; the runner will not overwrite it.
