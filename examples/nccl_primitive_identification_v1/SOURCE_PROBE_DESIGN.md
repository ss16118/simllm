# TRAF-94 source probe design

## Purpose

This document describes the executable that closes the device-measurement gap
in the frozen TRAF-94 experiment.  It is intentionally specific about what is
measured and what must cause a capability failure.  A successful NCCL call is
not, by itself, evidence that the requested primitive specialization ran.

The probe is built against NCCL commit
`7b83616df3ae082a1f32bb74c27458bfe8153a13`.  A small, versioned patch adds a
diagnostic control block to that exact source tree.  The patch does not replace
the LL, LL128, or Simple primitive implementations: it observes their existing
readiness, publication, source-load, staging, remote-access, and consumption
sites.  Interventions enter at those sites so the measured work continues to
execute NCCL's primitive code and memory-order instructions.

## Ordinary and diagnostic builds

The patch supports two run modes selected before communicator construction:

- **ordinary** executes the intervention but does not update diagnostic
  counters.  These runs provide timing samples;
- **diagnostic** executes identical primitive and intervention branches and
  additionally updates counters and timestamps.  These runs only decide
  whether a specialization is capable of producing ordinary evidence.

The executable and library hashes identify the patched build.  The canonical
patch digest and the pristine upstream source-file hashes are also recorded.
The build record therefore distinguishes this experimental NCCL from both the
unmodified pinned library and any other dirty checkout.

## Device control block

Each rank owns a device-resident `ncclTraf94Control` block.  Its immutable
request section contains the requested family, delay, diagnostic flag, and
expected protocol/placement.  Its result section contains counters and
device-local timestamps.  `ncclTraf94SetControl` copies the block pointer into
that communicator's `ncclKernelComm`; a null pointer restores normal NCCL
behavior.

The control is deliberately per communicator rather than global.  This avoids
rank races, lets the harness assign producer and consumer roles explicitly,
and makes every instrumented access follow the same device pointer that the
launched NCCL kernel already receives.

## Mapping from frozen family to source operation

| Frozen family | Device intervention and diagnostic proof |
|---|---|
| `already_ready` | NCCL still receives the matched send/receive calls together, as its rendezvous requires. At the first consumer readiness source site, the consumer waits for the producer's exact publication witness, discards any pre-gate stale load, starts its local interval, and reloads. The diagnostic must record zero failed post-gate observations. |
| `delayed_publication` | The first consumer readiness check publishes a device start flag.  The producer publication site observes it, delays by local `clock64()`, records its own elapsed cycles, and then executes the original publication instruction.  Delay zero uses the same branch and handshake. |
| `delayed_consumption` | After readiness succeeds, all primitive workers wait by local `clock64()` before data consumption and before the original head-return instruction.  The producer's live FIFO reservation therefore remains unavailable. |
| empty/nonempty | The harness always submits a positive-size NCCL work item.  For requested zero useful bytes, the source hook sets the primitive's payload element count to zero only after launch/work selection.  The existing source-prescribed wait/post path still executes and the payload counters must stay zero. |
| `copy` / `sum` | Both use the same sum-capable primitive specialization and geometry.  A source hook selects identity-copy or the existing float32 sum at the `applyReduce` sites.  The diagnostic records source input count and bytes; geometry equality is checked across the pair before qualification. |
| reused/rotating working set | The harness changes only the initialized user-buffer offset.  The source operations, useful bytes, and geometry must match.  All offsets are touched before warmup. |
| resource sharing | Channel and protocol selection use the frozen NCCL controls; available SMs use CUDA green contexts. One independent Ring segment is submitted per requested channel so NCCL's small-message tuner cannot collapse the launch. The probe records the granted SM resource and an in-kernel SM-ID bitset written by the primitive block itself. |
| held-out confirmation | Confirmation is a segmented Ring float32 sum with the delayed-publication intervention. Its ranks, channel count, SM grant, bytes per channel, and delay come only from the held-out stage. It is not a new primitive family or a refit opportunity. |

## Protocol and placement controls

`NCCL_ALGO=RING`, `NCCL_PROTO=<LL|LL128|SIMPLE>`, and channel controls are set
in the child process before NCCL initialization. Ready-publication and reuse
use NCCL's public P2P send/receive path; data, sharing, and confirmation use
Ring AllReduce. The patch extends the P2P scheduler's explicit protocol field
to LL128 and sizes a one-direction P2P block from the requested working warps.
The diagnostic reads the actual primitive protocol, channel id/count, work
warps, block size, and connection-placement mask from the executing kernel. A
mismatch is unqualified, never relabeled.

Simple `buffered` uses the connection FIFO. `buffered_read` uses `ncclMemAlloc`
exportable VMM allocations, registers both user buffers, and requires the
source `DirectRead` flag. For Ring DirectRead the harness initializes the
distinct output allocation with the rank-local operand because that source
path pulls from output after a primitive acquires both receive and send roles.
The buffers remain distinct so NCCL performs both registrations. The primitive
records its actual placement even for empty work; byte counters independently
prove that nonempty read placement moved remote-read and no remote-write bytes.

LL and LL128 do not have a Simple placement choice.  The manifest expander
uses `not_applicable`; the probe must reject any other value for those
protocols.

## Timers and process boundaries

One probe invocation handles exactly one frozen cell and creates fresh NCCL
communicators.  It performs 20 warmups, synchronizes, and records 100
iterations.  Allocation, buffer initialization, communicator construction,
registration, correctness checking, and JSON writing are outside the measured
service interval.

For each repetition the probe writes three rows:

1. `same_device_interval`: consumer-side `%globaltimer` boundaries captured by
   the primitive block, in nanoseconds;
2. `cuda_event`: events recorded in the consumer stream;
3. `host_wall`: `steady_clock` around the same issued operation and stream
   completion.

`clock64()` is used only for the local delay intervention and its observed
cycle count. It is not used for a multi-channel interval because different
blocks can run on different SM-local clocks. No rank ever subtracts a timestamp
written by another GPU.

## Qualification guards

A capability is qualified only when all of the following hold:

- the loaded library, probe executable, patch, pristine source files, CUDA,
  driver, UUIDs, and topology match the identity record;
- the actual protocol, placement, channel count, block threads, work warps,
  rank count, SM grant, and primitive residency match the request;
- the expected readiness/publication/consumption operations execute and all
  required counters are present;
- source, staging, direct-remote, and useful byte counts conserve the requested
  work;
- device delays use the requested branch and the observed local delay is not
  shorter than requested;
- the result passes a post-timing float32 correctness check; and
- diagnostic canaries report no control-block overrun or unsupported source
  site.

Ordinary rows repeat the realized geometry and byte guards from a lightweight
non-counter result block.  If they differ from the qualified pilot, the rows
are retained as unqualified and the campaign completeness gate fails.

## Explicit non-goals

This probe does not infer primitive work from NCCL debug text, requested
environment variables, a stand-alone CUDA copy kernel, or a host sleep.  It
does not use cross-GPU timestamps.  It does not claim that a CUDA SM sampler is
the residency of the NCCL block.  Those shortcuts can be useful diagnostics,
but none satisfies the frozen TRAF-94 contract.

## Frozen H100 capability outcome

The final eight-H100 sweep ran all 388 support-sensitive pilots against one
probe identity. It qualified 334 pilots and froze 768 of the 848 planned cells.
The capability gate intentionally excluded three source-observed limits:

- Simple requests for 17 worker warps realize only 16 worker warps plus the
  synchronization warp in this pinned NCCL source;
- four-rank Simple DirectRead Ring reductions produce an incorrect value and
  therefore cannot contribute timing evidence; and
- zero-payload LL/LL128 delayed-publication work has no producer payload
  publication at which to observe the requested delay.

These are capability outcomes, not runner errors. Their exact keys and artifact
hashes are frozen in `h100-capability-freeze.json`. Future campaign code must
consume the 768-cell qualified inventory and must not weaken a guard or relabel
an excluded cell to increase coverage.
