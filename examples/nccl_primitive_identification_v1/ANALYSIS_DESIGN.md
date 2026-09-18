# TRAF-94 parameter fit and confirmation design

This document defines the post-capture analysis implemented by `analysis.py`
and the `fit` and `score` commands in `run_study.py`. It separates measured
component anchors, source-constrained transfer assumptions, and held-out
acceptance. A good confirmation score does not turn a transfer assumption into
a directly measured physical constant.

## Evidence boundary

Collection must first verify every per-work artifact hash and produce the
complete frozen row set. Validation checks row identity, requested and realized
controls, process/repetition/timer completeness, exact timer units, and timer
scope guards. Raw rows are immutable.

If the source-local interval exceeds the matching outer CUDA-event interval,
validation retains both rows and voids the complete
`(cell, same_device_interval)` scope. The fit removes every row in that scope;
it never removes only the process or repetition that exposed the violation.
CUDA-event and host-wall scopes remain separate evidence when their own guards
hold.

All analysis durations are converted to nanoseconds:

- `same_device_interval`: nanoseconds, multiplier 1;
- `cuda_event`: microseconds, multiplier 1,000;
- `host_wall`: microseconds, multiplier 1,000.

Each cell is summarized in two levels. The 100 recorded iterations first
produce one median per process. The reported central value, quartiles, IQR, and
paired interventions use those five process medians. Iterations are never
treated as 500 independent samples.

## Identification profile

`run_study.py fit` reads a complete row file but selects only cells whose stage
is not `confirmation`. It writes a self-digested profile containing:

- every eligible identification-cell process-median anchor;
- delayed-publication minus already-ready contrasts;
- sum minus copy contrasts;
- rotating minus reused working-set contrasts;
- adjacent FIFO-delay and reservation-count contrasts;
- adjacent channel, SM-grant, and worker-warp contrasts;
- a pooled nonempty publication response for each timer, protocol, placement,
  and requested delay.

A contrast is a separate term only when its absolute paired-process median is
greater than twice the sum of the two cell IQRs. Otherwise it remains a named
joint interval. The profile reports both counts; an unresolved response is not
silently renamed as a polling, publication, arithmetic, or memory constant.

The profile is nonparametric. Measured cell medians are service-cost anchors,
not samples for a free-offset regression. This is deliberate: the matrix can
identify useful component response surfaces and matched deltas, but it does not
justify one scalar offset for every protocol, width, and resource condition.

## Locked confirmation candidate

Confirmation uses one deterministic candidate frozen inside the fit artifact.
For a held-out cell, define:

- `D(B)`: linear interpolation/extrapolation of the rank-2, one-channel,
  `sum`, reused-working-set data anchor at `B` useful bytes;
- `S(C,M,W)`: linear interpolation in channel count of the rank-2 `sum`
  sharing anchor at 1,920 bytes per channel, exact requested SM grant `M`, and
  exact working-warp count `W`;
- `P(L)`: linear interpolation of the pooled nonempty delayed-minus-ready
  publication response at delay `L`.

For ranks `R`, active channels `C`, bytes per channel `B`, SM grant `M`, worker
warps `W`, and delay `L`, the prediction is:

```text
rank2_no_delay = S(C,M,W) + C * (D(B) - D(1920))
prediction     = (R - 1) * rank2_no_delay + P(L)
```

The `(R - 1)` factor is a source-constrained Ring step transfer from the
rank-2 identification cells. It is not independently measured by the v1
identification stages. Publication is added once because the source
intervention's per-rank atomic permits only the first producer publication to
apply the controlled delay. No prediction is clamped; a nonphysical
extrapolation must remain visible as a candidate failure.

Anchor IQRs use the same interpolation or extrapolation weights in absolute
value. Payload-component bounds are added, sharing and payload bounds receive
the source rank multiplier, and the publication bound is added once. This is a
conservative propagated-anchor spread, reported separately from the frozen
acceptance threshold.

## Held-out scoring

`run_study.py score` rejects a changed fit digest, another inventory, or another
observation-file hash. It then opens confirmation cells and forms adjacent
matched pairs along each of these axes while holding every other request field
fixed:

- ranks;
- active channels;
- available SMs;
- useful bytes per channel;
- publication delay.

The measured delta is the median of five paired process-median differences.
Only resolved deltas are acceptance instances. Their error bound is the frozen
greater of 10 percent of the measured magnitude or the pair's repeat-spread
resolution threshold. Unresolved deltas are retained and counted but cannot
pass or fail the predictive requirement.

An empty set of resolved deltas is not a pass. The summary separately reports
whether any held-out intervention resolved, how many resolved deltas were
accepted or rejected, and whether every resolved delta passed.

## H100 v1 outcome

The frozen H100 campaign completed all 3,840 work items and collected
1,152,000 rows from 768 qualified cells. Validation found no completeness or
identity failure. It retained the raw values but voided eight complete
`same_device_interval` scopes whose missing source-local start produced an
epoch-sized value outside the matching CUDA event; the other timer scopes
remain valid evidence.

The identification-only fit used zero confirmation rows. It produced 1,792
anchors and 3,012 matched contrasts: 656 resolved as separate terms and 2,356
remain explicit joint intervals. After that fit was content-locked, the scorer
opened 1,308 adjacent confirmation interventions. Of 441 resolved deltas, 97
met the frozen error bound and 344 did not; the remaining 867 were unresolved.
The published outcome is therefore **FAIL**.

This outcome distinguishes measurement completion from model calibration. The
primitive anchors and separability ledger are valid within the qualified H100
scope, but the v1 source-constrained additive interpolation is falsified as a
general transfer rule. It must not be installed in the channel model, widened
to pass after inspection, or refit against these opened confirmation cells. A
successor model needs a new frozen design and independent confirmation data.
Exact grouped failures and all content hashes are in `RESULTS.md` and
`h100-results.json`.

For execution scalability, fit and score first reduce the complete ordinary
capture to one median per `(cell, timer, process)` and reuse that index for all
anchors and contrasts. This is an implementation optimization only: it applies
the same row predicates, float conversion, median, pairing, and IQR rule as the
direct scan, and a regression test requires both paths to return identical
statistics.

## Commands

```bash
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

The compact fit, score, validation report, and merged-artifact manifest belong
in the published result. Bulk rows remain outside Git and are bound by their
SHA-256.
