# TRAF-94 v3 same-node replication calibration

## Why v3 narrows the claim

The v2 interaction surface was frozen before measurement and validly scored,
but failed 45 of 225 resolved intervention deltas. The failures are not missing
rows or bad timer scopes: the 168,000-row capture has zero validation failures
and zero void scopes. Most misses occur at Simple buffered rank, payload, and
channel transitions where the hardware exhibits regime changes that sparse
multilinear interpolation does not reproduce.

V3 does not repair that result after the fact. V2 remains an immutable
`FAIL`. Instead, v3 asks a narrower and operationally useful question: if each
qualified v2 control tuple is calibrated directly on this H100/NVL8 node, do
new OS processes reproduce the calibrated intervention deltas?

## Candidate model

The model is an exact empirical lookup table. For every qualified v2 cell and
timer, its prediction is the median of the five v2 process medians. A complete
requested-control object is the lookup key. There is deliberately:

- no interpolation;
- no extrapolation;
- no nearest-neighbour fallback;
- no use of any v3 ordinary timing during fitting.

The planned v3 matrix repeats the v2 controls but uses a new experiment role,
new content-derived cell ids, a new ordering seed, new process launches, and a
separate work root. The 16 rank-3 Simple buffered-read requests that failed v2
capability are not table entries. Their absence must again be represented by
capability qualification before timing rather than silently deleted.

## Evidence boundary

V2 is opened training evidence. Its qualified inventory, observations,
validation report, failed score, and model identity are pinned in the v3
manifest by digest or file SHA-256. The v3 plan and all 112 predictable timer
tables are committed and pushed before any v3 ordinary work item runs.

Capability results may be replayed because v3 uses the identical pinned probe
identity and identical requested controls; capability checks do not contain
ordinary timing values. Ordinary evidence is never replayed: all 560 v3 work
items are fresh executions guarded by the unchanged stable-idle and immediate
post-run selected-GPU checks.

## Scoring and interpretation

V3 retains the original matched-intervention rules. A delta resolves only when

```text
abs(delta) > 2 * (IQR_A + IQR_B)
```

and every resolved delta must satisfy

```text
abs(predicted_delta - measured_delta)
    <= max(10% * abs(measured_delta), 2 * (IQR_A + IQR_B)).
```

A `PASS` establishes repeatability only for the exact 112-cell table on the
same node, probe identity, protocol/placement implementations, and control
levels. It does not validate interpolation to unseen controls, a different
GPU generation, arbitrary NCCL collectives, or DeepEP. A `FAIL` means even
this bounded calibration table is not stable enough under the frozen rule and
must remain open.
