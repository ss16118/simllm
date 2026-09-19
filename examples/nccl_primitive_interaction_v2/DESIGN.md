# TRAF-94 v2 non-separable interaction model

## Why v2 exists

The completed v1 H100 campaign is valid evidence, but its additive transfer
candidate failed 344 of 441 resolved held-out intervention deltas. The largest
misses occur in channel scaling, followed by rank and payload scaling. V1
assumed those effects could be composed from independent rank-2 data and
sharing curves. The opened holdout shows that assumption is false.

V2 does not alter the v1 result, widen its acceptance bar, or reuse its opened
cells as confirmation. All 168 qualified v1 confirmation cells become declared
training evidence. A new matrix is frozen and measured only after the v2 model
and every planned prediction are content-locked.

## Candidate model

V2 builds one independent response tensor for each timer, protocol, and Simple
placement. Every v1 confirmation cell first yields one median per OS process;
the tensor anchor is the median of those five process medians. Its coordinates
are:

- ranks;
- active channels;
- available SMs (`full` is the measured H100 value 132);
- useful bytes per channel;
- controlled publication-delay cycles.

Prediction uses piecewise multilinear interpolation over the complete measured
tensor. Payload has three training levels and therefore selects one of two
local intervals; all other non-degenerate axes have two training levels. The
model includes the full product of axis weights, so rank, channel, SM, payload,
and delay interactions are retained instead of forced into additive terms.
Each protocol and placement has its own tensor. Simple buffered read has only
the qualified rank-2 plane and therefore refuses any other rank.

Extrapolation and clamping are forbidden. A planned holdout cell outside its
training hyperrectangle is a design error, not a prediction. The frozen model
artifact contains the exact prediction for every planned cell, so scoring
cannot change interpolation behavior after measurements are opened.

## New holdout

The matrix deliberately uses levels absent from the v1 confirmation grid while
remaining inside its numeric bounds:

| Axis | V1 training levels | V2 holdout levels |
|---|---|---|
| ranks | 2, 4 | 2, 3 |
| active channels | 2, 12 | 4, 8 |
| available SMs | 16, 132 | 32, 64 |
| useful bytes/channel | 124, 1,924, 32,772 | 512, 8,192 |
| delay cycles | 512, 2,048 | 768, 1,536 |

The full plan has 128 cells and 640 five-process work items. Capability pilots
may reject unsupported source specializations, but every surviving
protocol/placement stratum must remain a complete rectangular grid. Based on
the retained pinned-source finding, multi-rank Simple buffered read may be
rejected; that rejection is evidence, not a hard-coded deletion. No ordinary
timing starts until the qualified inventory is frozen and rectangular.

## Scoring

Scoring retains the v1 rule. Within each timer/protocol/placement stratum,
adjacent matched cells are compared along one axis while every other requested
control stays fixed. The measured value is the median of the five paired
process-median deltas. A delta resolves only when

```text
abs(delta) > 2 * (IQR_A + IQR_B)
```

and its prediction passes only when the absolute error is no greater than

```text
max(10% * abs(measured_delta), 2 * (IQR_A + IQR_B)).
```

An empty resolved set is not a pass, and **every** resolved intervention must
pass. Raw rows, invalid timer scopes, capability rejections, and unresolved
deltas remain visible. Once the holdout is opened, this model may not be
refitted; a failure requires another version and independent evidence.

## Execution boundary

The existing pinned NCCL 2.31.2 probe, build identity, serial supervisor,
stable-idle gate, immediate post-run contention guard, and hash-verifying
collector are reused unchanged. Capability and ordinary timing remain separate.
No NVML observer runs inside a timed work item, and no two work items overlap on
the shared NVSwitch node.

## Frozen identities

- manifest: `2bf7455bf91bf6081b11ed82652e5258826cf59486c30e618cbcda2a7e662942`;
- planned inventory: `0495e9583c5692ab229f3aa3bef93951a96a1a06ee7ddf40cc2ff6ce1a983cef`;
- locked model: `fba72bb7fe28e112cf4f2b73ca7381d5ea9ec9c55e29a1f4d1a4f6cfb1e15c0e`;
- v1 training observations: `594769e4d2871f119a05f47dd5f8a60cc03e8bb3475766da2d8e1162f86ecb5d`.

The locked model contains 112 predictions and names 16 intentionally
unpredicted planned cells. These identities are committed before capability or
ordinary v2 execution.
