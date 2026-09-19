# TRAF-94 v2 H100 result

## Outcome: FAIL

The prospectively frozen five-axis interaction tensor does not satisfy the
TRAF-94 confirmation rule. Of 816 held-out adjacent interventions, 225 resolve
above repeat spread. The frozen model accepts 180 and rejects 45; the contract
requires every resolved intervention to pass.

This is a model failure, not an execution failure. All 560 qualified work items
completed, yielding 168,000 rows across 112 cells. Validation found zero fatal
failures and zero void timer scopes. The observation SHA-256 is
`7c0299623ef0531f132fb5203ca97057dc3f6798fa094688aaab9a76f94e723b`.

## Capability boundary

The pre-timing capability sweep produced 28 qualified and four rejected keys.
All four rejections are rank-3 Simple buffered-read controls and fail the
destination check with `got 4, expected 6`. They exclude exactly the 16 cells
that the v1 training plane could not predict. The remaining 112-cell inventory
is rectangular within every protocol/placement stratum and every cell has a
prediction frozen before ordinary timing.

## Failure shape

The 45 rejected resolved deltas group primarily by:

| Factor | Rejected resolved deltas |
|---|---:|
| useful bytes per channel | 16 |
| ranks | 15 |
| active channels | 10 |
| publication delay | 3 |
| available SMs | 1 |

Thirty failures are in Simple and fifteen in LL. The largest misses include
sign reversals and approximately 0.45--0.52 ms cliffs in Simple buffered
timings. Those effects show that a sparse multilinear surface can retain all
formal axis interactions yet still smooth across discrete execution regimes.

The immutable compact record is [h100-v2-result.json](h100-v2-result.json).
Raw work directories, capability output, observations, validation, and the
complete intervention score remain outside Git and are bound into that record
by SHA-256.

## Consequence

V2 remains a valid negative result and must not be refitted. The follow-up v3
study narrows the claim to exact-control, same-node empirical calibration and
uses an entirely new set of process launches. A v3 success would establish
repeatability of that bounded table, not interpolation to unseen controls or
coverage of arbitrary NCCL/DeepEP communication.
