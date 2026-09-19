# TRAF-94 v3 same-node calibration result

## Outcome: FAIL

The exact-control empirical table does not satisfy the unchanged TRAF-94
confirmation rule. Of 816 fresh matched interventions, 195 resolve above
repeat spread. The table accepts 184 and rejects 11; the preregistered contract
requires every resolved intervention to pass.

This result is materially better than v2's sparse interaction interpolation
(11 rather than 45 rejected resolved deltas), but it is still not a pass. No
threshold was widened and no value from the v3 capture was used to change the
frozen table.

## Evidence integrity

The campaign completed all 560 qualified work items and collected 168,000 rows
from 112 cells. Validation reports zero fatal failures and zero void timer
scopes. The observation SHA-256 is
`e8da409b02c8b919179b48829b63bda6e1585fa576f59a2b941d10e661947b88`.

One attempt at work index 376 was rejected by the immediate post-run context
guard when an external training job entered a selected GPU. That attempt is
preserved in quarantine and excluded from collection. The frozen index reran
cleanly after another stable-idle gate. Active evidence has exactly 560
completion markers, zero failure markers, and zero contamination markers.

## Residual failure shape

| Factor | Rejected resolved deltas |
|---|---:|
| ranks | 5 |
| useful bytes per channel | 5 |
| active channels | 1 |

Six failures are Simple buffered, three are LL, and two are LL128. The five
rank misses are dominated by Simple buffered rank-2 to rank-3 transitions; the
largest differs from the frozen prior-run delta by about 0.46 ms. Four of the
payload misses are LL/LL128 large-payload deltas, with error-to-bound ratios
from approximately 1.05 to 1.34. This is cross-run drift or an unmodelled
execution-state effect, not an interpolation error: v3 used exact tuple lookup.

## Supported conclusion

The node-specific table is useful as a measured baseline, and 184 resolved
effects reproduce under the strict rule. It is not a fully calibrated point
model under that rule. This result does not justify unseen-control
interpolation, another GPU subset or generation, arbitrary NCCL collectives,
or DeepEP. A follow-up needs an explicitly stochastic or execution-state-aware
model and new independent evidence; repeatedly refitting point values to opened
runs would not resolve the identified uncertainty.

The immutable compact record is [h100-v3-result.json](h100-v3-result.json).
Raw work directories, observations, validation, and complete score remain
outside Git and are bound into that record by SHA-256.
