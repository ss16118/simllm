# TRAF-94 v4 selective uncertainty model

## Motivation

V2's five-axis interpolation rejected 45 of 225 resolved held-out effects. V3
removed interpolation entirely and still rejected 11 of 195 resolved effects
when exact control tuples were repeated. The remaining error is therefore not
fixed by another deterministic lookup. It is cross-run variability or an
unobserved execution-state effect, concentrated in Simple buffered rank scaling
and large-payload LL/LL128 scaling.

V4 changes the estimand from a point cell time to an interval for an adjacent
matched intervention delta. It also permits a prospective abstention when the
two opened runs show that a timer/protocol/placement/factor group cannot support
a useful interval. This is not a relabeling of v3: both earlier `FAIL` records
remain immutable training evidence, and all v4 intervals and abstentions are
frozen before a new timing sample is opened.

## Training transformation

For every one of the 816 matched intervention definitions, v2 and v3 each
provide five paired process deltas. V4 freezes:

1. the point center as the median of all ten paired deltas;
2. the resolution floor as the larger of the two runs' original
   `2 * (IQR_A + IQR_B)` values;
3. two directional normalized run residuals:

   ```text
   abs(v2_median - v3_median)
       / max(abs(source_run_median), source_run_resolution, 1 ns)
   ```

4. a finite-sample upper ranked 90% residual within each
   timer/protocol/placement/factor group;
5. the interval half-width as the larger of the resolution floor and that
   group residual multiplied by `max(abs(center), resolution_floor, 1 ns)`.

Directional residuals sharing an intervention are dependent. The ranked
quantile is therefore an empirical calibration rule, not a claim of formal
exchangeable split-conformal coverage.

## Selective prediction

A group predicts only when its normalized ranked residual is at most 1.0. A
larger value would require a half-width greater than the effect/resolution
scale, so the model abstains instead of publishing a vacuous interval. The
freeze has 57 groups: 49 predict and eight abstain. Historical two-way checks
cover 190/195 v3-resolved effects from v2 (97.4%) and 217/225 v2-resolved
effects from v3 (96.4%); these checks motivated the prospective contract but
are not v4 confirmation evidence.

The independent v4 result passes only if:

- at least one intervention resolves;
- at least 90% of resolved interventions belong to predicting groups;
- at least 90% of predicted resolved deltas fall inside their frozen interval;
- every timer has at least 85% coverage among its predicted resolved deltas;
- no used interval has normalized half-width above 1.0.

Legacy 10%-or-repeat-spread point accuracy remains reported as a secondary
diagnostic and cannot determine the v4 outcome.

## Fixed device cohort

Earlier raw rows bind the executable identity but not physical GPU UUIDs, and
v3 had to switch GPU sets after an external-job collision. V4 removes that
ambiguity. [device-cohort.json](device-cohort.json) freezes the ordered UUIDs
for physical GPUs 0, 1, and 2. The campaign supervisor verifies that exact
rank order, writes a self-digested campaign provenance record, and refuses a
resume with another cohort. If any selected GPU is occupied, execution waits;
switching cohorts after partial progress is forbidden.

## Scope

A v4 pass would calibrate interval coverage only for the exact 112-cell table,
pinned source/probe, fixed GPU cohort, and this node. It would not validate an
unseen control, another NCCL collective, another GPU generation, or DeepEP.
An abstention is part of the model output and must not be presented as a
successful prediction.
