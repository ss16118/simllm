"""Publish compact, hash-bound TRAF-94 results from complete external artifacts.

The full fit intentionally retains every process median and matched contrast and
can be several megabytes.  Raw rows are much larger.  This publisher keeps
those artifacts external while emitting the load-bearing component surfaces,
grouped identifiability/confirmation outcomes, and hashes needed for review.
It never recomputes or changes a fit or held-out score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.nccl_primitive_identification_v1.analysis import (
    FIT_SCHEMA,
    SCORE_SCHEMA,
)
from examples.nccl_primitive_identification_v1.matrix import content_digest

RESULT_SCHEMA = "simllm-nccl-primitive-hardware-result-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verified_digest(value: Mapping[str, Any], *, field: str, schema: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"unexpected artifact schema: {value.get('schema')!r}")
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if recorded != content_digest(unsigned):
        raise ValueError(f"{field} does not match artifact content")
    return str(recorded)


def _component_anchors(fit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain only surfaces used by the frozen confirmation equation."""

    result = []
    for row in fit["anchors"]:
        request = row["requested"]
        keep = (
            row["stage"] == "data_work"
            and row["family"] == "sum"
            and request.get("working_set") == "reused"
        ) or (row["stage"] == "sharing" and row["family"] == "sum")
        if not keep:
            continue
        summary = row["summary"]
        result.append(
            {
                "cell_id": row["cell_id"],
                "stage": row["stage"],
                "timer": row["timer"],
                "protocol": row["protocol"],
                "simple_placement": row["simple_placement"],
                "requested": request,
                "median_ns": summary["median_ns"],
                "q1_ns": summary["q1_ns"],
                "q3_ns": summary["q3_ns"],
                "iqr_ns": summary["iqr_ns"],
            }
        )
    return result


def _contrast_groups(contrasts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in contrasts:
        key = (
            str(row["stage"]),
            str(row["timer"]),
            str(row["protocol"]),
            str(row["simple_placement"]),
            str(row["factor"]),
        )
        grouped[key].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        deltas = [float(row["paired_median_delta"]) for row in rows]
        resolved = sum(bool(row["resolved"]) for row in rows)
        result.append(
            {
                "stage": key[0],
                "timer": key[1],
                "protocol": key[2],
                "simple_placement": key[3],
                "factor": key[4],
                "contrast_count": len(rows),
                "resolved_separate_terms": resolved,
                "joint_intervals": len(rows) - resolved,
                "minimum_paired_median_delta_ns": min(deltas),
                "maximum_paired_median_delta_ns": max(deltas),
            }
        )
    return result


def _confirmation_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["timer"]),
            str(row["protocol"]),
            str(row["simple_placement"]),
            str(row["factor"]),
        )
        grouped[key].append(row)
    result = []
    for key, members in sorted(grouped.items()):
        resolved = [row for row in members if row["acceptance_applicable"]]
        rejected = [row for row in resolved if not row["accepted_resolved_delta"]]
        normalized_errors = [
            float(row["absolute_error"]) / float(row["acceptance_bound"])
            if float(row["acceptance_bound"]) > 0
            else (0.0 if float(row["absolute_error"]) == 0 else float("inf"))
            for row in resolved
        ]
        result.append(
            {
                "timer": key[0],
                "protocol": key[1],
                "simple_placement": key[2],
                "factor": key[3],
                "intervention_count": len(members),
                "resolved_interventions": len(resolved),
                "unresolved_interventions": len(members) - len(resolved),
                "accepted_resolved_interventions": len(resolved) - len(rejected),
                "rejected_resolved_interventions": len(rejected),
                "worst_error_over_bound": max(normalized_errors, default=None),
            }
        )
    return result


def build_result(
    *,
    validation: Mapping[str, Any],
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
    merged_artifacts: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Cross-bind complete artifacts and return the compact publication."""

    fit_digest = _verified_digest(fit, field="fit_digest", schema=FIT_SCHEMA)
    score_digest = _verified_digest(score, field="score_digest", schema=SCORE_SCHEMA)
    if validation.get("schema") != "simllm-nccl-primitive-validation-v1":
        raise ValueError("validation report has an unsupported schema")
    if merged_artifacts.get("schema") != "simllm-nccl-primitive-merged-artifacts-v1":
        raise ValueError("merged artifact manifest has an unsupported schema")
    identities = {
        "manifest_digest": validation["manifest_digest"],
        "inventory_digest": validation["inventory_digest"],
        "observations_sha256": validation["observations_sha256"],
    }
    for artifact in (fit, score, merged_artifacts):
        for field, expected in identities.items():
            if artifact.get(field) != expected:
                raise ValueError(f"artifact {field} does not match validation")
    if score.get("fit_digest") != fit_digest:
        raise ValueError("held-out score does not use the supplied fit")
    if validation.get("failures"):
        raise ValueError("cannot publish a campaign with validation failures")
    if validation.get("fatal_guards") not in {"valid", "valid_outside_void_scopes"}:
        raise ValueError("cannot publish a campaign with invalid fatal guards")
    work_count = len(merged_artifacts.get("work_manifests", ()))
    if work_count <= 0 or int(merged_artifacts.get("row_count", 0)) != int(
        validation["row_count"]
    ):
        raise ValueError("merged artifact counts do not match validation")

    summary = score["summary"]
    passed = bool(summary["has_resolved_interventions"]) and bool(
        summary["all_resolved_interventions_accepted"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        **identities,
        "fit_digest": fit_digest,
        "score_digest": score_digest,
        "input_file_sha256": dict(input_hashes),
        "campaign": {
            "work_count": work_count,
            "row_count": validation["row_count"],
            "cell_count": validation["cell_count"],
            "fatal_guards": validation["fatal_guards"],
            "void_scopes": validation["void_scopes"],
        },
        "identification": {
            "fit_scope": fit["fit_scope"],
            "confirmation_rows_used": fit["confirmation_rows_used"],
            "anchor_count": len(fit["anchors"]),
            "published_component_anchors": _component_anchors(fit),
            "publication_response": fit["publication_response"],
            "separability": fit["separability"],
            "contrast_groups": _contrast_groups(fit["paired_contrasts"]),
            "confirmation_model": fit["confirmation_model"],
        },
        "confirmation": {
            "summary": summary,
            "groups": _confirmation_groups(score["scored_interventions"]),
        },
        "outcome": "PASS" if passed else "FAIL",
        "claim_boundary": (
            "TRAF-94 identifies primitive component response on the qualified H100 "
            "scope. TRAF-43 retains full collective accuracy and uncertainty; "
            "TRAF-54 retains integration into finite channel/GPU resources."
        ),
    }
    result["result_digest"] = content_digest(result)
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a deterministic human review of the compact JSON authority."""

    campaign = result["campaign"]
    identification = result["identification"]
    confirmation = result["confirmation"]["summary"]
    lines = [
        "# TRAF-94 H100 primitive-identification results",
        "",
        f"Outcome: **{result['outcome']}**.",
        "",
        "## Evidence",
        "",
        f"- Work items: {campaign['work_count']}",
        f"- Raw rows: {campaign['row_count']}",
        f"- Qualified cells: {campaign['cell_count']}",
        f"- Fatal guards: `{campaign['fatal_guards']}`",
        f"- Voided cell/timer scopes: {len(campaign['void_scopes'])}",
        f"- Observation SHA-256: `{result['observations_sha256']}`",
        f"- Fit digest: `{result['fit_digest']}`",
        f"- Score digest: `{result['score_digest']}`",
        "",
        "## Identification",
        "",
        (
            f"The fit used `{identification['fit_scope']}` and "
            f"{identification['confirmation_rows_used']} confirmation rows. It contains "
            f"{identification['anchor_count']} process-median anchors. Of "
            f"{identification['separability']['contrast_count']} matched contrasts, "
            f"{identification['separability']['resolved_separate_terms']} resolve as "
            f"separate terms and {identification['separability']['joint_intervals']} "
            "remain joint intervals."
        ),
        "",
        "## Held-out confirmation",
        "",
        "| Timer | Protocol | Placement | Factor | Total | Resolved | Accepted | Rejected | Unresolved | Worst error / bound |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["confirmation"]["groups"]:
        worst = row["worst_error_over_bound"]
        lines.append(
            "| {timer} | {protocol} | {placement} | {factor} | {total} | {resolved} | "
            "{accepted} | {rejected} | {unresolved} | {worst} |".format(
                timer=row["timer"],
                protocol=row["protocol"],
                placement=row["simple_placement"],
                factor=row["factor"],
                total=row["intervention_count"],
                resolved=row["resolved_interventions"],
                accepted=row["accepted_resolved_interventions"],
                rejected=row["rejected_resolved_interventions"],
                unresolved=row["unresolved_interventions"],
                worst="—" if worst is None else f"{worst:.4f}",
            )
        )
    lines.extend(
        [
            "",
            (
                f"Resolved interventions: {confirmation['resolved_interventions']}; "
                f"accepted: {confirmation['accepted_resolved_interventions']}; rejected: "
                f"{confirmation['rejected_resolved_interventions']}."
            ),
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
            (
                "The adjacent JSON result is authoritative for exact component surfaces, "
                "grouped contrasts, void scopes, identities, and input hashes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--merged-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "validation": args.validation,
        "fit": args.fit,
        "score": args.score,
        "merged_artifacts": args.merged_artifacts,
    }
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    result = build_result(
        validation=values["validation"],
        fit=values["fit"],
        score=values["score"],
        merged_artifacts=values["merged_artifacts"],
        input_hashes={name: _sha256(path) for name, path in paths.items()},
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": result["outcome"],
                "result_digest": result["result_digest"],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
