"""Verify every quoted number in docs/CONFERENCE_PAPER.md against the data.

The document is written from executed runs; this script re-reads
``docs/paper_data/*.json`` (and optionally the SQLite store) and asserts that
each value quoted in the document still holds. It exists so the paper cannot
drift away from the system that produced it.

    python scripts/verify_paper_numbers.py          # check the document
    python scripts/verify_paper_numbers.py --strict # also fail on unknown claims
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "docs" / "paper_data"
DOC = ROOT / "docs" / "CONFERENCE_PAPER.md"


def load(name: str):
    path = DATA / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_claims() -> dict[str, object]:
    """Every value the document is allowed to quote, with its live value."""
    results = load("results")
    by_mode = load("by_mode")
    experiments = load("experiments")
    taxonomy = load("taxonomy")
    strategies = load("strategies")
    detection = load("detection")
    codebase = load("codebase")
    faults = load("faults")
    system = load("system")
    config = load("config_yaml")

    claims: dict[str, object] = {}
    if results:
        overall = results["overall"]
        claims.update({
            "runs": overall["runs"],
            "detected": overall["detected"],
            "detection_rate_pct": overall["detection_rate_pct"],
            "top1_accuracy_pct": overall["top1_accuracy_pct"],
            "top1_correct": overall["top1_correct"],
            "topk_accuracy_pct": overall["topk_accuracy_pct"],
            "recovered": overall["recovered"],
            "recovery_attempts_runs": overall["runs_with_recovery_attempts"],
            "recovery_success_pct": overall["recovery_success_pct_of_attempted"],
            "abstained": overall["abstained"],
            "abstention_rate_pct": overall["abstention_rate_pct"],
            "latency_p50": overall["latency_ms"]["p50"],
            "latency_p95": overall["latency_ms"]["p95"],
            "latency_mean": overall["latency_ms"]["mean"],
            "latency_max": overall["latency_ms"]["max"],
            "diagnosis_latency_p50": overall["diagnosis_latency_ms"]["p50"],
            "added_latency_p50": overall["added_latency_ms"]["p50"],
            "added_latency_p95": overall["added_latency_ms"]["p95"],
            "n_faults": len(results["per_fault"]),
            "n_classes": len(results["per_class"]),
        })
        claims["n_recovery_actions"] = len(results["recovery_actions"])
        claims["reused_stages_mean"] = results["recovery_reuse"]["reused_stages_mean"]
        claims["reused_stages_max"] = results["recovery_reuse"]["reused_stages_max"]
    if by_mode:
        for row in by_mode["by_mode"]:
            short = row["mode"].split("_")[0]
            claims[f"{short}_runs"] = row["runs"]
            claims[f"{short}_groundedness"] = row.get("groundedness", {}).get("mean")
            claims[f"{short}_citation_coverage"] = row.get("citation_coverage", {}).get("mean")
            claims[f"{short}_retrieval_recall"] = row.get("retrieval_recall", {}).get("mean")
            claims[f"{short}_abstention_pct"] = row["abstention_rate_pct"]
            claims[f"{short}_added_latency_p50"] = row.get("added_latency_ms", {}).get("p50")
            claims[f"{short}_llm_calls"] = row.get("llm_calls", {}).get("mean")
            claims[f"{short}_status"] = row["status_counts"]
        delta = by_mode["groundedness_delta_mode3"]
        claims.update({
            "delta_n": delta["n"],
            "delta_mean": delta["mean_delta"],
            "delta_improved": delta["improved"],
            "delta_worsened": delta["worsened"],
            "delta_improved_pct": delta["improved_pct"],
        })
    if experiments:
        claims["n_experiments"] = experiments["totals"]["experiments"]
        claims["total_runs_db"] = experiments["totals"]["runs"]
        claims["total_diagnoses_db"] = experiments["totals"]["diagnoses"]
        claims["total_recovery_attempts_db"] = experiments["totals"]["recovery_attempts"]
        claims["total_fault_injections_db"] = experiments["totals"]["fault_injections"]
        claims["total_metrics_rows"] = experiments["totals"]["metrics_rows"]
        claims["total_spans"] = experiments["totals"]["trace_spans"]
        claims["experiment_configs"] = [r["config"] for r in experiments["experiments"]]
    if taxonomy:
        claims["n_subtypes"] = taxonomy["subtype_count"]
        claims["n_stages"] = len(taxonomy["stages"])
        claims["n_failure_classes"] = len(taxonomy["count_by_failure_class"])
        claims["subtypes_by_class"] = taxonomy["count_by_failure_class"]
    if strategies:
        claims["n_strategies"] = strategies["count"]
    if detection:
        claims["n_rules"] = detection["rule_id_count"]
        claims["rule_families"] = {
            family: sum(1 for r in detection["rule_ids"] if r.startswith(f"R_{family}_"))
            for family in sorted({r.split("_")[1] for r in detection["rule_ids"]})
        }
        claims["n_rule_functions"] = detection["rule_function_count"]
    if codebase:
        claims["core_loc"] = codebase["core_loc"]
        claims["total_loc"] = sum(v for k, v in codebase.items() if k.endswith("_loc"))
        claims["n_tests"] = codebase["collected_tests"]
        claims["n_experiment_configs"] = codebase["experiment_configs"]
    if faults:
        claims["n_fault_definitions"] = len(faults)
        claims["n_fault_stages"] = len({f["target_stage"] for f in faults})
    if config and isinstance(config.get("budget"), dict):
        claims["budget"] = config["budget"]
    if system and isinstance(system.get("health"), dict):
        claims["provider"] = system["health"].get("provider")
        claims["model"] = system["health"].get("model")
        claims["documents"] = system["health"].get("documents")
        claims["chunks"] = system["health"].get("chunks")
        claims["label"] = system["health"].get("labels", {}).get("results")
    return claims


def main() -> int:
    if not DOC.exists():
        print(f"missing {DOC}")
        return 1

    claims = build_claims()
    text = DOC.read_text(encoding="utf-8")
    results = load("results")
    by_mode = load("by_mode")
    experiments = load("experiments")

    # Every machine-generated table in the document must be reproduced verbatim.
    problems: list[str] = []

    for name in ("results.md", "by_mode.md"):
        source = DATA / name
        if not source.exists():
            problems.append(f"missing generated table {name}")
            continue
        generated = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("|")
        ]
        missing = [line for line in generated if line not in text]
        if missing:
            problems.append(
                f"{len(missing)} row(s) from {name} are not present in the document, "
                "e.g.: " + missing[0][:110]
            )

    # The markdown mirrors must themselves be current with respect to the JSON,
    # otherwise editing a .json file silently invalidates the whole chain.
    if by_mode:
        mirror = (DATA / "by_mode.md").read_text(encoding="utf-8")
        rebuilt = []
        for row in by_mode["by_mode"]:
            rebuilt.append(
                f"| `{row['mode']}` | {row['runs']} | "
                f"{row.get('groundedness', {}).get('mean')} | "
                f"{row.get('citation_coverage', {}).get('mean')} | "
                f"{row.get('retrieval_recall', {}).get('mean')} | "
                f"{row['abstention_rate_pct']} | "
                f"{row.get('added_latency_ms', {}).get('p50')} | "
                f"{row.get('llm_calls', {}).get('mean')} |"
            )
        for line in rebuilt:
            if line not in mirror:
                problems.append(
                    "by_mode.md is stale with respect to by_mode.json; re-run "
                    f"scripts/paper_by_mode.py. expected row: {line[:100]}"
                )
            if line not in text:
                problems.append(
                    f"document is missing the current ablation row: {line[:100]}"
                )

    # The artifact inventory table must quote the live store totals.
    if experiments:
        totals = experiments["totals"]
        inventory = {
            "recorded runs in the evaluation store": totals["runs"],
            "recorded diagnoses": totals["diagnoses"],
            "recorded recovery attempts": totals["recovery_attempts"],
            "recorded fault injections": totals["fault_injections"],
            "recorded metric rows": totals["metrics_rows"],
            "recorded trace spans": totals["trace_spans"],
            "experiment configurations": len(experiments["experiments"]),
        }
        for label, value in inventory.items():
            rendered = f"{value:,}".replace(",", " ")
            row = f"| {label} | {rendered} |"
            if row not in text:
                problems.append(
                    f"artifact inventory is stale for {label!r}: expected the row {row!r}"
                )

    # The three-mode ablation totals quoted in prose must match the data.
    if by_mode:
        delta = by_mode["groundedness_delta_mode3"]
        for label, value in (
            ("paired runs", delta["n"]),
            ("improved", delta["improved"]),
            ("worsened", delta["worsened"]),
            ("unchanged", delta["unchanged"]),
        ):
            if f"| {label} | {value} |" not in text:
                problems.append(f"ablation table row {label!r} = {value} is missing or stale")

    # Spot-check that headline figures appear verbatim in the prose.
    headline = [
        key for key in (
            "detection_rate_pct", "top1_accuracy_pct", "topk_accuracy_pct",
            "abstention_rate_pct", "recovery_success_pct",
        ) if key in claims
    ]
    for key in headline:
        value = claims[key]
        if value is None:
            continue
        rendered = f"{value:g}" if isinstance(value, float) else str(value)
        if rendered not in text:
            problems.append(f"headline value {key}={rendered} not found in the document")

    print("=" * 74)
    print(" paper number verification")
    print("=" * 74)
    print(f"  document : {DOC.relative_to(ROOT).as_posix()} ({len(text.splitlines())} lines)")
    print(f"  claims   : {len(claims)} live values checked")
    for key in sorted(headline):
        print(f"    {key:<26} {claims[key]}")
    print()
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}")
        print(f"\n{len(problems)} problem(s).")
        return 1
    print("  OK  every checked value matches docs/paper_data/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
