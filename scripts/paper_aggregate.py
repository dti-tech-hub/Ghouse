"""Aggregate the raw sweep + experiment runs into the tables used by the paper.

Reads:
  docs/paper_data/fault_sweep.json   (from scripts/paper_fault_sweep.py)
  storage/agentic_rag.db             (experiments, traces, recovery attempts)

Writes:
  docs/paper_data/results.json       every number quoted in the document
  docs/paper_data/results.md         the same numbers as markdown tables

No value here is hand-written; all of it is derived from executed runs.
"""

from __future__ import annotations

import json
import statistics
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "docs" / "paper_data"
DB = ROOT / "storage" / "agentic_rag.db"


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 3),
        "p50": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "max": round(ordered[-1], 3),
        "stdev": round(statistics.stdev(ordered), 3) if len(ordered) > 1 else 0.0,
    }


def main() -> int:
    sweep = json.loads((DATA / "fault_sweep.json").read_text(encoding="utf-8"))
    runs = sweep["runs"]
    results: dict = {
        "label": sweep["label"],
        "generated_from": {
            "fault_sweep_runs": len(runs),
            "questions": len(sweep["questions"]),
            "faults": len({r["fault_type"] for r in runs}),
        },
    }

    # ---------------------------------------------------------------- overall
    detected = sum(1 for r in runs if r["detected"])
    top1 = sum(1 for r in runs if r["match"])
    in_list = sum(1 for r in runs if r["match_rank"] is not None)
    recovered = sum(1 for r in runs if r["recovered"])
    abstained = sum(1 for r in runs if r["abstained"])
    verified = sum(1 for r in runs if r["verified_recovery"])
    attempted = sum(1 for r in runs if r["n_recovery_attempts"] > 0)

    results["overall"] = {
        "runs": len(runs),
        "detected": detected,
        "detection_rate_pct": pct(detected, len(runs)),
        "top1_accuracy_pct": pct(top1, len(runs)),
        "top1_correct": top1,
        "topk_accuracy_pct": pct(in_list, len(runs)),
        "topk_correct": in_list,
        "runs_with_recovery_attempts": attempted,
        "recovered": recovered,
        "recovery_success_pct_of_attempted": pct(recovered, attempted),
        "verified_recovery": verified,
        "abstained": abstained,
        "abstention_rate_pct": pct(abstained, len(runs)),
        "latency_ms": describe([r["total_latency_ms"] for r in runs]),
        "diagnosis_latency_ms": describe([r["diagnosis_latency_ms"] for r in runs]),
        "added_latency_ms": describe([r["added_latency_ms"] for r in runs if r["n_recovery_attempts"]]),
    }

    # ------------------------------------------------------------- per fault
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_fault[run["fault_type"]].append(run)

    per_fault = []
    for fault_type, group in by_fault.items():
        actions: Counter = Counter()
        for run in group:
            for attempt in run["recovery_actions"]:
                actions[attempt["action"]] += 1
        per_fault.append({
            "fault_type": fault_type,
            "target_stage": group[0]["target_stage"],
            "expected_subtype": group[0]["expected_subtype"],
            "expected_class": group[0]["expected_class"],
            "n": len(group),
            "detected": sum(1 for r in group if r["detected"]),
            "top1": sum(1 for r in group if r["match"]),
            "topk": sum(1 for r in group if r["match_rank"] is not None),
            "recovered": sum(1 for r in group if r["recovered"]),
            "abstained": sum(1 for r in group if r["abstained"]),
            "detection_rate_pct": pct(sum(1 for r in group if r["detected"]), len(group)),
            "top1_pct": pct(sum(1 for r in group if r["match"]), len(group)),
            "topk_pct": pct(sum(1 for r in group if r["match_rank"] is not None), len(group)),
            "recovery_success_pct": pct(sum(1 for r in group if r["recovered"]), len(group)),
            "recovered": sum(1 for r in group if r["recovered"]),
            "n": len(group),
            "abstention_pct": pct(sum(1 for r in group if r["abstained"]), len(group)),
            "median_latency_ms": round(statistics.median(r["total_latency_ms"] for r in group), 2),
            "mean_confidence": round(statistics.fmean(
                [r["confidence"] for r in group if r["confidence"] is not None] or [0.0]), 4),
            "actions": dict(actions),
        })
    per_fault.sort(key=lambda row: row["fault_type"])
    results["per_fault"] = per_fault

    # ------------------------------------------------------------ per class
    from agentic_rag_recovery.taxonomy import class_for_subtype

    def expected_class(run: dict) -> str:
        """Ground-truth class, derived from the declared expected subtype."""
        if run.get("expected_class"):
            return run["expected_class"]
        try:
            return class_for_subtype(run["expected_subtype"]).value
        except Exception:
            return "UNKNOWN"

    by_class: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_class[expected_class(run)].append(run)
    results["per_class"] = [
        {
            "failure_class": name,
            "n": len(group),
            "detection_rate_pct": pct(sum(1 for r in group if r["detected"]), len(group)),
            "top1_pct": pct(sum(1 for r in group if r["match"]), len(group)),
            "topk_pct": pct(sum(1 for r in group if r["match_rank"] is not None), len(group)),
            "recovery_success_pct": pct(sum(1 for r in group if r["recovered"]), len(group)),
            "recovered": sum(1 for r in group if r["recovered"]),
            "n": len(group),
            "abstention_pct": pct(sum(1 for r in group if r["abstained"]), len(group)),
            "median_latency_ms": round(statistics.median(r["total_latency_ms"] for r in group), 2),
        }
        for name, group in sorted(by_class.items())
    ]

    # ------------------------------------------------- per stage / confusion
    confusion: dict[str, Counter] = defaultdict(Counter)
    for run in runs:
        if run["predicted_subtype"]:
            confusion[run["expected_subtype"]][run["predicted_subtype"]] += 1
    results["confusion"] = {
        expected: dict(counts.most_common()) for expected, counts in sorted(confusion.items())
    }

    misroutes = Counter()
    for run in runs:
        if run["predicted_subtype"] and not run["match"]:
            misroutes[(expected_class(run), run["predicted_class"] or run["predicted_subtype"])] += 1
    results["top_misroutes"] = [
        {"from_class": a, "to_subtype": b, "count": c} for (a, b), c in misroutes.most_common(12)
    ]

    # -------------------------------------------------------------- outcomes
    results["outcomes"] = dict(Counter(r["status"] for r in runs))
    results["diagnosis_tiers"] = dict(Counter(str(r["tier"]) for r in runs if r["tier"]))
    results["rules_triggered"] = dict(Counter(r["rule_id"] for r in runs if r["rule_id"]))

    # ------------------------------------------------------------- recovery
    attempt_rows = [a for r in runs for a in r["recovery_actions"]]
    action_stats: dict[str, Counter] = defaultdict(Counter)
    for attempt in attempt_rows:
        action_stats[attempt["action"]]["attempts"] += 1
        if "success" in attempt["status"].lower():
            action_stats[attempt["action"]]["success"] += 1
        if attempt["verified"]:
            action_stats[attempt["action"]]["verified"] += 1
    results["recovery_actions"] = [
        {
            "action": action,
            "attempts": stats["attempts"],
            "success": stats["success"],
            "verified": stats["verified"],
            "success_rate_pct": pct(stats["success"], stats["attempts"]),
        }
        for action, stats in sorted(action_stats.items(), key=lambda kv: -kv[1]["attempts"])
    ]
    results["recovery_reuse"] = {
        "reused_stages_mean": round(statistics.fmean(
            [len(a["reused_stages"]) for a in attempt_rows] or [0.0]), 3),
        "reused_stages_max": max((len(a["reused_stages"]) for a in attempt_rows), default=0),
        "added_latency_ms": describe([a["added_latency_ms"] for a in attempt_rows]),
    }

    # ------------------------------------------------------- signal ablation
    # Which signals fire for each failure class -> detection evidence.
    evidence: dict[str, Counter] = defaultdict(Counter)
    for run in runs:
        for item in (run["evidence"] or [])[:1]:
            evidence[expected_class(run)][item.split(";")[0][:70]] += 1
    results["leading_evidence"] = {
        cls: [{"signal": sig, "count": c} for sig, c in counts.most_common(4)]
        for cls, counts in evidence.items()
    }

    # --------------------------------------------------- experiments in the DB
    if DB.exists():
        conn = sqlite3.connect(str(DB))
        conn.row_factory = sqlite3.Row
        try:
            exp_rows = conn.execute(
                "SELECT * FROM experiments ORDER BY rowid"
            ).fetchall()
            experiments = []
            for row in exp_rows:
                record = dict(row)
                payload = {}
                for key, value in record.items():
                    if isinstance(value, str) and value.strip().startswith(("{", "[")):
                        try:
                            payload[key] = json.loads(value)
                        except Exception:
                            payload[key] = value
                    else:
                        payload[key] = value
                experiments.append(payload)
            if experiments:
                metrics_keys: set[str] = set()
                for record in experiments:
                    for source in (record.get("summary"), record.get("metrics"), record.get("results")):
                        if isinstance(source, dict):
                            metrics_keys.update(source.keys())
                results["experiments"] = {
                    "count": len(experiments),
                    "configs": [
                        {
                            "experiment_id": record.get("experiment_id"),
                            "config_name": record.get("config_name") or record.get("name"),
                            "seed": record.get("seed"),
                            "runs": record.get("runs") or record.get("n_runs"),
                            "status": record.get("status"),
                        }
                        for record in experiments
                    ],
                    "records": experiments,
                }
            table_names = [
                r["name"] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            results["database"] = {
                "tables": table_names,
                "row_counts": {
                    name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                    for name in table_names
                },
            }
        finally:
            conn.close()

    # ------------------------------------------------------------ stage cost
    stage_costs: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for step_name, latency in (run.get("steps") or []):
            stage_costs[step_name].append(float(latency))
    if stage_costs:
        results["stage_latency_ms"] = {k: describe(v) for k, v in sorted(stage_costs.items())}

    (DATA / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------- markdown
    lines: list[str] = []
    add = lines.append
    add("# Measured results (auto-generated)")
    add("")
    add(f"Label: **{results['label']}**. "
        f"Source: {results['generated_from']['fault_sweep_runs']} executed runs "
        f"({results['generated_from']['faults']} faults x "
        f"{results['generated_from']['questions']} questions).")
    add("")
    add("## Overall")
    add("")
    add("| metric | value |")
    add("|---|---|")
    o = results["overall"]
    for key in ("runs", "detected", "detection_rate_pct", "top1_correct", "top1_accuracy_pct",
                "topk_correct", "topk_accuracy_pct", "runs_with_recovery_attempts",
                "recovered", "recovery_success_pct_of_attempted", "verified_recovery",
                "abstained", "abstention_rate_pct"):
        add(f"| {key} | {o[key]} |")
    for key in ("latency_ms", "diagnosis_latency_ms", "added_latency_ms"):
        d = o[key]
        add(f"| {key} (n={d.get('n', 0)}) | p50 {d.get('p50')} / mean {d.get('mean')} / "
            f"p95 {d.get('p95')} / max {d.get('max')} |")
    add("")
    add("## Per fault")
    add("")
    add("| fault | stage | expected subtype | det. % | top-1 % | top-k % | recovered % | abstained % | p50 ms |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in results["per_fault"]:
        add(f"| `{row['fault_type']}` | {row['target_stage']} | `{row['expected_subtype']}` | "
            f"{row['detection_rate_pct']} | {row['top1_pct']} | {row['topk_pct']} | "
            f"{row['recovery_success_pct']} | {row['abstention_pct']} | {row['median_latency_ms']} |")
    add("")
    add("## Per failure class")
    add("")
    add("| failure class | n | det. % | top-1 % | top-k % | recovered % | abstained % | p50 ms |")
    add("|---|---|---|---|---|---|---|---|")
    for row in results["per_class"]:
        add(f"| `{row['failure_class']}` | {row['n']} | {row['detection_rate_pct']} | "
            f"{row['top1_pct']} | {row['topk_pct']} | {row['recovery_success_pct']} | "
            f"{row['abstention_pct']} | {row['median_latency_ms']} |")
    add("")
    add("## Recovery actions")
    add("")
    add("| action | attempts | success | verified | success % |")
    add("|---|---|---|---|---|")
    for row in results["recovery_actions"]:
        add(f"| `{row['action']}` | {row['attempts']} | {row['success']} | {row['verified']} | "
            f"{row['success_rate_pct']} |")
    add("")
    add("## Outcomes")
    add("")
    add("| outcome | runs |")
    add("|---|---|")
    for key, value in sorted(results["outcomes"].items()):
        add(f"| {key} | {value} |")
    add("")
    (DATA / "results.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"runs analysed : {len(runs)}")
    print(f"detection     : {o['detection_rate_pct']}%")
    print(f"top-1 accuracy: {o['top1_accuracy_pct']}%")
    print(f"top-k accuracy: {o['topk_accuracy_pct']}%")
    print(f"recovered     : {o['recovered']}/{o['runs_with_recovery_attempts']} attempted "
          f"({o['recovery_success_pct_of_attempted']}%)")
    print(f"abstention    : {o['abstention_rate_pct']}%")
    print(f"latency p50   : {o['latency_ms']['p50']} ms, p95 {o['latency_ms']['p95']} ms")
    print(f"wrote         : {(DATA / 'results.json').relative_to(ROOT)}")
    print(f"wrote         : {(DATA / 'results.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
