"""Per-mode ablation table over every recorded run.

This is the study's central comparison: the same queries and the same injected
faults executed under the three operating modes

  mode1  plain RAG                  - no agent, no detection, no recovery
  mode2  agentic RAG, no recovery   - agentic retrieval, detection only
  mode3  agentic RAG + recovery     - targeted, verified, bounded recovery

All three modes run on every query of every experiment, so the comparison is
paired rather than between-subjects.

Writes docs/paper_data/by_mode.json
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "docs" / "paper_data"
DB = ROOT / "storage" / "agentic_rag.db"

NUMERIC_METRICS = [
    "groundedness",
    "baseline_groundedness",
    "citation_coverage",
    "context_coverage",
    "useful_ratio",
    "retrieval_recall",
    "retrieval_mrr",
    "total_latency_ms",
    "baseline_latency_ms",
    "added_latency_ms",
    "llm_calls",
    "tokens_used",
    "cost",
    "cost_per_recovery",
    "recovery_attempts",
    "successful_recoveries",
    "failures_detected",
]


def main() -> int:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM agent_runs").fetchall()

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            payload = json.loads(row["metrics"] or "{}")
        except Exception:
            payload = {}
        record = {
            "run_id": row["run_id"],
            "mode": row["mode"],
            "status": row["status"],
            "abstained": bool(row["abstained"]),
            "abstention": row["abstention"],
            "faults": json.loads(row["faults"] or "[]"),
            "metrics": payload,
        }
        buckets[row["mode"]].append(record)

    summary = []
    for mode, group in sorted(buckets.items()):
        stats = {"mode": mode, "runs": len(group)}
        stats["status_counts"] = dict(Counter(r["status"] for r in group))
        stats["abstention_rate_pct"] = round(
            100.0 * sum(1 for r in group if r["abstained"]) / len(group), 2)
        stats["runs_with_faults"] = sum(1 for r in group if r["faults"])
        for name in NUMERIC_METRICS:
            values = [
                float(r["metrics"][name])
                for r in group
                if isinstance(r["metrics"].get(name), (int, float))
            ]
            if values:
                stats[name] = {
                    "mean": round(statistics.fmean(values), 4),
                    "p50": round(statistics.median(values), 4),
                    "p95": round(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))], 4),
                }
        summary.append(stats)

    # ---- paired comparison on mode3 runs that carry a baseline --------------
    grounded_gains = []
    for record in buckets.get("mode3_agentic_with_recovery", []):
        metrics = record["metrics"]
        before = metrics.get("baseline_groundedness")
        after = metrics.get("groundedness")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            grounded_gains.append(float(after) - float(before))

    improved = sum(1 for g in grounded_gains if g > 1e-9)
    worsened = sum(1 for g in grounded_gains if g < -1e-9)
    unchanged = len(grounded_gains) - improved - worsened

    payload = {
        "by_mode": summary,
        "groundedness_delta_mode3": {
            "n": len(grounded_gains),
            "mean_delta": round(statistics.fmean(grounded_gains), 4) if grounded_gains else None,
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "improved_pct": round(100.0 * improved / len(grounded_gains), 2) if grounded_gains else None,
        },
    }
    conn.close()

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "by_mode.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # markdown mirror, so scripts/verify_paper_numbers.py can diff the document
    lines = [
        "| mode | runs | groundedness | citation cov. | recall | abstention % | +latency p50 | llm calls |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['runs']} | "
            f"{row.get('groundedness', {}).get('mean')} | "
            f"{row.get('citation_coverage', {}).get('mean')} | "
            f"{row.get('retrieval_recall', {}).get('mean')} | "
            f"{row['abstention_rate_pct']} | "
            f"{row.get('added_latency_ms', {}).get('p50')} | "
            f"{row.get('llm_calls', {}).get('mean')} |"
        )
    delta = payload["groundedness_delta_mode3"]
    lines += [
        "",
        "| quantity | value |",
        "|---|---|",
        f"| paired runs | {delta['n']} |",
        f"| mean groundedness delta | +{delta['mean_delta']} |",
        f"| improved | {delta['improved']} |",
        f"| worsened | {delta['worsened']} |",
        f"| unchanged | {delta['unchanged']} |",
        f"| improved % | {delta['improved_pct']} |",
    ]
    (DATA / "by_mode.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("| mode | runs | groundedness | citation cov. | recall | abstention % | +latency p50 | llm calls |")
    print("|---|---|---|---|---|---|---|---|")
    for row in summary:
        print(
            f"| {row['mode']} | {row['runs']} | "
            f"{row.get('groundedness', {}).get('mean')} | "
            f"{row.get('citation_coverage', {}).get('mean')} | "
            f"{row.get('retrieval_recall', {}).get('mean')} | "
            f"{row['abstention_rate_pct']} | "
            f"{row.get('added_latency_ms', {}).get('p50')} | "
            f"{row.get('llm_calls', {}).get('mean')} |"
        )
    print()
    print("status counts:", {r["mode"]: r["status_counts"] for r in summary})
    print("groundedness delta (mode3):", payload["groundedness_delta_mode3"])
    print(f"wrote: {(DATA / 'by_mode.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
