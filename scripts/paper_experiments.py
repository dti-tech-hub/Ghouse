"""Extract per-experiment metrics from SQLite into an ablation table.

Reads the ``experiments`` and ``metrics`` tables written by
``scripts/run_experiments.py`` and emits one row per experiment configuration,
which is the ablation table in the paper.

Writes docs/paper_data/experiments.json
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "docs" / "paper_data"
DB = ROOT / "storage" / "agentic_rag.db"

MODE_METRICS = [
    "groundedness",
    "answer_correctness",
    "abstention_rate",
    "diagnosis_accuracy",
    "recovery_success_rate",
    "retrieval_mrr",
    "context_coverage",
    "citation_coverage",
    "total_latency_ms",
    "added_latency_ms",
    "llm_calls",
    "cost",
]

SUMMARY_METRICS = [
    "diagnosis_accuracy",
    "recovery_success_rate",
    "groundedness",
    "answer_correctness",
    "abstention_rate",
    "added_latency_ms",
    "cost_per_recovery",
    "retrieval_mrr",
    "context_coverage",
    "citation_coverage",
    "recovery_attempts",
    "recovered",
    "abstained",
    "llm_calls",
]


def main() -> int:
    if not DB.exists():
        print("no database")
        return 1
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    experiments = [dict(r) for r in conn.execute("SELECT * FROM experiments ORDER BY started_at")]
    if not experiments:
        print("no experiments recorded - run: python scripts/run_experiments.py --all")
        return 1

    # metric rows keyed by experiment, preferring aggregate (non per_run) values
    by_experiment: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in conn.execute(
        "SELECT experiment_id, name, value, value_json, group_name FROM metrics "
        "WHERE experiment_id IS NOT NULL AND experiment_id != 'None'"
    ):
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        by_experiment[row["experiment_id"]][row["name"]].append(value)

    # per (experiment, mode) metrics: the three operating modes must never be
    # averaged together, they are the ablation axis of this study
    run_mode = {
        r["run_id"]: r["mode"]
        for r in conn.execute("SELECT run_id, mode FROM agent_runs")
    }
    by_mode: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in conn.execute(
        "SELECT run_id, name, value FROM metrics "
        "WHERE run_id IS NOT NULL AND run_id != 'None'"
    ):
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        mode = run_mode.get(row["run_id"])
        if mode is None:
            continue
        by_mode[("per_run", mode)][row["name"]].append(value)

    out_rows = []
    mode_rows = []
    for experiment in experiments:
        eid = experiment["experiment_id"]
        metrics = by_experiment.get(eid, {})
        row = {
            "experiment_id": eid,
            "config": experiment["name"],
            "description": experiment["description"],
            "dataset": experiment["dataset"],
            "fault_types": json.loads(experiment["fault_types"] or "[]"),
            "model": experiment["model"],
            "modes": (experiment["mode"] or "").split(","),
            "repetitions": experiment["repetitions"],
            "seed": experiment["seed"],
            "status": experiment["status"],
            "config_hash": experiment["config_hash"],
            "retrieval_config": json.loads(experiment["retrieval_config"] or "{}"),
            "recovery_config": json.loads(experiment["recovery_config"] or "{}"),
            "started_at": experiment["started_at"],
            "finished_at": experiment["finished_at"],
            "label": experiment["result_label"],
        }
        for name in SUMMARY_METRICS:
            values = metrics.get(name) or []
            row[name] = round(statistics.fmean(values), 4) if values else None
            row[f"{name}_n"] = len(values)
        out_rows.append(row)

    for (_, mode), metrics in by_mode.items():
        record = {"mode": mode}
        for name in MODE_METRICS:
            values = metrics.get(name) or []
            record[name] = round(statistics.fmean(values), 4) if values else None
        mode_rows.append(record)
    mode_rows.sort(key=lambda r: r["mode"])

    payload = {
        "experiments": out_rows,
        "by_mode": mode_rows,
        "totals": {
            "experiments": len(out_rows),
            "runs": conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0],
            "diagnoses": conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0],
            "recovery_attempts": conn.execute("SELECT COUNT(*) FROM recovery_attempts").fetchone()[0],
            "fault_injections": conn.execute("SELECT COUNT(*) FROM fault_injections").fetchone()[0],
            "metrics_rows": conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0],
            "queries": conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0],
            "retrieval_runs": conn.execute("SELECT COUNT(*) FROM retrieval_runs").fetchone()[0],
            "trace_spans": conn.execute("SELECT COUNT(*) FROM trace_spans").fetchone()[0],
        },
    }
    conn.close()

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "experiments.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    header = (
        "| config | runs | diag acc | rec succ | grounded | answer F1 | abstain | +ms |"
        " cost/rec |"
    )
    print(header)
    print("|" + "---|" * 9)
    for row in out_rows:
        print(
            f"| {row['config']} | {row['repetitions']} | "
            f"{row['diagnosis_accuracy']} | {row['recovery_success_rate']} | "
            f"{row['groundedness']} | {row['answer_correctness']} | "
            f"{row['abstention_rate']} | {row['added_latency_ms']} | "
            f"{row['cost_per_recovery']} |"
        )
    print()
    print("by operating mode (the ablation axis):")
    print("| mode | grounded | answer F1 | abstain | diag acc | rec succ | +ms | llm calls |")
    print("|---|---|---|---|---|---|---|---|")
    for row in mode_rows:
        print(
            f"| {row['mode']} | {row.get('groundedness')} | {row.get('answer_correctness')} | "
            f"{row.get('abstention_rate')} | {row.get('diagnosis_accuracy')} | "
            f"{row.get('recovery_success_rate')} | {row.get('added_latency_ms')} | "
            f"{row.get('llm_calls')} |"
        )
    print()
    print("totals:", json.dumps(payload["totals"]))
    print(f"wrote  : {(DATA / 'experiments.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
