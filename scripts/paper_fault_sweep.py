"""Per-fault sweep: real detection and recovery outcomes against ground truth.

For every fault in the catalogue this injects the fault, runs the full
agentic pipeline, and records what the system actually did:

* whether the injected fault was **detected** at all,
* whether the top diagnosis matched the declared ground-truth subtype,
* the diagnosis rank, tier (rule/LLM), confidence and latency,
* the recovery actions attempted, whether verification passed, whether the run
  was recovered or honestly abstained,
* the end-to-end latency and the groundedness before/after recovery.

Ground truth lives in ``FaultDefinition`` and is read *only* to score the run
after the fact; it is never exposed to the diagnoser.

Output: docs/paper_data/fault_sweep.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import create_application, ensure_demo_ready  # noqa: E402
from agentic_rag_recovery.evaluation.datasets import load_demo_dataset  # noqa: E402
from agentic_rag_recovery.faults.definitions import FaultRegistry  # noqa: E402
from agentic_rag_recovery.models import RunMode  # noqa: E402

OUT = ROOT / "docs" / "paper_data"


def first_diagnosis(outcome):
    diagnoses = getattr(outcome.run, "diagnoses", None) or []
    return diagnoses[0] if diagnoses else None


def score(outcome, definition, question) -> dict:
    truth = definition.expected_subtype
    diagnoses = getattr(outcome.run, "diagnoses", None) or []
    top = diagnoses[0] if diagnoses else None

    predicted = getattr(top, "subtype", None) if top else None
    matched = predicted == truth
    rank = None
    if matched is False:
        for index, d in enumerate(diagnoses):
            if getattr(d, "subtype", None) == truth:
                rank = index
                break
    else:
        rank = 0

    recoveries = []
    for attempt in getattr(getattr(outcome.run, "recoveries", None) or [], "__iter__", lambda: [])():
        verification = getattr(attempt, "verification", None) or {}
        recoveries.append({
            "action": str(getattr(attempt, "action", "")).rsplit(".", 1)[-1],
            "status": str(getattr(attempt, "status", "")),
            "verified": bool(verification.get("passed")) if isinstance(verification, dict) else None,
            "re_run_stage": str(getattr(attempt, "re_run_stage", "")),
            "added_latency_ms": round(float(getattr(attempt, "added_latency_ms", 0.0) or 0.0), 3),
            "reused_stages": list(getattr(attempt, "skipped_stages", []) or []),
            "failed_criteria": [
                c.get("criterion")
                for c in (verification.get("criteria", []) if isinstance(verification, dict) else [])
                if not c.get("passed")
            ],
        })

    state = getattr(outcome, "state", None)
    signals = getattr(state, "signals", None) or getattr(outcome, "run", None) and outcome.run.signals or {}
    signals = signals if isinstance(signals, dict) else {}
    gen_before = (signals.get("generation") or {}).get("groundedness")

    return {
        "run_id": outcome.run.run_id,
        "question": question.question,
        "expected_subtype": truth,
        "expected_class": definition.expected_class if hasattr(definition, "expected_class") else None,
        "target_stage": definition.target_stage,
        "detected": bool(diagnoses),
        "predicted_subtype": predicted,
        "predicted_class": getattr(top, "class_value", None) if top else None,
        "match": matched,
        "match_rank": rank,
        "n_diagnoses": len(diagnoses),
        "tier": str(getattr(top, "tier", "")) if top else None,
        "confidence": round(float(getattr(top, "confidence", 0.0)), 4) if top else None,
        "diagnosis_latency_ms": round(
            sum(float(getattr(s, "latency_ms", 0.0)) for s in outcome.run.steps
                if str(getattr(s, "stage", "")) == "diagnosis"), 3),
        "rule_id": getattr(top, "rule_id", None) if top else None,
        "evidence": list(getattr(top, "evidence", []) or []) if top else [],
        "status": getattr(outcome.run, "status", None) or ("abstained" if outcome.abstained else ("recovered" if outcome.recovered else "success")),
        "recovered": bool(getattr(outcome, "recovered", False)),
        "abstained": bool(getattr(outcome, "abstained", False)),
        "recovery_actions": recoveries,
        "n_recovery_attempts": len(recoveries),
        "verified_recovery": any(r["verified"] for r in recoveries),
        "groundedness": None if gen_before is None else round(float(gen_before), 4),
        "total_latency_ms": round(float(outcome.run.total_latency_ms), 3),
        "added_latency_ms": round(sum(r["added_latency_ms"] for r in recoveries), 3),
    }


def main() -> int:
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.mkdir(parents=True, exist_ok=True)

    app = create_application()
    ensure_demo_ready(app)
    dataset = load_demo_dataset(app.settings)
    questions = dataset.questions
    registry = FaultRegistry()

    print(f"faults={len(registry.list_all())} questions={len(questions)} repetitions={repetitions}")
    print(f"label: {app.settings.labels.get('results')}")
    print()

    rows: list[dict] = []
    for definition in registry.list_all():
        fault_type = definition.fault_type
        # one question per repetition, cycled, so the sweep is not tied to a
        # single query while remaining deterministic
        for rep in range(repetitions):
            question = questions[rep % len(questions)]
            started = time.perf_counter()
            outcome = app.orchestrator.answer(
                question.question,
                mode=RunMode.MODE_3_AGENTIC_WITH_RECOVERY,
                faults=[fault_type],
                relevant_documents=question.relevant_documents,
                answer_keywords=question.answer_keywords,
            )
            wall = (time.perf_counter() - started) * 1000
            record = score(outcome, definition, question)
            record["fault_type"] = fault_type
            record["wall_ms"] = round(wall, 3)
            rows.append(record)
            print(
                f"  {fault_type:<30} rep{rep} "
                f"expected={definition.expected_subtype:<28} "
                f"got={str(record['predicted_subtype']):<28} "
                f"{'MATCH' if record['match'] else ('miss@' + str(record['match_rank']) if record['match_rank'] is not None else 'MISS'):<9} "
                f"rec={record['recovered']} abst={record['abstained']}"
            )

    (OUT / "fault_sweep.json").write_text(json.dumps({
        "label": app.settings.labels.get("results"),
        "repetitions": repetitions,
        "questions": [q.question for q in questions],
        "runs": rows,
    }, indent=2, default=str), encoding="utf-8")

    matched = sum(1 for r in rows if r["match"])
    detected = sum(1 for r in rows if r["detected"])
    recovered = sum(1 for r in rows if r["recovered"])
    print()
    print(f"runs            : {len(rows)}")
    print(f"detected        : {detected}/{len(rows)}")
    print(f"top-1 subtype   : {matched}/{len(rows)}")
    print(f"recovered       : {recovered}/{len(rows)}")
    print(f"median latency  : {statistics.median(r['total_latency_ms'] for r in rows):.2f} ms")
    print(f"wrote           : {(OUT / 'fault_sweep.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
