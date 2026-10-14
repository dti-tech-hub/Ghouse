#!/usr/bin/env python
"""End-to-end demonstration - no API key, no network, no setup beyond
``pip install -r requirements.txt``.

Run it with::

    python run_demo.py              # full demonstration
    python run_demo.py --quick      # one question per scenario
    python run_demo.py --experiments
    python run_demo.py --reset      # wipe the SQLite database first

The demonstration walks through the ten scenarios required for a project
viva: healthy RAG, five injected failure families, diagnosis, targeted
recovery, a verified successful recovery, a failed recovery and an honest
abstention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import (  # noqa: E402
    bootstrap_demo_data,
    create_application,
    ensure_demo_ready,
)
from agentic_rag_recovery.evaluation.datasets import load_demo_dataset  # noqa: E402
from agentic_rag_recovery.evaluation.metrics import evaluate_run  # noqa: E402
from agentic_rag_recovery.evaluation.runner import ExperimentRunner  # noqa: E402
from agentic_rag_recovery.faults.definitions import FaultRegistry  # noqa: E402
from agentic_rag_recovery.models import RunMode  # noqa: E402
from agentic_rag_recovery.settings import get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# terminal formatting
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def _colour(text: str, code: str, enabled: bool = True) -> str:
    return f"{code}{text}{RESET}" if enabled else text


class Printer:
    def __init__(self, colour: bool = True):
        self.colour = colour

    def title(self, text: str) -> None:
        print()
        print(_colour("=" * 78, DIM, self.colour))
        print(_colour(f" {text}", BOLD + CYAN, self.colour))
        print(_colour("=" * 78, DIM, self.colour))

    def step(self, text: str) -> None:
        print(_colour(f"\n>>> {text}", BOLD, self.colour))

    def info(self, text: str) -> None:
        print(f"    {text}")

    def kv(self, key: str, value: Any, colour: str = "") -> None:
        rendered = str(value)
        if colour:
            rendered = _colour(rendered, colour, self.colour)
        print(f"    {key:<34} {rendered}")

    def good(self, text: str) -> None:
        print(_colour(f"    OK   {text}", GREEN, self.colour))

    def warn(self, text: str) -> None:
        print(_colour(f"    WARN {text}", YELLOW, self.colour))

    def bad(self, text: str) -> None:
        print(_colour(f"    FAIL {text}", RED, self.colour))

    def note(self, text: str) -> None:
        print(_colour(f"    {text}", DIM, self.colour))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
SCENARIOS: List[Dict[str, Any]] = [
    {
        "key": "healthy",
        "title": "1. HEALTHY AGENTIC RAG (no fault)",
        "question_id": "q001",
        "faults": [],
        "expect": "No failure detected, no recovery, grounded answer with citations.",
    },
    {
        "key": "retrieval",
        "title": "2. RETRIEVAL FAILURE - query drift, then QUERY_REWRITE recovery",
        "question_id": "q006",
        "faults": ["BAD_QUERY_REWRITE"],
        "expect": "Retrieval evidence does not cover the question; a rewrite recovers it.",
    },
    {
        "key": "ranking",
        "title": "3. RANKING FAILURE - corrupt reranker, then RERANK recovery",
        "question_id": "q001",
        "faults": ["RANKING_CORRUPTION"],
        "expect": "Retrieval finds the evidence, ranking loses it; re-ranking the cached pool recovers it.",
    },
    {
        "key": "context",
        "title": "4. CONTEXT FAILURE - truncated context, then CONTEXT_EXPANSION recovery",
        "question_id": "q003",
        "faults": ["CONTEXT_TRUNCATION"],
        "expect": "The context cannot cover the question until it is expanded.",
    },
    {
        "key": "generation",
        "title": "5. GENERATION FAILURE - grounding removed, then CONSTRAINED_REGENERATION",
        "question_id": "q013",
        "faults": ["GENERATION_CONSTRAINT_REMOVAL"],
        "expect": "Unsupported claims appear; regeneration under restored constraints fixes them.",
    },
    {
        "key": "conflict",
        "title": "6. CONTEXT FAILURE - conflicting evidence in the assembled context",
        "question_id": "q016",
        "faults": ["CONFLICTING_EVIDENCE"],
        "expect": "The context contradicts itself; detected and escalated.",
    },
    {
        "key": "infra",
        "title": "7. INFRASTRUCTURE FAILURE - retrieval timeout, then alternate path",
        "question_id": "q010",
        "faults": ["RETRIEVAL_TIMEOUT"],
        "expect": "The vector path fails; the independent BM25 path takes over.",
    },
    {
        "key": "stale",
        "title": "8. RETRIEVAL FAILURE - stale (superseded) document version",
        "question_id": "q019",
        "faults": ["STALE_DOCUMENT"],
        "expect": "Evidence is outdated; detected, recovery escalates to abstention.",
    },
    {
        "key": "unrecoverable",
        "title": "9. UNRECOVERABLE - the answer-bearing document is removed",
        "question_id": "q004",
        "faults": ["RELEVANT_DOC_REMOVAL"],
        "expect": "No safe recovery exists; the system must abstain rather than invent an answer.",
    },
    {
        "key": "abstain",
        "title": "10. BOUNDED LOOP EXHAUSTION - abstention after verified failures",
        "question_id": "q022",
        "faults": ["RELEVANT_DOC_REMOVAL", "EMPTY_RETRIEVAL"],
        "expect": "Multiple fixes attempted, none verified, attempt budget respected, abstention recorded.",
    },
]


def run_scenario(app: Any, printer: Printer, dataset: Any, scenario: Dict[str, Any]) -> Dict[str, Any]:
    printer.title(scenario["title"])
    question = dataset.by_id(scenario["question_id"])
    if question is None:
        printer.bad(f"question {scenario['question_id']} not found in the demo dataset")
        return {"key": scenario["key"], "ok": False}

    definition = FaultRegistry.get(scenario["faults"][0]) if scenario["faults"] else None
    printer.info(f"Question : {question.question}")
    if definition:
        printer.kv("Injected fault", f"{definition.fault_type} ({definition.title})", YELLOW)
        printer.kv("Expected class", f"{definition.expected_failure_class.value}/{definition.expected_subtype}")
        printer.kv("Expected symptom", definition.expected_symptom[:70])
    printer.kv("Reference answer", question.answer[:90])
    printer.note(f"Expectation: {scenario['expect']}")

    started = time.perf_counter()
    outcome = app.orchestrator.answer(
        question.question,
        mode=RunMode.MODE_3_AGENTIC_WITH_RECOVERY,
        faults=scenario["faults"] or None,
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
    )
    elapsed = (time.perf_counter() - started) * 1000

    # ---------------- pipeline trace ----------------
    printer.step("Pipeline stages")
    for step in outcome.run.steps:
        if step.stage in {"retrieval", "ranking", "generation", "diagnosis", "recovery_attempt", "abstain"}:
            icon = {"ok": "ok", "error": "ERR", "skipped": "skip", "recovered_not_verified": "unver"}.get(
                step.status, step.status
            )
            printer.info(f"  [{icon:>5}] {step.stage:<18} {step.latency_ms:7.2f} ms  {step.output_summary[:66]}")

    retrieval = outcome.run.retrieval_diagnostics
    if retrieval:
        printer.step("Retrieval diagnostics")
        printer.kv("method / top_k", f"{retrieval.method} / k={retrieval.top_k}")
        printer.kv("returned", retrieval.returned)
        printer.kv("question coverage (top)", f"{retrieval.top_overlap:.3f}")
        printer.kv("question coverage (mean)", f"{retrieval.mean_overlap:.3f}")
        printer.kv("useful ratio", f"{retrieval.useful_ratio:.3f}")
        printer.kv("score distribution", [round(s, 3) for s in retrieval.score_distribution[:6]])
        printer.kv("source diversity", f"{retrieval.source_diversity} ({retrieval.unique_documents} docs)")

    signals = outcome.run.signals
    if signals:
        printer.step("Signals")
        printer.kv("groundedness", f"{signals.generation.get('groundedness')}")
        printer.kv("citation coverage", f"{signals.generation.get('citation_coverage')}")
        printer.kv("context coverage", f"{signals.context.get('context_coverage')}")
        printer.kv("fragmentation / conflict", f"{signals.context.get('fragmentation_ratio')} / {signals.context.get('conflict_score')}")
        printer.kv("infra errors / timeouts", f"{signals.infrastructure.get('infra_errors')} / {signals.infrastructure.get('timeouts')}")

    # ---------------- diagnosis ----------------
    printer.step("Diagnosis (two-tier)")
    for index, diagnosis in enumerate(outcome.run.diagnoses, start=1):
        tag = "pre-recovery" if index == 1 else "post-recovery"
        printer.kv(f"[{index}] tier", diagnosis.tier.value, MAGENTA)
        printer.kv("    failure class", diagnosis.class_value if diagnosis.failure_detected else "NONE")
        printer.kv("    subtype / stage", f"{diagnosis.subtype} @ {diagnosis.stage.value if diagnosis.stage else '-'}")
        printer.kv("    severity / confidence", f"{diagnosis.severity.value} / {diagnosis.confidence:.3f}")
        printer.kv("    recommended recovery", diagnosis.recommended_recovery.value if diagnosis.recommended_recovery else "-")
        printer.kv(f"    evidence ({tag})", "; ".join(diagnosis.evidence[:2])[:110])
        if diagnosis.rule_id:
            printer.kv("    triggered rule", diagnosis.rule_id)

    # ---------------- recovery ----------------
    printer.step("Recovery (bounded loop)")
    printer.kv("max attempts configured", app.settings.budget.max_recovery_attempts)
    if not outcome.run.recoveries:
        printer.info("  (no recovery was required or planned)")
    for attempt in outcome.run.recoveries:
        colour = {"success": GREEN, "failure": YELLOW, "skipped": DIM, "error": RED}.get(
            attempt.status.value, ""
        )
        printer.kv(
            f"  attempt {attempt.attempt_index} {attempt.action.value}",
            attempt.status.value,
            colour,
        )
        if attempt.reused_stage:
            printer.info(f"        reused stages : {', '.join(attempt.skipped_stages)}")
        printer.info(f"        rationale      : {attempt.rationale[:100]}")
        verification = attempt.verification or {}
        if verification:
            failed = [c["criterion"] for c in verification.get("criteria", []) if not c["passed"]]
            printer.info(
                f"        verification   : passed={verification.get('passed')} "
                f"score={verification.get('score')} failed={failed or '-'}"
            )
        printer.info(f"        cost           : {attempt.latency_ms:.2f} ms, {attempt.tokens_used} tokens")

    # ---------------- outcome ----------------
    printer.step("Final answer")
    answer = outcome.run.answer or ""
    for line in _wrap(answer, 72)[:8]:
        printer.info(f"  {line}")
    if outcome.abstained:
        printer.bad("ABSTAINED - insufficient reliable evidence after bounded recovery")
        abstention = outcome.run.abstention
        if abstention:
            printer.kv("  reason", abstention.reason)
            printer.kv("  attempted recoveries", [a.value for a in abstention.attempted_recoveries])
            printer.kv("  attempts used", f"{abstention.attempts_used}/{abstention.max_attempts}")
            printer.kv("  budget consumed", {
                "latency_ms": abstention.budget_consumed.get("latency_used_ms"),
                "tokens": abstention.budget_consumed.get("tokens_used"),
                "llm_calls": abstention.budget_consumed.get("llm_calls_used"),
            })
    elif outcome.recovered:
        printer.good("RECOVERED - the repair was verified before the answer was accepted")
    else:
        printer.good("ANSWERED - no failure was detected")

    record = evaluate_run(
        outcome.run,
        state=outcome.state,
        reference_answer=question.answer,
        answer_keywords=question.answer_keywords,
        ground_truth=(outcome.state.ground_truth or {}),
        label=app.settings.labels.get("results", ""),
    )
    printer.step("Per-request metrics")
    printer.kv("groundedness", f"{record.groundedness:.3f}")
    printer.kv("answer token-F1 vs reference", f"{record.answer_f1:.3f}")
    printer.kv("retrieval recall / MRR", f"{record.recall} / {record.mrr}")
    printer.kv("total / added latency", f"{outcome.run.total_latency_ms:.2f} ms / {outcome.run.added_latency_ms:.2f} ms")
    printer.kv("llm calls / tokens / cost", f"{outcome.run.llm_calls} / {outcome.run.tokens_used} / {outcome.run.cost:.6f}")
    if scenario["faults"]:
        ground_truth = outcome.state.ground_truth or {}
        printer.kv(
            "diagnosis vs ground truth",
            f"expected {ground_truth.get('expected_subtype')} | got {record.diagnosis_tier}",
            GREEN if record.diagnosis_subtype_correct else YELLOW,
        )
    printer.note(f"run_id={outcome.run.run_id}  wall={elapsed:.1f} ms")

    return {
        "key": scenario["key"],
        "run_id": outcome.run.run_id,
        "abstained": outcome.abstained,
        "recovered": outcome.recovered,
        "diagnosis_subtype_correct": record.diagnosis_subtype_correct,
        "groundedness": record.groundedness,
        "answer_f1": record.answer_f1,
        "added_latency_ms": outcome.run.added_latency_ms,
    }


def _wrap(text: str, width: int) -> List[str]:
    words = (text or "").split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or ["(empty answer)"]


# ---------------------------------------------------------------------------
def run_experiments(app: Any, printer: Printer, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    runner = ExperimentRunner(app)
    names = names or runner.available_configs()
    results: List[Dict[str, Any]] = []
    printer.title("EXPERIMENTS - three-mode comparison, executed for real")
    header = f"{'experiment':<24}{'mode3 runs':>11}{'diag acc':>10}{'rec succ':>10}{'grounded':>10}{'abstain':>9}{'+ms':>8}"
    print(_colour(header, BOLD, printer.colour))
    for name in names:
        try:
            config = runner.load_config(name)
            result = runner.run(config)
        except Exception as exc:
            printer.bad(f"{name}: {exc}")
            continue
        m3 = result.mode3
        row = {
            "experiment": name,
            "experiment_id": result.experiment_id,
            "runs": m3.total_runs if m3 else 0,
            "diagnosis_accuracy": m3.diagnosis_accuracy if m3 else 0.0,
            "recovery_success_rate": m3.recovery_success_rate if m3 else 0.0,
            "groundedness": m3.groundedness if m3 else 0.0,
            "abstention_rate": m3.abstention_rate if m3 else 0.0,
            "added_latency_ms": m3.added_latency_ms if m3 else 0.0,
        }
        results.append(row)
        print(
            f"{row['experiment']:<24}{row['runs']:>11}{row['diagnosis_accuracy']:>10.3f}"
            f"{row['recovery_success_rate']:>10.3f}{row['groundedness']:>10.3f}"
            f"{row['abstention_rate']:>9.3f}{row['added_latency_ms']:>8.1f}"
        )
    printer.note(f"All figures above were produced by this run. Label: {app.settings.labels.get('results')}")
    return results


def print_fault_catalogue(printer: Printer) -> None:
    printer.title("FAULT INJECTION CATALOGUE (declared ground truth)")
    print(f"{'fault_type':<32}{'stage':<20}{'expected class':<26}{'expected subtype':<26}{'sev':<9}")
    for definition in FaultRegistry.list_all():
        print(
            f"{definition.fault_type:<32}{definition.target_stage.value:<20}"
            f"{definition.expected_failure_class.value:<26}{definition.expected_subtype:<26}"
            f"{definition.severity.value:<9}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the SQLite database before starting")
    parser.add_argument("--quick", action="store_true", help="run three representative scenarios only")
    parser.add_argument("--experiments", action="store_true", help="run the experiment suite instead of scenarios")
    parser.add_argument("--experiments-only", action="store_true", help="scenarios then experiments")
    parser.add_argument("--catalogue", action="store_true", help="print the fault catalogue and exit")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--json", dest="json_path", default="", help="write a machine-readable summary here")
    args = parser.parse_args(argv)

    printer = Printer(colour=not args.no_colour)

    if args.catalogue:
        print_fault_catalogue(printer)
        return 0

    if args.reset:
        db_path = get_settings().db_path()
        if db_path.exists():
            db_path.unlink()
            printer.info(f"removed {db_path}")

    printer.title("AGENTIC RAG FAILURE DIAGNOSIS AND RECOVERY SYSTEM - DEMONSTRATION")
    app = create_application()
    status = ensure_demo_ready(app)
    printer.kv("mode", app.settings.mode)
    printer.kv("LLM provider", f"{app.settings.generation.provider} ({app.settings.generation.model})")
    printer.kv("retrieval", f"{app.settings.retrieval.method} (top_k={app.settings.retrieval.top_k}, backend={app.orchestrator.retriever.vector_index.backend})")
    printer.kv("embeddings", f"{app.orchestrator.embedder.name} dim={app.orchestrator.embedder.dim}")
    printer.kv("documents / chunks", f"{status['documents']} / {status['chunks']}")
    printer.kv("database", status["db_path"])
    printer.kv("result label", app.settings.labels.get("results"), YELLOW)
    if app.settings.generation.provider == "mock":
        printer.note(
            "DEMO MODE: the LLM is a deterministic rule-based generator, not a language model. "
            "All numbers below are DEMO / MOCK RESULTS and are labelled as such in the database."
        )

    dataset = load_demo_dataset(app.settings)
    scenarios = SCENARIOS
    if args.quick:
        scenarios = [s for s in SCENARIOS if s["key"] in {"healthy", "generation", "unrecoverable"}]

    summary: Dict[str, Any] = {
        "label": app.settings.labels.get("results"),
        "scenarios": [],
        "experiments": [],
    }
    if not args.experiments_only:
        for scenario in scenarios:
            summary["scenarios"].append(run_scenario(app, printer, dataset, scenario))
    if args.experiments or args.experiments_only:
        summary["experiments"] = run_experiments(app, printer)

    printer.title("SUMMARY")
    recovered = sum(1 for s in summary["scenarios"] if s.get("recovered"))
    abstained = sum(1 for s in summary["scenarios"] if s.get("abstained"))
    correct = sum(1 for s in summary["scenarios"] if s.get("diagnosis_subtype_correct"))
    printer.kv("scenarios executed", len(summary["scenarios"]))
    printer.kv("diagnosis matched ground truth", f"{correct}/{len(summary['scenarios'])}")
    printer.kv("verified recoveries", recovered)
    printer.kv("abstentions", abstained)
    printer.kv("experiments executed", len(summary["experiments"]))
    printer.kv("sqlite database", str(app.settings.db_path()))
    printer.note("Every number above comes from an executed run stored in SQLite; nothing is hard-coded.")

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        printer.note(f"machine-readable summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())