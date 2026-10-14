# Agent notes — how to work in this repository

## Commands

```bash
# environment
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# database + data
python scripts/init_db.py --reset
python scripts/ingest.py --force

# demonstration
python run_demo.py                     # 10 scenarios, full trace output
python run_demo.py --quick             # 3 scenarios
python run_demo.py --experiments       # experiment suite
python run_demo.py --catalogue         # fault catalogue with ground truth

# services
uvicorn backend.main:app --reload      # http://127.0.0.1:8000/docs
streamlit run dashboard/app.py         # http://127.0.0.1:8501

# experiments
python scripts/run_experiments.py --config baseline
python scripts/run_experiments.py --all

# tests + validation
pytest
pytest --cov=agentic_rag_recovery --cov-report=term-missing
python scripts/validate_all.py         # full end-to-end validation
python scripts/db_summary.py           # what is stored in SQLite
```

## Invariants — do not break these

1. **The diagnoser must never see fault ground truth.** Ground truth lives in
   `runtime.gold_chunk_ids` / `FaultInjection` and is written to SQLite before the run.
   `tests/test_faults.py::test_ground_truth_is_recorded_not_leaked` enforces this.
2. **Recovery must be targeted.** Every strategy declares the earliest stage to re-run;
   earlier state is reused. Never re-run the whole pipeline as a recovery.
3. **Recovery must be verified.** Never set `RecoveryOutcome.SUCCESS` without running
   `RecoveryVerifier`.
4. **Never repeat an action.** `BudgetTracker.can_attempt()` returns `BLOCK_REPEAT`, and the
   planner refuses no-op configurations.
5. **Every ladder terminates in `ABSTAIN`.** Enforced by `tests/test_recovery.py`.
6. **Never fabricate a result.** All metrics come from executed runs; mock-provider results
   carry `labels.results = "DEMO / MOCK RESULTS"`.
7. **Never widen a threshold to make something pass.** Thresholds live in
   `configs/default.yaml` and are only changed with a documented calibration reason.
8. **User input is never executed.** Only text reaches the system.

## Where things live

| Task | File |
|---|---|
| Add a failure subtype | `agentic_rag_recovery/taxonomy.py` (`_define`) + `detection/rules.py` |
| Add a recovery action | `taxonomy.py` + `recovery/strategies.py` (`@strategy`) |
| Add a fault | `faults/definitions.py` + `faults/injector.py` (`_apply_*`) |
| Add a signal | `signals/collector.py` + `detection/rules.py` |
| Add a metric | `signals/metrics.py` or `evaluation/metrics.py` |
| Add an API endpoint | `backend/main.py` + `backend/schemas.py` |
| Add a dashboard page | `dashboard/pages/N_Name.py` + `dashboard/api_client.py::PAGES` |
| Add an experiment | `experiments/configs/<name>.yaml` (a `seed` is mandatory) |

## Gotchas found the hard way

* **Ranking happens in `stage_ranking`, not in `HybridRetriever.search`.** The retriever is
  called with `use_rerank=False`; otherwise a `reverse` reranker is applied twice and the
  `RANKING_CORRUPTION` fault silently cancels out.
* **`HybridRetriever.rebuild()` refits the embedder.** Without it, re-chunking leaves the
  dense vectors with stale IDF weights and retrieval silently degrades.
* **Agent-control signals are frozen at baseline** (`state.metadata["agent_signals_snapshot"]`),
  otherwise repeats introduced by our own recovery loop are misread as agent failures.
* **Detection rules compare lexical coverage, not `top_score`.** Cosine, BM25 and RRF scores
  are on incomparable scales.
* **`pipeline/__init__.py` resolves exports lazily** (PEP 562) to avoid a cycle with
  `faults.injector`. Keep it lazy.
* **Tests must use the `settings` fixture.** An autouse fixture points the backend at the
  isolated test database; without it the API tests write into `storage/agentic_rag.db`.
* **Sentence splitting rejoins hard-wrapped lines first.** Markdown line breaks inside a
  sentence would otherwise corrupt the groundedness and conflict metrics.