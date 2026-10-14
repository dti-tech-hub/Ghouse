# Testing

```bash
pytest                      # everything
pytest -v                   # per-test names
pytest tests/test_recovery.py -q
pytest -k "abstention or bounded" 
pytest --cov=agentic_rag_recovery --cov-report=term-missing
```

Current suite: **214 tests, all passing**.

## 1. Coverage map

| File | Covers |
|---|---|
| `conftest.py` | isolated settings (temp DB), populated app, demo dataset, embeddings, chunks, retriever; **autouse** backend isolation so API tests never touch the developer's database |
| `test_ingestion.py` | loaders (txt/md/json/pdf/docx dispatch, unknown type, missing file, directory), cleaning, chunking (size/overlap/offsets/metadata/configurability), idempotent ingestion, text ingestion |
| `test_retrieval.py` | hashing embedder (determinism, L2 norm, meaningful similarity, state round-trip, rejection of bad input), vector index (build/search/save/load/mismatch/empty), BM25 (ranking, zero scores), hybrid retriever (methods, result fields, diagnostics, rebuild refits, service outage, timeout, embedding fallback, document removal), rerankers (lexical, corrupt, factory) |
| `test_signals_diagnosis.py` | custom metrics (claims, groundedness, citation coverage, conflicts incl. negative cases, coverage, IR metrics, correctness, repetition), signal collector, **every** Tier-1 rule family, diagnosis schema (valid, repairs, no-repair, garbage text), two-tier engine (rule tier, LLM tier, no-crash, stats), LLM diagnoser |
| `test_recovery.py` | taxonomy ladders, budget (snapshot, repeat block, attempt block, token block, latency, cost), planner (ordering, repeat block, no-op block, budget, healthy), every action has a strategy, targeted re-runs (RERANK, CONSTRAINED_REGENERATION), ceiling refusal, verification, end-to-end recovery/abstention/mode-2 |
| `test_faults.py` | catalogue completeness (15 required faults), declared ground truth, parameter build, ladder availability, unknown-fault rejection, **ground truth never leaks to the diagnoser**, every fault runs end-to-end, release restores the runtime, diagnosis accuracy against ground truth |
| `test_database.py` | schema (18 tables), primary-key lookup, insert, reset, decode, document/chunk round-trips, run/diagnosis/abstention/fault/metrics persistence, experiment lifecycle, model & config registration, counts, traces |
| `test_api.py` | health, OpenAPI, config, taxonomy, strategies, documents, ingestion, datasets, query (all three modes), fault injection, experiments, traces, diagnoses, metrics, abstentions, models, **input validation and injection safety** |
| `test_evaluation.py` | dataset loading, diagnosis scoring (strict/class/acceptable), aggregation, mode comparison, distributions, run evaluation, config discovery & seeds, execution, **reproducibility**, fault sweep, full persistence |

## 2. Test design principles

* **No hidden global state.** Every test gets an isolated settings object and a temporary
  SQLite database. An autouse fixture points the FastAPI backend at the same isolated store.
* **Assertions on behaviour, not implementation.** Tests check that a query retrieves the
  answer-bearing chunk, not that a specific private function was called.
* **Negative cases are first class.** "The conflict metric must *not* fire on a contrastive
  pair", "the recovery planner must refuse a no-op action", "a blank query must be
  rejected" — these protect against the over-detection failure mode that makes such systems
  useless.
* **Tests found real bugs.** The suite caught: a pipeline executed seven times per request,
  state leaking between requests, a double reranking that silently cancelled the
  `RANKING_CORRUPTION` fault, stale embedder statistics after re-chunking, and a circular
  import between `pipeline` and `faults`.

## 3. Markers

```bash
pytest -m "not slow"       # skip long runs
```

`slow` and `integration` markers are declared in `pyproject.toml`.

## 4. Adding a test

1. Put it in the file matching the subsystem (`test_recovery.py` for recovery, …).
2. Use the fixtures — do not create your own `Settings` or `Database`.
3. For a new fault: add the catalogue entry, the injector handler, and let
   `test_every_fault_runs_end_to_end` exercise it.
4. For a new recovery action: add the taxonomy entry, a `@strategy(...)` function and
   `test_every_action_has_a_strategy` will enforce the registration.