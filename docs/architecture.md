# Architecture

## 1. Purpose

The system makes failures inside an **agentic RAG pipeline observable**, diagnoses them
with a two-tier (rule + LLM) diagnoser, applies a **targeted, bounded** recovery,
**verifies** that the recovery actually worked, and **abstains** when no reliable answer
can be produced.

```
USER QUERY
   ↓
QUERY UNDERSTANDING            →  Monitor
   ↓
QUERY DECOMPOSITION / REWRITE
   ↓
RETRIEVAL                      →  Detect
   ↓
RANKING / RERANKING
   ↓
CONTEXT ASSEMBLY
   ↓
GENERATION
   ↓
ANSWER VALIDATION              →  Diagnose (Tier 1 rules → Tier 2 LLM)
   ↓
RECOVERY PLANNER               →  Recover (bounded loop)
   ↓
TARGETED RECOVERY              →  Re-run only the affected stage
   ↓
VERIFY                         →  Verify
   ↓
FINAL ANSWER  or  ABSTENTION
```

## 2. Package layout

```
agentic_rag_recovery/
├── settings.py            Configuration (YAML + dataclasses), path resolution
├── taxonomy.py            Failure taxonomy, stage vocabulary, recovery ladders
├── models.py              Pydantic domain models (run, diagnosis, recovery, …)
├── textutil.py            Deterministic text utilities (tokenising, stemming, …)
├── prompts.py             Prompt construction + machine-readable payload block
├── bootstrap.py           Application factory (db + repository + orchestrator)
├── db/
│   ├── schema.sql         SQLite schema (18 tables)
│   ├── database.py        Thread-safe connection factory, query helpers
│   └── repository.py      Persistence for runs, diagnoses, recoveries, metrics
├── embeddings/            Embedder interface, hashed TF-IDF, optional ST
├── ingestion/             Loaders (PDF/TXT/MD/DOCX/HTML/JSON/CSV), cleaning, chunking
├── retrieval/             Vector index, BM25, hybrid fusion, rerankers, diagnostics
├── signals/               Signal collection + deterministic custom metrics
├── detection/             Tier-1 rule engine
├── diagnosis/             Strict schema, Tier-2 LLM diagnoser, two-tier engine
├── recovery/              Budget, planner, strategies, bounded executor
├── validation/            Recovery verification
├── faults/                Fault catalogue + injector (with ground truth)
├── pipeline/              State, stages, runtime, orchestrator, runners
├── evaluation/            Datasets, metrics, experiment engine, optional adapters
└── observability/         Trace collector, logging, optional exporters
```

## 3. Layering rules

| Layer | May depend on | Must not depend on |
|---|---|---|
| `retrieval`, `ingestion`, `embeddings` | `models`, `settings`, `textutil` | `diagnosis`, `recovery`, `pipeline` |
| `signals` | `models`, `textutil`, `taxonomy`, `retrieval.diagnostics` | `diagnosis`, `recovery` |
| `detection` | `signals`, `taxonomy`, `settings` | LLM providers, database |
| `diagnosis` | `signals`, `detection`, `llm`, `taxonomy` | `recovery`, `pipeline` |
| `recovery` | `diagnosis`, `validation`, `taxonomy`, `pipeline.state` | `pipeline.orchestrator` |
| `pipeline` | everything below it | — |
| `backend` / `dashboard` | `bootstrap`, `evaluation` | each other (dashboard talks HTTP) |

`pipeline/__init__.py` resolves its heavy re-exports lazily (PEP 562) so that
`agentic_rag_recovery.faults` can be imported first without creating a cycle.

## 4. Execution runtime

`pipeline/runtime.py::Runtime` is the mutable object shared by the stages, the fault
injector and the recovery strategies. It owns:

* `settings`, `repository`, `embedder`, `retriever`, `provider`
* `knobs` (`GenerationKnobs`) — per-request configuration overrides
* `fault_flags` — the *only* place the injector may write
* `baseline_chunks` — a pristine snapshot used to restore state between requests
* `gold_chunk_ids` / `answer_keywords` — experiment ground truth (never given to the diagnoser)

### Targeted recovery

Every recovery strategy declares the **earliest stage that must be re-run**. Everything
before it is reused verbatim:

| Action | Re-runs from | Reuses |
|---|---|---|
| `QUERY_REWRITE` | retrieval | understanding, decomposition |
| `HYBRID_SEARCH` | retrieval | understanding, decomposition |
| `EXPAND_TOP_K` / `BROADEN_RETRIEVAL` | retrieval | understanding, decomposition |
| `RERANK` | ranking | retrieval candidate pool |
| `CHUNK_EXPANSION` / `CONTEXT_EXPANSION` | context assembly | retrieval, ranking |
| `CONSTRAINED_REGENERATION` | generation | the whole retrieved context |
| `BOUNDED_RETRY` / `ALTERNATE_PATH` / `TOOL_REPAIR` | earliest failed stage | earlier stages |

`RecoveryAttempt.skipped_stages` records exactly which stages were reused, and
`RecoveryAttempt.added_latency_ms` measures the real overhead.

## 5. Pipeline runners

* `SequentialPipelineRunner` — a plain state machine; always available.
* `LangGraphPipelineRunner` — the same stage functions wired into a LangGraph graph;
  falls back to the sequential runner when LangGraph is absent or compilation fails.

`build_runner(runtime, prefer="auto")` picks LangGraph when installed. Both runners
execute the *same* functions, so behaviour is identical.

## 6. Data flow of one request

1. `answer()` builds a `Runtime` + `TraceCollector` + `PipelineState`.
2. Ground truth (experiments only) is resolved and written to `runtime`, never to the diagnoser.
3. Faults are activated (experiment mode) by the injector.
4. Stages run once: understanding → decomposition → retrieval (+ranking) → context → generation → validation.
5. Agent-control signals are **frozen** so that repeats introduced by the recovery loop are not
   misread as agent failures.
6. Tier-1 rules evaluate; if confidence ≥ threshold the diagnosis is final, else Tier-2 runs.
7. The bounded recovery loop applies at most `max_recovery_attempts` distinct actions.
8. After each attempt the pipeline re-collects signals and verifies against
   class-specific criteria.
9. If nothing verified, an `Abstention` record is produced and the generation stage is
   replaced by a controlled abstention message.
10. The run, its steps, spans, diagnoses, recoveries, faults and metrics are persisted.

## 7. Data stores

* **SQLite** (`storage/agentic_rag.db`) — the single source of truth for runs, diagnoses,
  recoveries, faults, experiments and metrics.
* **Index files** (`storage/index/`) — optional vector/BM25 snapshots for fast reloads.
  The system can always rebuild from SQLite, so these are a cache, not a dependency.