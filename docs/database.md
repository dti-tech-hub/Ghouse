# Database

Primary store: **SQLite** (`storage/agentic_rag.db`). No PostgreSQL, MySQL or MongoDB.

Schema: `agentic_rag_recovery/db/schema.sql`
Access: `db/database.py` (connection factory), `db/repository.py` (persistence)

## 1. Design

* **Thread-safe** — one connection per thread (`threading.local`), WAL journal, foreign keys on.
* **Idempotent** — every write uses `INSERT OR REPLACE`, so re-running the demo never
  duplicates rows.
* **Self-describing** — the JSON payload of each complex column is decoded on read
  (`decode_row` / `Repository.decode`).
* **Provenance-first** — experiments record their config hash, seed, retrieval config,
  recovery config and result label, so any metric can be traced back to the run that
  produced it and to the configuration under which it ran.

## 2. Tables

### Knowledge base

| Table | Purpose |
|---|---|
| `documents` | one row per ingested document (`checksum` is unique, enabling idempotent ingestion) |
| `chunks` | one row per chunk with `document_id`, `chunk_index`, `char_start/end`, `page`, `section` |
| `index_versions` | index build metadata (backend, embedding model, dimension, chunk config) |

### Queries and retrieval

| Table | Purpose |
|---|---|
| `queries` | raw query text, rewrite, sub-questions, intent |
| `retrieval_runs` | one row per retrieval: method, top-k, latency, full diagnostics JSON |
| `retrieved_chunks` | ranked results with dense/sparse/rerank scores and the gold-truth flag |

### Agentic execution

| Table | Purpose |
|---|---|
| `agent_runs` | the run: mode, status, answer, baseline answer, understanding, context, generation, signals, metrics, faults, labels, latency, tokens, cost, abstention |
| `agent_steps` | ordered stage/action log with status, latency and action signature |
| `trace_spans` | OpenTelemetry-style spans with inputs, outputs, metrics and tags |

### Diagnosis, faults, recovery

| Table | Purpose |
|---|---|
| `diagnoses` | class, stage, subtype, severity, confidence, **tier**, rule id, evidence, recovery ladder, signals used, validation errors, repairs, raw LLM output |
| `fault_injections` | declared ground truth: expected class, subtype, symptom, injection parameters |
| `recovery_attempts` | action, target stage, status, rationale, reused stage, skipped stages, verification JSON, before/after state, latency, tokens, cost |
| `abstentions` | reason, failure class, attempted recoveries, attempts used, budget consumed |

### Experiments and metrics

| Table | Purpose |
|---|---|
| `experiments` | name, dataset, fault types, model, retrieval/recovery config, repetitions, seed, status, config hash, result label |
| `experiments_config` | stored configuration bodies |
| `metrics` | `(experiment_id, run_id, name, value, group_name, label, seed)` — the metric store |
| `models` | provider, model, params, token prices, default flag |
| `configurations` | named configuration snapshots with a content hash |

## 3. Metric storage

Metrics are written as long-format rows, which makes aggregation trivial and keeps every
number attributable:

```
group_name = "per_run"                     → one row per metric per request
group_name = "summary:mode3_agentic_with_recovery"  → one row per metric per experiment mode
label      = "DEMO / MOCK RESULTS"
```

`Repository.aggregate_metrics(group_name)` returns `AVG(value) GROUP BY name`, which is what
`GET /metrics` serves and what the dashboard charts.

## 4. Concurrency and safety

* WAL mode allows readers while a write is in progress.
* `Database.transaction()` is a re-entrant lock around commit/rollback.
* All writes go through `execute`/`execute_many`/`insert`, so parameter binding is always used
  (no string interpolation of values anywhere in the data path).
* `Database.reset()` drops and recreates every table (used by tests and `scripts/init_db.py`).

## 5. Rebuilding

```bash
python scripts/init_db.py --reset      # drop + recreate schema
python scripts/ingest.py --force       # re-ingest the demo corpus
```

Indexes are a **cache**, not a dependency: the system always rebuilds them from
`documents`/`chunks`, so losing `storage/index/` costs only start-up time.