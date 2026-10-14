# HTTP API

Base URL: `http://127.0.0.1:8000` · Interactive docs: `/docs` · Schema: `/openapi.json`

Start with:

```bash
uvicorn backend.main:app --reload
```

## 1. System

| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, version, provider, model, document/chunk counts, database path, labels |
| GET | `/system` | health + external evaluation availability + fault count + experiment configs |
| GET | `/config` | full settings, failure taxonomy, fault catalogue, budgets |
| GET | `/taxonomy` | stages, classes and every subtype with its recovery ladder |
| GET | `/recovery/strategies` | registered strategies, actions, classes, stages |
| GET | `/` | endpoint index |

## 2. Knowledge base

| Method | Path | Description |
|---|---|---|
| GET | `/documents` | list documents (`active_only`, `limit`) with chunk counts |
| GET | `/documents/{id}` | one document with all of its chunks |
| POST | `/documents/ingest` | ingest a server-side path (`{"path": "...", "force": false}`) |
| POST | `/documents/text` | ingest raw text (`{"text": "...", "source": "..."}`) |
| POST | `/documents/reindex` | rebuild vector + BM25 indexes |
| GET | `/datasets` | demo dataset summary, every demo question, public dataset availability |

## 3. Query

| Method | Path | Description |
|---|---|---|
| POST | `/query` | full pipeline with optional fault injection |
| POST | `/query/recover` | explicit recovery entry point (same behaviour) |
| GET | `/query/demo` | run one demo question (`question_id`, optional `fault`, `recover`) |

### `POST /query` request

```json
{
  "query": "Within how many minutes must a Severity 1 incident be acknowledged?",
  "mode": "mode3_agentic_with_recovery",
  "faults": ["GENERATION_CONSTRAINT_REMOVAL"],
  "fault_parameters": {},
  "top_k": 5,
  "retrieval_method": "hybrid",
  "use_rerank": true,
  "answer_keywords": ["severity", "acknowledged", "5", "minutes"],
  "relevant_documents": ["incident_response_runbook.md"],
  "persist": true,
  "recover": true
}
```

`mode` ∈ `mode1_normal_rag | mode2_agentic_no_recovery | mode3_agentic_with_recovery`.
`recover: false` with mode 3 is downgraded to mode 2 (diagnosis reported, no action taken).

### `POST /query` response

Returns the complete pipeline artefact, in the order the dashboard renders it:

`run_id`, `mode`, `status`, `query`, `answer`, `baseline_answer`, `abstained`,
`abstention`, `recovered`, `understanding`, `sub_results`, `retrieved`,
`retrieval_diagnostics`, `context`, `generation`, `signals`, `diagnoses`,
`recoveries`, `steps`, `metrics`, `faults`, `labels`, `total_latency_ms`,
`baseline_latency_ms`, `added_latency_ms`.

## 4. Fault injection

| Method | Path | Description |
|---|---|---|
| GET | `/faults` | the catalogue with declared ground truth for every fault |
| POST | `/faults/inject` | record a fault, or inject **and run** when `query` is supplied |

```bash
curl -X POST localhost:8000/faults/inject -H 'content-type: application/json' \
  -d '{"fault_type":"RANKING_CORRUPTION","parameters":{"mode":"reverse"}}'
```

With a `query`, the response embeds the full `QueryResponse` so the caller can compare the
declared ground truth against the diagnosis the system actually produced.

## 5. Experiments

| Method | Path | Description |
|---|---|---|
| GET | `/experiments` | experiment history |
| GET | `/experiments/{id}` | one experiment with metrics and runs |
| GET | `/experiments/configs` | stored configuration names |
| POST | `/experiments/run` | run a stored config or an ad-hoc experiment |
| POST | `/experiments/batch` | run several stored configs |

```json
{
  "name": "generation_failure",
  "modes": ["mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"],
  "fault_type": "GENERATION_CONSTRAINT_REMOVAL",
  "question_ids": ["q001", "q013"],
  "repetitions": 1,
  "seed": 42,
  "persist": true
}
```

## 6. Observability

| Method | Path | Description |
|---|---|---|
| GET | `/traces` | recent runs |
| GET | `/traces/{run_id}` | steps, spans, signals, diagnoses, recoveries, retrieved chunks, metrics |
| GET | `/diagnoses` | recent diagnoses |
| GET | `/diagnoses/{id}` | one diagnosis including raw LLM output and validation errors |
| GET | `/metrics` | counts, failures by class/stage, severity distribution, aggregates |
| GET | `/metrics/runs` | per-run metric rows |
| GET | `/abstentions` | abstention records |
| GET | `/models` | registered models |

## 7. Input validation and safety

* Every body is a Pydantic model with `extra="forbid"` where appropriate, so unknown fields
  are rejected rather than ignored.
* `query`: 1–4000 characters, non-blank after stripping.
* `mode`, `retrieval_method`, `retrieval_method` and fault names are constrained to
  enumerated values; an unknown fault type returns `404` with the catalogue hint.
* `parameters` must be a JSON object.
* **No user input is ever executed.** Injection attempts such as
  `{"query": "__import__('os').system('echo pwned')"}` are stored as literal text
  (`tests/test_api.py::test_code_like_input_is_treated_as_text`).
* File ingestion is **server-side path only**; missing paths return `404` rather than
  touching the filesystem arbitrarily.
* Errors are normalised: `ValueError → 422`, `FileNotFoundError → 404`, with a JSON body
  `{"detail": ..., "code": ..., "hint": ...}`.
* Every response carries `X-Response-Time-ms`.

## 8. CORS

`CORSMiddleware` is enabled for all origins with no credentials, so the Streamlit dashboard
can be served from a different port during development. Tighten it for production.