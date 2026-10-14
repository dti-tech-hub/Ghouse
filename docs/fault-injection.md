# Fault Injection

Source: `agentic_rag_recovery/faults/`.

## 1. Purpose

Fault injection is the **experimental instrument** of this project. It makes a failure
reproducible, known in advance and scoreable, which is what allows diagnosis accuracy and
recovery success rate to be measured at all.

Each fault declares its own ground truth:

```python
FaultDefinition(
    fault_type="RANKING_CORRUPTION",
    title="Ranking / reranking corruption",
    target_stage=Stage.RANKING,
    description="...",
    expected_symptom="relevant chunks move down, irrelevant chunks rise",
    expected_failure_class=FailureClass.RANKING_FAILURE,
    expected_subtype="poor_reranking",
    severity=Severity.MEDIUM,
    default_parameters={"mode": "reverse"},
    category="algorithmic",
    targets=("poor_reranking", "irrelevant_ranked_high", "relevant_ranked_low"),
)
```

`expected_failure_class` / `expected_subtype` are written to SQLite **before** the run and
are used only when scoring. The detector, the diagnoser and the planner never receive them —
`tests/test_faults.py::test_ground_truth_is_recorded_not_leaked` asserts this.

## 2. Catalogue

| # | Fault type | Target stage | Expected class / subtype | Category |
|---|---|---|---|---|
| 1 | `QUERY_CORRUPTION` | query_understanding | RETRIEVAL_FAILURE / low_recall | semantic |
| 2 | `BAD_QUERY_REWRITE` | query_understanding | QUERY_FAILURE / incorrect_rewrite | agentic |
| 3 | `RELEVANT_DOC_REMOVAL` | retrieval | RETRIEVAL_FAILURE / no_useful_evidence | data |
| 4 | `IRRELEVANT_DOC_INJECTION` | retrieval | RETRIEVAL_FAILURE / irrelevant_retrieval | data |
| 5 | `LOW_TOP_K` | retrieval | RANKING_FAILURE / relevant_ranked_low | configuration |
| 6 | `RANKING_CORRUPTION` | ranking | RANKING_FAILURE / poor_reranking | algorithmic |
| 7 | `CHUNK_FRAGMENTATION` | context_assembly | CONTEXT_FAILURE / fragmented_context | configuration |
| 8 | `CONTEXT_TRUNCATION` | context_assembly | CONTEXT_FAILURE / insufficient_context | configuration |
| 9 | `CONFLICTING_EVIDENCE` | context_assembly | CONTEXT_FAILURE / conflicting_evidence | data |
| 10 | `STALE_DOCUMENT` | retrieval | RETRIEVAL_FAILURE / stale_documents | data |
| 11 | `GENERATION_CONSTRAINT_REMOVAL` | generation | GENERATION_FAILURE / hallucination | agentic |
| 12 | `TOOL_TIMEOUT` | retrieval | INFRASTRUCTURE_FAILURE / timeout | infrastructure |
| 13 | `RETRIEVAL_TIMEOUT` | infrastructure | INFRASTRUCTURE_FAILURE / timeout | infrastructure |
| 14 | `EMPTY_RETRIEVAL` | retrieval | RETRIEVAL_FAILURE / empty_retrieval | infrastructure |
| 15 | `EMBEDDING_FAILURE` | infrastructure | INFRASTRUCTURE_FAILURE / embedding_failure | infrastructure |
| 16 | `MODEL_FAILURE` | infrastructure | INFRASTRUCTURE_FAILURE / malformed_response | infrastructure |
| 17 | `AMBIGUOUS_QUERY` | query_understanding | QUERY_FAILURE / ambiguous_query | agentic |
| 18 | `INCOMPLETE_DECOMPOSITION` | query_decomposition | QUERY_FAILURE / incomplete_decomposition | agentic |

`GET /faults` returns the full catalogue as JSON, including default parameters and the
subtypes a fault may legitimately surface as (`targets`).

## 3. How injection works

The injector is the only component allowed to write `runtime.fault_flags` or
`runtime.retriever.inject`. It never edits thresholds, and it releases everything in a
`finally` block.

| Fault | Mechanism |
|---|---|
| `QUERY_CORRUPTION` | deterministic word dropout of the query (seeded) |
| `BAD_QUERY_REWRITE` | replaces the query with a drifting paraphrase (`drift ≥ 0.75`) or a lossy partial rewrite |
| `AMBIGUOUS_QUERY` | truncates the query to `keep_words` (default 2) |
| `INCOMPLETE_DECOMPOSITION` | keeps only the first `k` sub-questions |
| `RELEVANT_DOC_REMOVAL` | removes every chunk of the answer-bearing document from the index |
| `IRRELEVANT_DOC_INJECTION` | replaces the result list with chunks sharing no vocabulary with the query |
| `LOW_TOP_K` | caps `top_k` (default 1) |
| `RANKING_CORRUPTION` | swaps in `CorruptReranker` (`reverse` / `identity` / seeded noise) |
| `CHUNK_FRAGMENTATION` | re-chunks the corpus at `chunk_size=50, overlap=6` and rebuilds both indexes |
| `CONTEXT_TRUNCATION` | caps the assembled context (`max_context_tokens=40`, `max_chunks=1`) |
| `CONFLICTING_EVIDENCE` | inserts a decoy that negates the highest-scoring anchor sentence **into context assembly** (the declared target stage) |
| `STALE_DOCUMENT` | replaces the current document with a superseded version carrying changed figures and `stale`/`superseded` metadata |
| `GENERATION_CONSTRAINT_REMOVAL` | drops the grounding and citation constraints from the prompt |
| `EMPTY_RETRIEVAL` | forces zero results |
| `RETRIEVAL_TIMEOUT` / `TOOL_TIMEOUT` | forces a `RetrievalTimeout` |
| `EMBEDDING_FAILURE` | the dense branch raises; hybrid degrades to BM25 and records the failure |
| `MODEL_FAILURE` | the generation call fails and is recorded as a parse/infrastructure error |

`CONFLICTING_EVIDENCE` injects at context assembly rather than forcing a rank, because its
declared target stage *is* context assembly — retrieval and ranking stay untouched and the
decoy competes only through the ordinary reranker.

## 4. State isolation

`Runtime.baseline_chunks` holds a deep copy of the corpus captured **before** any fault is
activated. `restore_clean_state()` rebuilds both indexes from that snapshot, resets knobs,
clears the retriever's inject dict and error counters, and restores the chunking
configuration. Without this, faults would leak between requests — a real bug that the test
suite caught (`test_release_restores_the_runtime`).

## 5. Using the injector

```python
runtime = orchestrator.build_runtime()
injector = FaultInjector(runtime)

fault = injector.build("RANKING_CORRUPTION", {"mode": "reverse"}, experiment_id="exp_1")
injector.activate(fault, query)
try:
    ...
finally:
    injector.release()
```

or in one call:

```python
outcome = orchestrator.answer(question, faults=["RANKING_CORRUPTION"],
                              fault_parameters={"mode": "reverse"})
```

HTTP:

```bash
curl -X POST localhost:8000/faults/inject \
  -H 'content-type: application/json' \
  -d '{"fault_type":"RANKING_CORRUPTION","query":"..."}'
```

## 6. Scoring

`evaluation/metrics.py::diagnosis_correct()` returns three booleans — strict subtype,
failure class, acceptable — and all three are reported. A fault that legitimately surfaces
as a neighbouring subtype (declared in its `targets`) counts as acceptable but **not** as a
strict hit; neither number is presented as the other.

## 7. Limitations

* Detection and diagnosis are scored on a corpus of 10 short policy documents. The fault
  catalogue is tuned for that setting; a larger corpus may need new rules.
* Some faults are only observable with ground truth (`RELEVANT_DOC_REMOVAL`). This is
  reported rather than hidden.
* Faults are injected one category at a time in the sweep; interactions between faults are
  exercised individually by the demo scenarios, not statistically.