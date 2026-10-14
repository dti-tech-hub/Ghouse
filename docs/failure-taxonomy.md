# Failure Taxonomy

The taxonomy is **declarative**: each failure subtype carries metadata that the detector,
the diagnoser, the planner and the dashboard all consume, so those components can never
disagree about what a failure means.

Source: `agentic_rag_recovery/taxonomy.py`.

## 1. Stages

| Stage | Description |
|---|---|
| `query_understanding` | normalisation, constraint extraction, ambiguity |
| `query_decomposition` | sub-question planning |
| `retrieval` | candidate generation (dense / sparse / hybrid) |
| `ranking` | ordering and cut-off selection |
| `context_assembly` | packing, expansion, conflict detection |
| `generation` | answer synthesis |
| `validation` | signal collection |
| `agent_control` | tool selection, repetition, budgets |
| `infrastructure` | embedding store, providers, timeouts |

## 2. Classes and subtypes

### QUERY_FAILURE — stage `query_understanding` / `query_decomposition`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `ambiguous_query` | medium | query is under-specified | QUERY_REWRITE → DECOMPOSITION_REPAIR → ABSTAIN |
| `incorrect_rewrite` | high | rewrite changed the meaning | QUERY_REWRITE → ABSTAIN |
| `incomplete_decomposition` | high | complex query split too coarsely | DECOMPOSITION_REPAIR → QUERY_REWRITE → ABSTAIN |
| `over_decomposition` | low | redundant sub-questions, duplicated evidence | DECOMPOSITION_REPAIR → ABSTAIN |

### RETRIEVAL_FAILURE — stage `retrieval`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `low_recall` | high | evidence exists but was not retrieved | QUERY_REWRITE → HYBRID_SEARCH → EXPAND_TOP_K → BROADEN_RETRIEVAL → ABSTAIN |
| `irrelevant_retrieval` | high | top-k is off-topic | QUERY_REWRITE → HYBRID_SEARCH → RERANK → ABSTAIN |
| `no_useful_evidence` | critical | nothing in the result set can answer | QUERY_REWRITE → HYBRID_SEARCH → EXPAND_TOP_K → BROADEN_RETRIEVAL → ABSTAIN |
| `empty_retrieval` | critical | zero candidates returned | BOUNDED_RETRY → ALTERNATE_PATH → ABSTAIN |
| `stale_documents` | medium | evidence is an outdated version | ALTERNATE_PATH → BOUNDED_RETRY → ABSTAIN |

### RANKING_FAILURE — stage `ranking`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `relevant_ranked_low` | high | relevant chunk pushed below the cut | RERANK → EXPAND_TOP_K → ABSTAIN |
| `irrelevant_ranked_high` | medium | irrelevant chunk occupies a top slot | RERANK → CHUNK_EXPANSION → ABSTAIN |
| `poor_reranking` | medium | reranker is worse than the base retriever | RERANK → ABSTAIN |

### CONTEXT_FAILURE — stage `context_assembly`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `fragmented_context` | high | answer sentences split across chunks | CHUNK_EXPANSION → CONTEXT_EXPANSION → ABSTAIN |
| `insufficient_context` | high | context too small for the information need | CONTEXT_EXPANSION → EXPAND_TOP_K → CHUNK_EXPANSION → ABSTAIN |
| `excessive_context` | medium | context dilutes the relevant evidence | RERANK → CONTEXT_EXPANSION → ABSTAIN |
| `conflicting_evidence` | high | context contradicts itself | RERANK → CHUNK_EXPANSION → ALTERNATE_PATH → ABSTAIN |
| `missing_evidence` | high | a constraint of the question is uncovered | CONTEXT_EXPANSION → EXPAND_TOP_K → HYBRID_SEARCH → ABSTAIN |

### GENERATION_FAILURE — stage `generation`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `hallucination` | critical | fabricated content | CONSTRAINED_REGENERATION → CONTEXT_EXPANSION → ABSTAIN |
| `unsupported_claim` | high | a claim is not entailed by the evidence | CONSTRAINED_REGENERATION → ABSTAIN |
| `incomplete_answer` | medium | part of the question is unanswered | CONSTRAINED_REGENERATION → CONTEXT_EXPANSION → ABSTAIN |
| `contradiction` | critical | answer contradicts the evidence | CONSTRAINED_REGENERATION → ABSTAIN |

### AGENT_CONTROL_FAILURE — stage `agent_control`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `wrong_tool_selection` | medium | tool cannot serve the subgoal | TOOL_REPAIR → ALTERNATE_PATH → ABSTAIN |
| `unnecessary_tool_call` | low | tool result unused | TOOL_REPAIR |
| `repeated_action` | high | an action that already failed was repeated | TOOL_REPAIR → ALTERNATE_PATH → ABSTAIN |
| `incorrect_workflow` | high | the plan cannot reach the goal | ALTERNATE_PATH → ABSTAIN |

### INFRASTRUCTURE_FAILURE — stage `infrastructure`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `retrieval_service_unavailable` | critical | index unreachable | ALTERNATE_PATH → BOUNDED_RETRY → ABSTAIN |
| `timeout` | high | stage exceeded its time budget | BOUNDED_RETRY → ALTERNATE_PATH → ABSTAIN |
| `malformed_response` | high | unparseable model output | BOUNDED_RETRY → ALTERNATE_PATH → ABSTAIN |
| `embedding_failure` | critical | embedding backend broken | ALTERNATE_PATH → BOUNDED_RETRY → ABSTAIN |
| `model_failure` | critical | provider error / empty completion | BOUNDED_RETRY → ALTERNATE_PATH → ABSTAIN |

### RESOURCE_FAILURE — stage `agent_control`

| Subtype | Severity | Meaning | Recovery ladder |
|---|---|---|---|
| `budget_exhausted` | high | attempt budget consumed | ABSTAIN |
| `latency_budget_exceeded` | high | latency budget consumed | ABSTAIN |
| `token_budget_exceeded` | high | token/cost budget consumed | ABSTAIN |

## 3. Invariants

1. **Every ladder terminates in `ABSTAIN`.** No failure class can produce an infinite escalation.
2. **`ABSTAIN` is never inferred silently.** It is either planned explicitly or forced by a
   budget, and the reason is stored on the `Abstention` record.
3. **The ground truth declared by a fault always exists in the taxonomy**, so an injected
   fault can never be unscoreable (`tests/test_faults.py` asserts this).
4. **Severity is a default, not a verdict.** The diagnoser may raise or lower it, but the
   taxonomy default is what the recovery ladder is keyed on.

## 4. Extending the taxonomy

1. Add a `FailureDefinition` with `_define(...)` in `taxonomy.py`.
2. Add the subtype to `_class_of`'s mapping.
3. Add a Tier-1 rule in `detection/rules.py`.
4. Add a fault (if the subtype should be injectable) in `faults/definitions.py` and a handler in
   `faults/injector.py`.
5. Extend `tests/test_faults.py::test_every_fault_declares_ground_truth` — it will fail until
   the mapping is consistent.