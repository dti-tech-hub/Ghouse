# Recovery

Source: `agentic_rag_recovery/recovery/`, `agentic_rag_recovery/validation/`.

## 1. Principle

Recovery is **targeted**, **bounded** and **verified**. The system never re-runs the whole
pipeline, never repeats an action, never exceeds a budget, and never assumes that a repair
worked.

## 2. Planner

`recovery/planner.py` turns a diagnosis into an ordered ladder:

1. take the taxonomy ladder for the diagnosed (class, subtype),
2. score each candidate action from the runtime signals,
3. drop actions that were **already attempted** (`BLOCK_REPEAT`),
4. drop actions that **cannot change anything** in the current configuration (`NO-OP`),
5. drop actions the budget forbids,
6. execute the first survivor; `ABSTAIN` is always last.

### Scoring

Scores encode which repair addresses the evidence that was actually observed, e.g.

```
RETRIEVAL_FAILURE        → QUERY_REWRITE 2.2, HYBRID_SEARCH 2.0, EXPAND_TOP_K 1.6, …
RANKING_FAILURE          → RERANK 2.3, EXPAND_TOP_K 1.7, …
CONTEXT_FAILURE          → CHUNK_EXPANSION 2.2, CONTEXT_EXPANSION 2.0, …
GENERATION_FAILURE       → CONSTRAINED_REGENERATION 2.5
INFRASTRUCTURE_FAILURE   → BOUNDED_RETRY 2.3, ALTERNATE_PATH 2.2 (2.8 if embedding failed)
```

plus conditional boosts (e.g. `fragmentation_ratio > 0.5` raises `CHUNK_EXPANSION` to 2.6)
and the diagnosis's own `recommended_recovery` (+2.0).

### No-op detection

| Action | Refused when |
|---|---|
| `HYBRID_SEARCH` | hybrid retrieval is already the active method |
| `RERANK` | a healthy reranker is already applied |
| `QUERY_REWRITE` | the query has already been rewritten twice |
| `EXPAND_TOP_K` | `top_k` already reached the ceiling |

## 3. Strategies

| Action | What it does | Re-runs from | Reuses |
|---|---|---|---|
| `QUERY_REWRITE` | expands the query with alternative surface forms | retrieval | understanding, decomposition |
| `HYBRID_SEARCH` | switches to RRF fusion of dense + BM25 | retrieval | understanding, decomposition |
| `EXPAND_TOP_K` | widens the cut-off (bounded by `max(top_k*3, candidate_k)`) | retrieval | understanding, decomposition |
| `BROADEN_RETRIEVAL` | wider pool + hybrid, thresholds untouched | retrieval | understanding, decomposition |
| `RERANK` | re-orders the **cached** candidate pool with a healthy cross-encoder | ranking | retrieval |
| `CHUNK_EXPANSION` | pulls the immediate neighbour chunks of each retrieved chunk | context assembly | retrieval, ranking |
| `CONTEXT_EXPANSION` | lifts an artificial truncation, adds neighbours | context assembly | retrieval, ranking |
| `CONSTRAINED_REGENERATION` | re-generates with grounding + citation constraints restored | generation | the whole context |
| `DECOMPOSITION_REPAIR` | re-plans sub-questions without the injected truncation | decomposition | understanding |
| `BOUNDED_RETRY` | one retry after clearing transient conditions | earliest failed stage | earlier stages |
| `ALTERNATE_PATH` | switches to an independent backend (BM25 needs no embedding service) | earliest failed stage | earlier stages |
| `TOOL_REPAIR` | removes the misbehaving tool from the plan | ranking or retrieval | earlier stages |
| `ABSTAIN` | terminal | — | — |

Every strategy returns a `StrategyOutcome` with the earliest stage to re-run, a rationale,
and the list of reused stages. A strategy that cannot help returns `applied=False` with a
reason; the executor records it as `SKIPPED` and moves on rather than burning a blind retry.

## 4. Budget

`recovery/budget.py` enforces every hard limit:

| Limit | Default | Checked by |
|---|---|---|
| `max_recovery_attempts` | 4 | `BudgetTracker.preflight()` |
| `max_total_latency_ms` | 45 000 | `preflight()` |
| `max_llm_calls` | 14 | `preflight()` |
| `max_token_budget` | 24 000 | `preflight()` |
| `max_repeated_actions` | 1 | `can_attempt()` |

`preflight()` returns the first exhausted resource, so the executor can stop with a precise
reason rather than a generic failure.

## 5. Verification

`validation/verifier.py` re-collects signals after every attempt and applies
**class-specific acceptance criteria**. Nothing is assumed.

| Failure class | Criteria (all must pass) |
|---|---|
| RETRIEVAL | retrieval non-empty; useful ratio ≥ 0.40; question coverage ≥ 0.34; coverage improved; duplicates bounded; recall ≥ max(0.5, previous) |
| RANKING | no relevant chunk below the cut; useful ratio ≥ 0.40; regression rate ≤ 0.34; MRR not reduced |
| CONTEXT | coverage ≥ 0.50; all constraints covered; tokens within `[min,max]`; no conflicts; coverage not degraded |
| GENERATION | groundedness ≥ 0.62; no unsupported claims; citation coverage ≥ 0.30; groundedness not degraded |
| QUERY | rewrite drift ≤ 0.62; rewrite actually changed the query; retrieval improved |
| AGENT_CONTROL | no repeats; no failed actions |
| INFRASTRUCTURE | no new errors; service available; no timeouts; no embedding errors |

Retrieval criteria deliberately use **lexical coverage**, not `top_score`: raw scores are not
comparable across backends, so comparing them across a recovery would be meaningless.

## 6. Loop

```
for attempt in 1..MAX_RECOVERY_ATTEMPTS:
    plan    = planner(diagnosis, attempted, budget, active_config)
    if plan.must_abstain: break
    action  = next executable action            # never repeats, never a no-op
    apply(action)                               # targeted re-run
    verify()                                    # class-specific criteria
    if verified: STOP (SUCCESS)
if not recovered: ABSTAIN
```

Terminal reasons recorded on `RecoveryOutcomeSummary.stop_reason`:
`"no executable recovery left"`, `"only ABSTAIN remains - a budget or repeat limit was reached"`,
`"planner selected terminal ABSTAIN"`, `"budget exhausted: <resource>"`,
`"maximum recovery attempts exhausted"`.

## 7. Abstention

`Abstention` records:

```python
Abstention(
    abstained=True,
    reason=...,
    failure_class=..., stage=..., subtype=...,
    attempted_recoveries=[...], attempts_used=..., max_attempts=...,
    budget_consumed={latency_used_ms, tokens_used, llm_calls_used, attempts_used, ...},
    message="I could not obtain sufficient reliable evidence to answer this question.",
)
```

The generation stage is replaced by the abstention message, so no unsupported text ever
reaches the user, and the record is persisted to the `abstentions` table.

## 8. Typical ladders

**Retrieval failure** → `QUERY_REWRITE` → `HYBRID_SEARCH`(no-op, skipped) →
`EXPAND_TOP_K` → `BROADEN_RETRIEVAL` → `ABSTAIN`

**Ranking failure** → `RERANK` (retrieval reused) → `EXPAND_TOP_K` → `ABSTAIN`

**Generation failure** → `CONSTRAINED_REGENERATION` (context reused) → `ABSTAIN`

**Infrastructure failure** → `BOUNDED_RETRY` → `ALTERNATE_PATH` (BM25) → `ABSTAIN`