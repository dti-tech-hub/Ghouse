# Measured results (auto-generated)

Label: **DEMO / MOCK RESULTS**. Source: 432 executed runs (18 faults x 24 questions).

## Overall

| metric | value |
|---|---|
| runs | 432 |
| detected | 432 |
| detection_rate_pct | 100.0 |
| top1_correct | 265 |
| top1_accuracy_pct | 61.34 |
| topk_correct | 273 |
| topk_accuracy_pct | 63.19 |
| runs_with_recovery_attempts | 342 |
| recovered | 192 |
| recovery_success_pct_of_attempted | 56.14 |
| verified_recovery | 192 |
| abstained | 172 |
| abstention_rate_pct | 39.81 |
| latency_ms (n=432) | p50 91.147 / mean 132.489 / p95 394.174 / max 882.511 |
| diagnosis_latency_ms (n=432) | p50 0.253 / mean 0.573 / p95 2.433 / max 4.665 |
| added_latency_ms (n=342) | p50 72.287 / mean 107.33 / p95 331.519 / max 803.235 |

## Per fault

| fault | stage | expected subtype | det. % | top-1 % | top-k % | recovered % | abstained % | p50 ms |
|---|---|---|---|---|---|---|---|---|
| `AMBIGUOUS_QUERY` | query_understanding | `ambiguous_query` | 100.0 | 91.67 | 91.67 | 83.33 | 16.67 | 161.25 |
| `BAD_QUERY_REWRITE` | query_understanding | `incorrect_rewrite` | 100.0 | 95.83 | 95.83 | 66.67 | 33.33 | 102.85 |
| `CHUNK_FRAGMENTATION` | context_assembly | `fragmented_context` | 100.0 | 37.5 | 70.83 | 66.67 | 29.17 | 69.6 |
| `CONFLICTING_EVIDENCE` | context_assembly | `conflicting_evidence` | 100.0 | 66.67 | 66.67 | 0.0 | 70.83 | 143.41 |
| `CONTEXT_TRUNCATION` | context_assembly | `insufficient_context` | 100.0 | 95.83 | 95.83 | 95.83 | 4.17 | 120.98 |
| `EMBEDDING_FAILURE` | infrastructure | `embedding_failure` | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 174.03 |
| `EMPTY_RETRIEVAL` | retrieval | `empty_retrieval` | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 4.13 |
| `GENERATION_CONSTRAINT_REMOVAL` | generation | `hallucination` | 100.0 | 95.83 | 95.83 | 91.67 | 8.33 | 92.45 |
| `INCOMPLETE_DECOMPOSITION` | query_decomposition | `incomplete_decomposition` | 100.0 | 4.17 | 4.17 | 0.0 | 8.33 | 70.93 |
| `IRRELEVANT_DOC_INJECTION` | retrieval | `irrelevant_retrieval` | 100.0 | 0.0 | 0.0 | 0.0 | 100.0 | 272.56 |
| `LOW_TOP_K` | retrieval | `relevant_ranked_low` | 100.0 | 0.0 | 0.0 | 20.83 | 4.17 | 36.02 |
| `MODEL_FAILURE` | infrastructure | `malformed_response` | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 9.84 |
| `QUERY_CORRUPTION` | query_understanding | `low_recall` | 100.0 | 4.17 | 4.17 | 16.67 | 33.33 | 47.53 |
| `RANKING_CORRUPTION` | ranking | `poor_reranking` | 100.0 | 25.0 | 25.0 | 0.0 | 87.5 | 48.85 |
| `RELEVANT_DOC_REMOVAL` | retrieval | `no_useful_evidence` | 100.0 | 12.5 | 12.5 | 20.83 | 58.33 | 166.73 |
| `RETRIEVAL_TIMEOUT` | infrastructure | `timeout` | 100.0 | 100.0 | 100.0 | 95.83 | 4.17 | 49.12 |
| `STALE_DOCUMENT` | retrieval | `stale_documents` | 100.0 | 75.0 | 75.0 | 45.83 | 54.17 | 215.43 |
| `TOOL_TIMEOUT` | retrieval | `timeout` | 100.0 | 100.0 | 100.0 | 95.83 | 4.17 | 44.79 |

## Per failure class

| failure class | n | det. % | top-1 % | top-k % | recovered % | abstained % | p50 ms |
|---|---|---|---|---|---|---|---|
| `CONTEXT_FAILURE` | 72 | 100.0 | 66.67 | 77.78 | 54.17 | 34.72 | 117.34 |
| `GENERATION_FAILURE` | 24 | 100.0 | 95.83 | 95.83 | 91.67 | 8.33 | 92.45 |
| `INFRASTRUCTURE_FAILURE` | 96 | 100.0 | 100.0 | 100.0 | 72.92 | 27.08 | 46.82 |
| `QUERY_FAILURE` | 72 | 100.0 | 63.89 | 63.89 | 50.0 | 19.44 | 117.85 |
| `RANKING_FAILURE` | 48 | 100.0 | 12.5 | 12.5 | 10.42 | 45.83 | 45.65 |
| `RETRIEVAL_FAILURE` | 120 | 100.0 | 38.33 | 38.33 | 16.67 | 69.17 | 148.89 |

## Recovery actions

| action | attempts | success | verified | success % |
|---|---|---|---|---|
| `BOUNDED_RETRY` | 133 | 70 | 70 | 52.63 |
| `QUERY_REWRITE` | 98 | 39 | 39 | 39.8 |
| `CHUNK_EXPANSION` | 79 | 31 | 31 | 39.24 |
| `ALTERNATE_PATH` | 60 | 5 | 5 | 8.33 |
| `EXPAND_TOP_K` | 46 | 1 | 1 | 2.17 |
| `BROADEN_RETRIEVAL` | 41 | 0 | 0 | 0.0 |
| `CONTEXT_EXPANSION` | 37 | 23 | 23 | 62.16 |
| `CONSTRAINED_REGENERATION` | 26 | 22 | 22 | 84.62 |
| `DECOMPOSITION_REPAIR` | 5 | 1 | 1 | 20.0 |

## Outcomes

| outcome | runs |
|---|---|
| abstained | 172 |
| recovered | 192 |
| success | 68 |
