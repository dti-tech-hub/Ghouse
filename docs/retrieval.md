# Retrieval

Source: `agentic_rag_recovery/retrieval/`.

## 1. Pipeline

```
query
  ├─ vector search   (FAISS IndexFlatIP on L2-normalised vectors, or NumPy brute force)
  ├─ BM25 search     (Okapi BM25, k1=1.5, b=0.75)
  └─ hybrid          (Reciprocal Rank Fusion, alpha=0.55, rrf_k=60)
        ↓
   candidate pool (candidate_k = 12)
        ↓
   reranking (lexical cross-encoder emulation)
        ↓
   top-k (default 5)
        ↓
   RetrievalDiagnostics
```

The retriever returns the **candidate pool**, not just top-k. Ranking selects from that
pool, which is what makes a *ranking* failure distinguishable from a *retrieval* failure:
`pre_rank_top_overlap` (what the pool contained) versus `top_overlap` (what survived
ranking). `ranking_coverage_loss` is the difference.

## 2. Embeddings

The default backend is `HashingEmbedder`: a signed random projection of an IDF-weighted
bag of stemmed tokens into a 384-dimensional space.

* deterministic — identical corpus + seed produce identical vectors
* dependency-free — no download, no model, no network
* a *real* embedding, not a stub: IDF weighting means common words contribute little and
  discriminative terms dominate cosine similarity

`fit(corpus)` learns document frequencies; `HybridRetriever.rebuild()` **refits** the
embedder whenever the corpus changes (re-chunking, document removal, injected decoys),
because stale IDF statistics silently degrade the dense ranking.

Optional: `retrieval.embedding_backend: sentence_transformer` uses `sentence-transformers`.

## 3. Stemming

`textutil.stem` combines suffix stripping with a fixed-width prefix fold (`PREFIX_FOLD_LENGTH = 5`):

| word | stem | word | stem |
|---|---|---|---|
| cancellation | cance | cancel | cance |
| charge | charg | charged | charg |
| acknowledgement | ackno | acknowledged | ackno |
| deployment | deplo | deployments | deplo |
| policies | polic | policy | polic |

This matters because lexical coverage — and therefore every retrieval metric — depends on
morphological agreement. Without it "cancellation" and "cancel" are different tokens and a
perfectly retrievable chunk looks irrelevant.

## 4. Hybrid fusion

`hybrid_alpha` weights dense vs sparse contribution inside the RRF sum:

```
fused(c) = alpha / (rrf_k + rank_dense(c)) + (1 - alpha) / (rrf_k + rank_sparse(c))
```

RRF is used instead of raw score fusion because cosine, BM25 and RRF scores live on
incomparable scales. **Consequence:** raw similarity scores are *not* comparable across
methods, which is why every decision rule uses *lexical coverage*
(`top_overlap`, `mean_overlap`, `useful_ratio`) rather than `top_score`.

## 5. Diagnostics

`RetrievalDiagnostics` exposes:

| Field | Meaning |
|---|---|
| `top_k`, `candidate_pool`, `returned` | sizes |
| `top_score`, `mean_score`, `median_score`, `score_std`, `score_gap` | score shape |
| `score_distribution` | the raw list, for plotting |
| `top_overlap`, `mean_overlap` | lexical coverage of the question's content terms |
| `pre_rank_top_overlap`, `pre_rank_useful_ratio` | what the *pool* contained |
| `ranking_coverage_loss` | coverage that ranking threw away |
| `useful_ratio`, `useful_overlap` | fraction / threshold of chunks covering ≥ `useful_overlap` |
| `candidate_chunk_ids` | the pool (used for ground-truth ranking signals) |
| `duplicate_ratio`, `source_diversity`, `unique_documents` | redundancy |
| `rank_changes`, `rerank_score_margin` | reranking effect |
| `latency_ms`, `method`, `reranker_applied`, `notes` | operational |

`retrieval/diagnostics.py::diagnose_retrieval()` turns this into a flag list
(`empty_retrieval`, `weak_top_coverage`, `flat_score_distribution`, …) for the trace viewer.

## 6. Ground truth and evaluation

`retrieval_ground_truth()` identifies answer-bearing chunks lexically; the orchestrator
refines this with `_resolve_gold()`, which restricts the pool to the declared relevant
documents and selects the chunks that actually carry the answer keywords. Marking *every*
chunk of a document as gold would make recall structurally `< 1` for any `top_k`, so gold
is always evidence-bearing chunks only.

Metrics (`signals/metrics.py`): `recall_at_k`, `precision_at_k`, `hit_rate`, `mrr`, `ndcg_at_k`.

## 7. Degradation paths

| Condition | Behaviour |
|---|---|
| `service_available = False` | raises `RetrievalError` → `retrieval_service_unavailable` |
| `force_timeout` (fault) | raises `RetrievalTimeout` → `timeout` |
| embedding failure in hybrid | dense branch drops out, BM25 carries the search, the failure is still recorded |
| embedding failure in vector-only | raises `RetrievalError` → `embedding_failure` |
| FAISS absent | transparent fallback to the NumPy backend, same semantics |

## 8. Ingestion

| Format | Reader |
|---|---|
| `.txt`, `.md`, `.rst` | native, markdown-aware |
| `.pdf` | `pypdf` (page markers `[page:N]` are preserved) |
| `.docx` | `python-docx` (paragraphs + tables) |
| `.html` | tag/script stripping |
| `.json`, `.csv` | record flattening |

Chunking is a recursive character splitter (`ingestion/chunker.py`) with configurable size,
overlap and separator priority. Offsets are recomputed against the source text so every
chunk knows exactly where it came from. **Chunking configuration is a runtime knob** because
aggressive re-chunking is one of the injected failure scenarios.