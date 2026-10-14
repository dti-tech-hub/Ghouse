# Limitations

Honest assessment of what this system does **not** do. Read this before quoting any number.

## 1. Evaluation scale

* The demonstration corpus is **10 short policy documents / ~49 chunks** with **24
  questions**. Every published figure is a feasibility result, not a benchmark.
* Fault sweeps use a fixed question set. Per-question variance is visible in
  `per_query`, but it is not summarised statistically (no confidence intervals).
* `MockProvider` is a deterministic rule-based generator, **not** a language model. It can
  only extract sentences from the supplied context, so it cannot exhibit the generative
  failure modes a real model shows (e.g. partially-grounded multi-step reasoning).
  All figures produced with it are labelled `DEMO / MOCK RESULTS` and must **not** be
  compared with real-LLM results.
* The hashed TF-IDF embedder is a bag-of-words projector. It has no synonymy, no
  compositionality and no world knowledge. This makes the offline demo reproducible, but a
  real deployment should use `sentence-transformers`.

## 2. Detection gaps found by our own experiments

| Fault | Observation |
|---|---|
| `RELEVANT_DOC_REMOVAL` | Frequently **undetectable without ground truth**. Removing the answer-bearing document can leave lexically plausible evidence behind; no runtime signal distinguishes it from a correct answer. Reported as a miss, not hidden. |
| `LOW_TOP_K` | Only observable when the answer is not already at rank 1. With `top_k=1` on a question whose answer is rank 1 there is genuinely nothing to detect. |
| `INCOMPLETE_DECOMPOSITION` | Only meaningful for questions the planner classifies as complex; a single-clause question cannot be under-decomposed. |

These are properties of the problem, not bugs. They are also a concrete argument for the
project: *a monitor that only works with ground truth is not a monitor.*

## 3. Grounding metric

The groundedness metric is **lexical and numeric**, not a trained entailment model. It can
be fooled by:

* an answer that copies evidence sentences verbatim while answering a subtly different
  question (lexical support is high, semantic support is not),
* paraphrase that replaces every content word with a synonym.

A trained NLI cross-checker (or DeepEval's `GroundednessMetric`, available as an optional
adapter) would be stronger, but it is not reproducible offline and its results drift with
model versions, so it is deliberately not the source of the reported numbers.

## 4. Diagnosis

* **Thresholds are calibrated on one corpus.** `min_top_overlap = 0.34` was fitted to the
  demonstration set; another corpus needs recalibration. The thresholds are configuration,
  not constants, but no automatic calibration is implemented.
* **Tier-2 is measured on a biased subset.** The LLM diagnoser only runs when Tier 1 is
  inconclusive, which is by construction the hard case. Its accuracy is therefore not
  comparable with Tier-1 accuracy across the same runs.
* **The detector is stage-preferring.** When several rules fire it prefers the earliest
  failing stage unless a downstream rule beats it by 0.12 confidence. This is a deliberate
  root-cause heuristic; it can choose the earlier stage when both explanations are
  genuinely plausible.
* No **online** learning: thresholds, rules and the taxonomy are fixed at configuration time.

## 5. Recovery

* Recovery is bounded and verified, but the verification criteria are **thresholds**, so a
  recovery can be judged "successful" while the answer is still mediocre (for example when
  the retriever promotes a partially relevant chunk). The trace always shows the before/after
  numbers so a reader can judge for themselves.
* `QUERY_REWRITE` depends on the provider producing a useful expansion. With a real LLM
  that behaviour is stochastic; with the mock it is deterministic and lexicon-driven.
* Recoveries that "repair" the *symptom* rather than the cause are possible in principle
  (for example `BROADEN_RETRIEVAL` after `RELEVANT_DOC_REMOVAL`). The bounded loop still
  terminates and abstains, but it spends attempts first.
* The planner's action scoring is a hand-designed heuristic, not a learned policy.

## 6. Engineering

* **SQLite, single process.** WAL mode handles the demo load; it is not a multi-tenant
  store, and there is no authentication, rate limiting or per-user isolation.
* **The corpus is trusted.** There is no adversarial-document defence: a document containing
  instructions addressed to the model is ingested like any other text. Prompt-injection
  defence is out of scope but is a genuine gap for deployment.
* **Path traversal on ingestion** is limited by server-side path resolution and a 404 for
  missing paths, but the API accepts any server-side path that exists — it is an operator
  tool, not a public endpoint.
* **No streaming**, no cancellation, no retries across process restarts.
* The LangGraph runner is optional and exercised only when LangGraph is installed; the
  sequential runner is the reference implementation.

## 7. Research framing

This is an **implementation and feasibility study**. It does not claim to reproduce, and
does not claim novelty over, prior work on RAG failure analysis, evidence-gated answering
or diagnosis-guided repair. What it contributes is a single, runnable system that puts
monitoring, a declared failure taxonomy, two-tier diagnosis, targeted bounded recovery,
verification and honest abstention into one loop, together with a fault-injection
instrumentation that makes its own behaviour measurable.

See `README.md` → *Research positioning* for the distinction between existing concepts,
this implementation, and the experimental findings.