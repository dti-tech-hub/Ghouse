# Metrics

Source: `agentic_rag_recovery/signals/metrics.py`, `agentic_rag_recovery/evaluation/`.

## 1. Result integrity rules

1. **Every number is computed from an executed run.** Nothing is hard-coded or hand-written.
2. **Mock-provider results are labelled.** Every metrics row carries
   `labels.results = "DEMO / MOCK RESULTS"` when the deterministic provider was used.
   Real-LLM runs carry their own label.
3. **Three diagnosis accuracies are reported side by side** and never substituted.
4. **External evaluation libraries are optional.** RAGAS / DeepEval are adapters
   (`evaluation/external.py`); the reported figures always come from the deterministic
   custom metrics so results are reproducible offline.

## 2. The six headline metrics

| Metric | Definition | Where computed |
|---|---|---|
| **Diagnosis accuracy** | correct diagnoses / diagnosable failures | `evaluation/metrics.py::aggregate` |
| **Recovery success rate** | verified recoveries / recovery attempts | `aggregate` |
| **Groundedness** | share of answer claims entailed by the retrieved evidence | `signals/metrics.py::groundedness_score` |
| **Added latency** | mean(total latency − baseline latency) over runs | `aggregate` |
| **Cost per recovery** | total cost / verified recoveries | `aggregate` |
| **Abstention rate** | abstentions / total requests | `aggregate` |

*Diagnosable failure* = a run with at least one injected fault **and** a positive
`failure_detected` diagnosis. A fault the system could not detect is counted as a miss,
never quietly excluded.

## 3. Supporting metrics

| Metric | Definition |
|---|---|
| `answer_correctness` | token-F1 against the reference answer (SQuAD-style proxy) |
| `baseline_answer_correctness` | the same on the pre-recovery answer |
| `retrieval_recall` / `_precision` / `mrr` / `ndcg@k` | standard IR metrics against gold chunks |
| `context_coverage` | question coverage by the assembled context |
| `citation_coverage` | share of retrieved passages actually cited |
| `avg_recovery_attempts` | mean attempts per request |
| `token_usage`, `total_cost` | aggregate accounting |
| `rule_diagnoses` / `llm_diagnoses` / `fallback_diagnoses` | tier mix |
| `failure_count_by_class` / `by_stage` / severity distribution | breakdowns |

## 4. Custom deterministic metrics

These are the project's own implementations; no external library is required.

### Groundedness

```
claims       = sentences of the answer, citations stripped
support(c)    = max over context sentences of
               0.55·token overlap + 0.30·character-4-gram overlap
               + 0.40·(numbers agree) + 0.25·(named entities agree)
groundedness = 1 − (claims with support < 0.34) / claims
```

Numeric and entity agreement are included because a correct figure or proper noun is
strong evidence of support even when the phrasing differs. Claims with a single content word
are kept (dropping them would inflate groundedness).

### Citation coverage

`len(set(cited markers) ∩ valid markers) / len(valid markers)`, where valid markers are the
context items actually assembled.

### Context coverage

Mean of: coverage of the question's content terms, of every explicit constraint, and of
every decomposed sub-question. Constraints are only quantitative or temporal expressions —
a bare topic word such as "threshold" is not a constraint, because treating it as one
produces false "missing evidence" diagnoses.

### Conflict score

A pair of context sentences is flagged only when they demonstrably make claims about the
*same* statement:

* shared content tokens ≥ 3, Jaccard ≥ 0.34, then
* **negation rule**: masking the negations makes the sentences ≥ 0.85 identical and exactly
  one contains a negation applied to a shared term, or
* **numeric rule**: exactly one quantity slot differs in value, and the sentences do not
  differ by a negation (which would change *which population* is described)

Score = `min(1, 0.5 × conflicts)`: one verified contradiction is already a serious defect.

This design deliberately avoids two false-positive families that a naive implementation
produces constantly:

* "Vendors that process customer data are audited every 12 months" vs
  "Vendors that do **not** process customer data are audited every 24 months" → contrast, not contradiction
* "Severity 1 … within 5 minutes" vs "Severity 2 … within 15 minutes" → two different
  facts, not a contradiction (two quantity slots differ)

### Fragmentation

Sentences are taken from the **whole documents**, not from chunk text — otherwise the split we
are trying to detect is invisible. A sentence counts as fragmented when no single chunk in
the corpus can hold ≥ 70 % of its content tokens.

### Duplicate ratio / source diversity

`1 − unique/len` over the result set; and the number of distinct source files.

## 5. Baselines

Three modes are compared on identical questions and configurations:

| Mode | Description |
|---|---|
| `mode1_normal_rag` | retrieval + generation, no agentic control, no recovery |
| `mode2_agentic_no_recovery` | full agentic pipeline, diagnosis reported, **no** action taken |
| `mode3_agentic_with_recovery` | MONITOR → DETECT → DIAGNOSE → RECOVER → VERIFY → ANSWER/ABSTAIN |

`compare_modes()` reports each mode's block plus `mode3_minus_mode1` deltas for groundedness,
answer correctness, abstention rate and added latency. Only deltas from executed runs appear.

## 6. Reporting template

When writing results, always state:

* the provider (`mock` or a real model) and the label applied,
* the dataset and question ids,
* the seed and repetition count,
* the configuration hash stored on the experiment row.

This is why `experiments` stores `config_hash`, `seed`, `repetitions`, `retrieval_config`,
`recovery_config` and `result_label`.

## 7. Optional external evaluators

```python
from agentic_rag_recovery.evaluation.external import evaluate_with_ragas, availability
availability()   # {'ragas': False, 'deepeval': False}
```

When installed, these provide an independent cross-check of faithfulness and answer
relevancy. They are **not** the source of the reported numbers, because their results vary
with model version and prompt formatting.