# Diagnosis

Source: `agentic_rag_recovery/detection/`, `agentic_rag_recovery/diagnosis/`.

## 1. Two tiers

```
                       ┌───────────────────────────┐
runtime signals  ───►  │ TIER 1: rule engine      │
                       │ 33 deterministic rules    │
                       └────────────┬──────────────┘
                                    │ confidence ≥ rule_confidence_threshold (0.55)
                    ┌───────────────┴───────────────┐
                 yes│                               │no
                    ▼                               ▼
          ┌───────────────────┐        ┌───────────────────────────┐
          │ diagnosis (rule)  │        │ TIER 2: LLM diagnoser     │
          └───────────────────┘        │ strict JSON schema         │
                                       │ + validation + repair      │
                                       └────────────┬──────────────┘
                                                    │ invalid / unusable
                                        ┌───────────┴────────────┐
                                     ok  ▼                        ▼  fallback
                              diagnosis (llm)          degraded to Tier-1 evidence
                                                          (tier=fallback, confidence × 0.7)
```

The tier that produced each diagnosis is recorded (`Diagnosis.tier`), and the engine keeps
counters (`rule`, `llm`, `fallback`, `escalated`) surfaced through `/system` and the
dashboard.

## 2. Tier 1 — rule engine

Rules live in `detection/rules.py` as pure functions of `(SignalBundle, Settings)`. They are
evaluated in a fixed order — infrastructure first, then budgets, agent control, generation,
context, ranking, retrieval, query — because a downstream symptom must not out-vote the
stage that actually broke.

Every rule returns a `RuleMatch` carrying:

* `rule_id` (e.g. `R_RET_004`), `subtype`, `failure_class`, `stage`
* `confidence` (how strongly the evidence fires)
* `severity` (taxonomy default, optionally overridden)
* `evidence` — human-readable strings that quote the **actual signal values**
* `signals` — the raw numbers used

### Root-cause preference

`RuleDetector.best()` does not simply return the highest confidence. It compares the
earliest failing stage against the top match and requires the downstream match to exceed it
by `DOWNSTREAM_OVERRIDE_MARGIN` (0.12). Rationale: when retrieval returns nothing, "the
answer is unsupported" is a *consequence*, and reporting the consequence would send the
recovery planner to fix the wrong stage.

### Threshold semantics

Decision rules use **lexical coverage** rather than raw similarity scores, because cosine,
BM25 and RRF scores are not comparable across backends:

| Threshold | Default | Meaning |
|---|---|---|
| `min_top_overlap` | 0.34 | the best retrieved chunk must cover ≥34 % of the question's content terms |
| `min_mean_overlap` | 0.12 | mean coverage across the result set |
| `useful_overlap` | 0.20 | a chunk covering ≥20 % of the query terms counts as "useful" |
| `min_useful_chunk_ratio` | 0.40 | share of the result set that must be useful |
| `min_groundedness` | 0.62 | share of answer claims entailed by the evidence |
| `min_context_coverage` | 0.50 | question coverage by the assembled context |
| `max_conflict_score` | 0.35 | one verified contradiction scores 0.5 |
| `max_fragmentation_ratio` | 0.50 | share of relevant sentences no single chunk can hold |

These were **calibrated on the demonstration corpus**, not guessed: `min_top_overlap = 0.34`
sits below the lowest healthy top-coverage observed (0.38) and above the failing cases.

## 3. Tier 2 — LLM diagnoser

`diagnosis/llm_diagnoser.py` sends a prompt containing:

* the flat numeric signal view (scalars only),
* the sectioned signal view,
* the full taxonomy as JSON,
* the shortlist of rules that fired (even below threshold),
* **the configured thresholds**, so the model uses the project's operating points rather
  than inventing its own.

It must return exactly:

```json
{
  "failure_detected": true,
  "failure_class": "RETRIEVAL_FAILURE",
  "stage": "retrieval",
  "subtype": "low_recall",
  "severity": "high",
  "confidence": 0.91,
  "evidence": ["candidate pool only reaches 0.20 question coverage"],
  "recommended_recovery": "QUERY_REWRITE"
}
```

### Validation and repair (`diagnosis/schema.py`)

Free-form output is **never** trusted. `validate_diagnosis_payload()` performs, in order:

1. key aliasing (`class` → `failure_class`, `fix` → `recommended_recovery`, …)
2. type coercion (confidence `"85"` → `0.85`, evidence string → list)
3. missing-field defaulting (when `repair=True`)
4. taxonomy reconciliation — an unknown subtype is replaced by the closest one for the
   declared class; an inconsistent `failure_class` is corrected *from the taxonomy*
5. stage and recovery-action validation against the enums and the subtype's ladder
6. Pydantic validation

Confidence is reduced by `0.08 × errors + 0.03 × repairs`, and **every** violation is stored
on `Diagnosis.validation_errors` / `Diagnosis.repairs`. If the output is unusable the
diagnoser returns `tier=fallback` with `failure_detected=false` — it never invents a cause.

## 4. Diagnosis record

```python
Diagnosis(
    diagnosis_id, run_id,
    failure_detected, failure_class, stage, subtype, severity, confidence,
    evidence=[...],                 # cites the signal values that justify it
    recommended_recovery, recovery_ladder,
    tier, rule_id, rule_matches,
    signals_used,                    # the exact numbers the decision used
    latency_ms, raw_llm_output, validation_errors, repairs,
)
```

## 5. Honest measurement

Diagnosis accuracy is scored against the **ground truth declared by the fault injector**,
never against what the system produced. The injector writes its expectation to SQLite before
the run starts, and the diagnoser never receives it — `tests/test_faults.py` asserts that no
`expected_subtype` string appears anywhere in the signals or diagnosis.

Three accuracies are reported side by side (never substituted for one another):

| Metric | Definition |
|---|---|
| **strict** | predicted subtype == declared expected subtype |
| **class** | predicted failure class == declared expected class |
| **acceptable** | either of the above |

`tests/test_evaluation.py::test_diagnosis_correct_strict_and_class` pins this behaviour.

## 6. Known limitations

* `RELEVANT_DOC_REMOVAL` is often **not** detectable without ground truth: removing the
  answer-bearing document can leave lexically plausible evidence behind, and no runtime
  signal distinguishes it from a correct answer. This is a genuine finding, reported in
  `limitations.md`, not a bug to be hidden.
* Detection quality depends on the calibration of the thresholds; a different corpus may
  require recalibration (`configs/default.yaml` makes this a configuration change).
* The Tier-2 diagnoser only runs when Tier 1 is inconclusive, so its accuracy is measured
  on a biased (harder) subset.