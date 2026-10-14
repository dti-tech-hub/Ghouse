# Experiments

Source: `agentic_rag_recovery/evaluation/runner.py`, configs in `experiments/configs/`.

## 1. What an experiment runs

```
BASELINE (mode 1: normal RAG)
   ↓
AGENTIC  (mode 2: diagnosis reported, no recovery)
   ↓
FAULT-INJECTED AGENTIC (mode 3: fault active → diagnose → recover → verify)
   ↓
EVALUATE + PERSIST
```

Every run writes its artefacts to SQLite, so any number in a report can be traced to a
request, and any request to the configuration that produced it.

## 2. Configuration format

```yaml
name: generation_failure
description: >
  The grounding constraint is removed from the generation prompt ...
seed: 42                       # required - every experiment is reproducible
repetitions: 1
dataset: demo
fault_type: GENERATION_CONSTRAINT_REMOVAL     # or:
fault_sweep: [FAULT_A, FAULT_B, ...]          # one experiment per fault type
question_ids: [q001, q006, q013]
category: retrieval                            # optional filter
limit: 10                                     # optional cap
modes:
  - mode1_normal_rag
  - mode2_agentic_no_recovery
  - mode3_agentic_with_recovery
retrieval: {top_k: 5}          # optional overrides
recovery: {max_recovery_attempts: 4}
```

Modes 1 and 2 never receive a fault: they are the *healthy* references.

## 3. Shipped configurations

| Config | Fault | Purpose |
|---|---|---|
| `baseline.yaml` | none | reference behaviour of all three modes on the healthy dataset |
| `retrieval_failure.yaml` | `BAD_QUERY_REWRITE` | vocabulary gap closed by `QUERY_REWRITE` |
| `ranking_failure.yaml` | `RANKING_CORRUPTION` | retrieval finds evidence, ranking loses it |
| `context_failure.yaml` | `CONTEXT_TRUNCATION` | context too small for the information need |
| `generation_failure.yaml` | `GENERATION_CONSTRAINT_REMOVAL` | unsupported claims repaired by regeneration |
| `infrastructure_failure.yaml` | `RETRIEVAL_TIMEOUT` | bounded retry and the alternate BM25 path |
| `abstention.yaml` | `RELEVANT_DOC_REMOVAL` | unrecoverable evidence loss |
| `full_recovery.yaml` | sweep of all 17 fault types, 3 repetitions | the table quoted in the report |

## 4. Running

```bash
# CLI
python scripts/run_experiments.py --config baseline
python scripts/run_experiments.py --config full_recovery --seed 42
python scripts/run_experiments.py --all

# Python
from agentic_rag_recovery.evaluation.runner import ExperimentRunner
result = ExperimentRunner(app).run(ExperimentRunner(app).load_config("ranking_failure"))
print(result.mode3.diagnosis_accuracy, result.mode3.recovery_success_rate)

# HTTP
curl -X POST localhost:8000/experiments/run -H 'content-type: application/json' \
  -d '{"name":"generation_failure","repetitions":1,"seed":42}'
```

## 5. Provenance

Every experiment row stores: `config_path`, `config_hash` (SHA-256 of the config),
`seed`, `repetitions`, `retrieval_config`, `recovery_config`, `model`,
`result_label`, `started_at`, `finished_at`, `status`.

Metric rows carry `label` and `seed`, so a report can state exactly which configuration and
provider produced a number.

## 6. Results

`ExperimentResult` contains `mode1`, `mode2`, `mode3` (`ExperimentMetrics`) and a
`comparison` block. Each `ExperimentMetrics` carries `label` — `"DEMO / MOCK RESULTS"` for
the deterministic provider — plus `per_query` records so every aggregate can be decomposed.

### Reading the numbers

| Figure | Meaning |
|---|---|
| `diagnosis_accuracy` | strict subtype matches / diagnosable failures |
| `recovery_success_rate` | verified recoveries / recovery attempts |
| `groundedness` | share of answer claims supported by the evidence |
| `added_latency_ms` | mean overhead of the recovery machinery |
| `cost_per_recovery` | total cost / verified recoveries |
| `abstention_rate` | share of requests that ended in an abstention |
| `avg_recovery_attempts` | mean attempts per request |
| `failure_count_by_class` | how many runs each fault contributed |

**Mode 1 and mode 2 are healthy references**, so their abstention rate should be ~0 and
their groundedness should match mode 3's *baseline* groundedness. A non-zero value there is
a bug, not a result.

### A fault sweep vs a single-fault experiment

`full_recovery.yaml` runs 17 fault types × 6 questions × 3 repetitions = 306 runs. The
aggregate mixes faults with very different difficulty; the per-family configs
(`generation_failure`, `context_failure`, …) isolate one failure mode and are the right
choice when you want to explain *why* a number is what it is.

## 7. Extending

1. Add `experiments/configs/<name>.yaml` with at least `seed` and `name`.
2. Run it; `tests/test_evaluation.py::test_every_config_has_a_seed` will fail if the seed is missing.
3. Compare with `compare_modes()` or the dashboard's Metrics page.

## 8. Limitations

* The demonstration corpus is 10 documents / ~49 chunks. Results are a feasibility study,
  not a benchmark.
* Fault sweeps use one question set; per-question variance is visible in `per_query` but is
  not statistically summarised.
* Real-LLM results require an API key or a local model; the shipped figures are labelled
  `DEMO / MOCK RESULTS` and must not be compared with them.