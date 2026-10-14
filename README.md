# Agentic RAG Failure Diagnosis & Recovery System

**Make an agentic RAG pipeline notice its own failures, say which stage broke, repair only
that stage, prove the repair worked, and refuse to answer when it cannot.**

A complete, locally-runnable research system: signal collection, typed failure taxonomy,
two-tier diagnosis, targeted & budgeted recovery, verification, abstention, a fault-injection
harness, an experiment suite, a REST API and a 12-page dashboard.

> **Runs offline.** No API key, no network, no cloud services. A deterministic mock provider
> stands in for the LLM, so every demo and experiment is reproducible bit-for-bit. Results
> produced this way are labelled `DEMO / MOCK RESULTS` and are never presented as real-model
> measurements.

---

## Table of contents

1. [Why](#why)
2. [Quick start](#quick-start)
3. [What it does](#what-it-does)
4. [Measured results](#measured-results)
5. [Using the dashboard & API](#using-the-dashboard--api)
6. [Commands](#commands)
7. [Project layout](#project-layout)
8. [Configuration](#configuration)
9. [Design rules](#design-rules)
10. [Paper & data](#paper--data)

---

## Why

An agentic RAG failure is invisible. The query is mis-rewritten, the reranker inverts the
ordering, the context window truncates the one answer-bearing sentence, the generator drops
its grounding constraint — and the system returns a **fluent, confident, wrong** answer.
Dashboards report latency and token counts. Nobody reports *which stage broke*.

Three things are hard, and this project is built around them:

| | Problem | How the system answers it |
|---|---|---|
| **P1** | Failures are staged and heterogeneous | A typed taxonomy: **33 subtypes**, **9 stages**, **8 failure classes** |
| **P2** | A detector must never see the answer key | Ground truth lives only in the injector; it is used to *score* a run after it finishes |
| **P3** | "It answered again" is not a repair | Every recovery is re-measured by an independent verifier, or it is rejected |

---

## Quick start

**Requirements:** Python **3.10+**. Nothing else — no Docker, no database server, no API key.

```bash
# 1. get the code and install dependencies
git clone https://github.com/dti-tech-hub/Ghouse.git
cd Ghouse
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# 2. build the demo knowledge base (10 documents -> 49 chunks)
python scripts/init_db.py --reset
python scripts/ingest.py --force

# 3. see it work, end to end
python run_demo.py
```

`run_demo.py` walks ten scenarios with full traces: a healthy pipeline, five injected failure
families, diagnosis, targeted recovery, a verified recovery, a rejected recovery, and an
honest abstention.

### Launch the dashboard

```bash
python scripts/serve.py
```

Then open:

| | |
|---|---|
| **Dashboard** | **http://127.0.0.1:8501** |
| API | http://127.0.0.1:8012 |
| API docs (Swagger) | http://127.0.0.1:8012/docs |

`scripts/serve.py` finds free ports automatically (if 8012/8501 are busy it picks another and prints
the URLs it used), waits for both services to report healthy, and writes PIDs to
`storage/logs/serve.pids`.

```bash
python scripts/serve.py --status   # show PIDs and health
python scripts/serve.py --stop     # stop both
```

### Run it manually instead

```bash
uvicorn backend.main:app --reload  # terminal 1 - API on :8000
streamlit run dashboard/app.py     # terminal 2 - dashboard on :8501
```

---

## What it does

```
   user query
       |
  +----v-----+  +-------------+  +----------+  +-------+  +-----------+  +-----------+
  |  query   |->| query       |->| RETRIEVAL|->| RANKING|->|  CONTEXT  |->| GENERATION|
  |  intake  |  | understd +  |  |  hybrid  |  | rerank |  |  assembly |  |           |
  +---------- +  decomposit. |  +----------+  +-------+  +-----------+  +-----+-----+
                              |                                          |           |
                              +------------> MONITOR <--------------------+           |
                                               |                               |
                            collect per-stage signals (overlap, coverage,          |
                            conflicts, groundedness, repeats, timeouts)           |
                                               |                               |
                                  +------------v-------------+                 |
                                  |  DETECT  - 43 rules     |                 |
                                  |  DIAGNOSE - rules, then |                 |
                                  |            LLM w/ schema |                 |
                                  +------------+-------------+                 |
                                               |                               |
                                  +------------v-------------+                 |
                                  |  PLAN  - targeted,       |                 |
                                  |         budget-aware     |                 |
                                  +------------+-------------+                 |
                                               |                               |
                                  +------------v-------------+                 |
                                  |  RECOVER - re-run ONLY   |                 |
                                  |           the broken     |                 |
                                  |           stage          |                 |
                                  +------------+-------------+                 |
                                               |                               |
                                  +------------v-------------+                 |
                                  |  VERIFY - re-measure.    |                 |
                                  |          passed -> answer|                 |
                                  |          failed -> next |                 |
                                  |          budget gone ->  |                 |
                                  |          ABSTAIN         |                 |
                                  +--------------------------+-----------------+
                                                             |
                                                     +-------v--------+
                                                     | answer or ABSTAIN|
                                                     +----------------+
```

Five properties hold at every step:

- **Targeted** — a strategy declares the earliest stage it re-runs; everything before it is
  reused. Across the evaluation, an attempt reused **2.718 stages on average (max 5)**. No
  recovery ever restarts the whole pipeline.
- **Verified** — `RecoveryOutcome.SUCCESS` is only set after `RecoveryVerifier` confirms the
  criteria. In 432 runs there were **192 verified recoveries and 0 unverified successes**.
- **Bounded** — a hard budget caps attempts, stage retries, repeated actions, LLM calls,
  tokens and total latency. `max_repeated_actions = 1`, so no action is ever tried twice.
- **Terminating** — every ladder ends in `ABSTAIN`. There is no infinite repair loop.
- **Honest** — an unrecoverable fault produces an abstention, not a confident wrong answer.

---

## Measured results

Produced by `python scripts/paper_fault_sweep.py 24`: **432 executed runs** covering all
**18 catalogue faults** crossed with all **24 evaluation questions**.

| metric | value |
|---|---|
| detection rate | **100.0 %** (432/432) |
| top-1 subtype accuracy | 61.34 % |
| top-k subtype accuracy | 63.19 % |
| recovery success (of attempted) | **56.14 %** (192/342), all verified |
| abstention rate | 39.81 % |
| diagnosis latency | p50 **0.253 ms** |
| added recovery latency | p50 72.287 ms / p95 331.519 ms |
| end-to-end latency | p50 91.147 ms |

A paired three-mode ablation over 1 086 further runs produced the most interesting result —
**answer-quality metrics cannot detect a corrupted pipeline**:

| mode | runs | groundedness | abstention % |
|---|---|---|---|
| `mode1` plain RAG | 361 | **0.9945** | 0.0 |
| `mode2` agentic, no recovery | 361 | **0.9945** | 0.0 |
| `mode3` agentic + recovery | 364 | 0.8516 | 34.07 |

A plain RAG baseline scores near-perfect groundedness *while its retrieval has been
truncated, reordered or replaced by a stale document*. Groundedness is therefore a quality
metric, not a detection signal — which is exactly why detection is built from stage-level
behavioural signals instead. Within mode 3, recovery still improves groundedness by
**+0.1319** on average (improved on 52 runs, worsened on 0).

Full analysis, tables and threats to validity: **[`docs/CONFERENCE_PAPER.md`](docs/CONFERENCE_PAPER.md)**.

---

## Using the dashboard & API

The dashboard is a thin client over the REST API, so both can be pointed at any backend via
the **API base URL** box in the sidebar (or the `ARR_API_URL` environment variable).

| page | what it shows |
|---|---|
| System Overview | health, budgets, taxonomy, strategies |
| Knowledge Base | documents, chunks, ingestion state |
| **Query Playground** | the whole pipeline for one query, with fault injection |
| Agent Trace | per-stage spans and latency for any recorded run |
| Failure Detection | counts by stage, class and severity |
| Diagnosis | evidence, confidence, tier, schema validation |
| Recovery | bounded loop, verification criteria, abstentions |
| Experiments | run stored configs or ad-hoc sweeps |
| Metrics | headline quality and cost metrics |
| Fault Injection | catalogue + inject-and-run console |
| Model Comparison | providers and retrieval configuration |
| Export | download traces, metrics and experiments |

The API exposes **30 endpoints**; see `/docs` for the full OpenAPI schema.

```bash
# ask a question
curl -X POST http://127.0.0.1:8012/query \
     -H "content-type: application/json" \
     -d '{"query":"What is the minimum password length?"}'

# ask the same question with a fault injected
curl -X POST http://127.0.0.1:8012/query \
     -H "content-type: application/json" \
     -d '{"query":"What is the minimum password length?","faults":["GENERATION_CONSTRAINT_REMOVAL"]}'
```

---

## Commands

```bash
# ---- data -----------------------------------------------------------------
python scripts/init_db.py --reset        # create the SQLite store
python scripts/ingest.py --force         # chunk + index the demo corpus
python scripts/db_summary.py             # what is stored, row by row

# ---- demonstrations -------------------------------------------------------
python run_demo.py                       # 10 scenarios, full traces
python run_demo.py --quick               # 3 scenarios
python run_demo.py --catalogue           # fault catalogue with ground truth

# ---- experiments ----------------------------------------------------------
python scripts/run_experiments.py --config baseline
python scripts/run_experiments.py --all  # 8 configs x 3 modes

# ---- validation -----------------------------------------------------------
python -B -m pytest -q                   # 295 tests
python scripts/validate_all.py           # 8-step end-to-end validation
python scripts/check_dashboard.py        # execute all 12 pages against a live API

# ---- services -------------------------------------------------------------
python scripts/serve.py [--status|--stop]
```

---

## Project layout

```
Ghouse/
├── agentic_rag_recovery/          # the core library
│   ├── pipeline/                  # stage graph, nodes, state, run records
│   ├── signals/                   # per-stage signal collection and metrics
│   ├── detection/                 # 43 deterministic detection rules
│   ├── diagnosis/                 # two-tier diagnosis, strict schema validation
│   ├── recovery/                  # planner, budget tracker, strategies, ladder
│   ├── validation/                # recovery verifier
│   ├── faults/                    # 18 fault definitions + injector
│   ├── evaluation/                # dataset, metrics, experiment runner
│   ├── llm/ retrieval/ storage/   # mock provider, hybrid retriever, SQLite
│   └── taxonomy.py                # 33 failure subtypes, recovery ladders
├── backend/                       # FastAPI app (30 endpoints) + schemas
├── dashboard/                     # Streamlit app + 12 pages
├── scripts/                       # init, ingest, serve, experiments, validation
├── experiments/configs/           # 8 seeded experiment configurations
├── data/                          # demo corpus (10 docs) and questions (24)
├── tests/                         # 295 tests
└── docs/
    ├── CONFERENCE_PAPER.md        # full write-up with measured results
    └── paper_data/                # raw per-run measurements
```

---

## Configuration

Everything tunable lives in **`configs/default.yaml`** — thresholds, budgets, retrieval and
generation settings. Thresholds are changed only with a documented calibration reason.

```yaml
budget:
  max_recovery_attempts: 4
  max_stage_retries: 2
  max_repeated_actions: 1
  max_llm_calls: 14
  max_token_budget: 24000
  max_total_latency_ms: 45000
```

Run in three modes for comparison: `mode1_normal_rag`, `mode2_agentic_no_recovery`,
`mode3_agentic_with_recovery`.

To use a real provider, set the provider and key in the environment — the mock is the
default and requires no configuration:

```bash
set OPENAI_API_KEY=sk-...      # then set generation.provider in configs/default.yaml
```

---

## Design rules

These are invariants the test suite enforces, not conventions:

1. **The diagnoser never sees fault ground truth.** Ground truth is recorded, not leaked
   (`tests/test_faults.py::test_ground_truth_is_recorded_not_leaked`).
2. **Recovery is always targeted.** Every strategy declares the earliest stage to re-run.
3. **Recovery is always verified.** Success is never asserted without `RecoveryVerifier`.
4. **No action is ever repeated.** `BudgetTracker.can_attempt()` returns `BLOCK_REPEAT`.
5. **Every ladder terminates in `ABSTAIN`** (`tests/test_recovery.py`).
6. **No fabricated results.** Every metric comes from an executed run; mock-provider results
   carry `DEMO / MOCK RESULTS`.
7. **No threshold is widened to make something pass.**

---

## Paper & data

`docs/CONFERENCE_PAPER.md` is a full conference-style write-up — problem statement, system
design, formal setup, results tables, ablation, discussion and threats to validity — and
every number in it is generated from executed runs.

The document is machine-verified against the data:

```bash
python scripts/paper_fault_sweep.py 24   # 432 runs  (~95 s)
python scripts/paper_aggregate.py        # -> docs/paper_data/results.{json,md}
python scripts/paper_by_mode.py          # -> docs/paper_data/by_mode.{json,md}
python scripts/paper_experiments.py      # -> docs/paper_data/experiments.json
python scripts/verify_paper_numbers.py   # re-checks every quoted value
```

`tests/test_paper.py` fails the build if a figure drifts from the data or if the mock-results
label is dropped.

---

## Troubleshooting

| symptom | fix |
|---|---|
| port 8012 / 8501 already in use | normal — `serve.py` picks free ports and prints the URLs |
| dashboard shows connection errors | the API is not running; check `python scripts/serve.py --status` |
| `ModuleNotFoundError` after install | the virtualenv is not active; re-activate it |
| want a clean slate | `python scripts/init_db.py --reset && python scripts/ingest.py --force` |
| want everything checked | `python scripts/validate_all.py` |

---

*B.Tech final-year project. All reported measurements come from the deterministic offline
provider and carry the label `DEMO / MOCK RESULTS`.*
