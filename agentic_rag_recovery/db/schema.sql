-- =====================================================================
--  Agentic RAG Failure Diagnosis and Recovery System
--  SQLite schema (primary persistent store)
--  All timestamps are ISO-8601 UTC strings.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- Knowledge base
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    title         TEXT,
    doc_type      TEXT,
    content       TEXT,
    checksum      TEXT UNIQUE,
    version       INTEGER DEFAULT 1,
    is_active     INTEGER DEFAULT 1,
    metadata      TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_source   ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_documents_active   ON documents(is_active);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index    INTEGER NOT NULL,
    content        TEXT NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    char_start     INTEGER DEFAULT 0,
    char_end       INTEGER DEFAULT 0,
    page           INTEGER,
    section        TEXT,
    source         TEXT,
    metadata       TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index    ON chunks(document_id, chunk_index);

CREATE TABLE IF NOT EXISTS index_versions (
    index_version_id TEXT PRIMARY KEY,
    vector_backend   TEXT,
    embedding_model TEXT,
    embedding_dim    INTEGER,
    chunk_config     TEXT,
    chunk_count      INTEGER,
    built_at         TEXT
);

-- ---------------------------------------------------------------------
-- Queries / runs
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS queries (
    query_id      TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    normalized    TEXT,
    rewrite       TEXT,
    subqueries    TEXT,
    intent        TEXT,
    is_complex    INTEGER DEFAULT 0,
    source        TEXT,             -- playground | experiment | api
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    retrieval_run_id TEXT PRIMARY KEY,
    run_id           TEXT,
    query_id         TEXT,
    query            TEXT NOT NULL,
    method           TEXT,
    top_k            INTEGER,
    candidate_k      INTEGER,
    latency_ms       REAL,
    diagnostics      TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_run ON retrieval_runs(run_id);

CREATE TABLE IF NOT EXISTS retrieved_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    retrieval_run_id TEXT NOT NULL REFERENCES retrieval_runs(retrieval_run_id) ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL,
    document_id     TEXT,
    rank            INTEGER,
    score           REAL,
    method          TEXT,
    dense_score     REAL,
    sparse_score    REAL,
    rerank_score    REAL,
    original_rank   INTEGER,
    is_relevant     INTEGER,
    source          TEXT,
    page            INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_run ON retrieved_chunks(retrieval_run_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id              TEXT PRIMARY KEY,
    experiment_id       TEXT,
    query_id            TEXT,
    mode                TEXT NOT NULL,
    status              TEXT NOT NULL,
    query               TEXT NOT NULL,
    answer              TEXT,
    baseline_answer     TEXT,
    understanding       TEXT,
    context             TEXT,
    generation          TEXT,
    retrieval_diag      TEXT,
    signals             TEXT,
    metrics             TEXT,
    faults              TEXT,
    labels              TEXT,
    baseline_latency_ms REAL DEFAULT 0,
    total_latency_ms    REAL DEFAULT 0,
    added_latency_ms    REAL DEFAULT 0,
    tokens_used         INTEGER DEFAULT 0,
    llm_calls           INTEGER DEFAULT 0,
    cost                REAL DEFAULT 0,
    abstained           INTEGER DEFAULT 0,
    abstention          TEXT,
    created_at          TEXT NOT NULL,
    finished_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_exp  ON agent_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_mode ON agent_runs(mode);
CREATE INDEX IF NOT EXISTS idx_agent_runs_q    ON agent_runs(query);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id          TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index       INTEGER NOT NULL,
    stage            TEXT,
    action           TEXT,
    tool_name        TEXT,
    input_summary    TEXT,
    output_summary   TEXT,
    status           TEXT,
    latency_ms       REAL,
    action_signature TEXT,
    error            TEXT,
    meta             TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id, step_index);

CREATE TABLE IF NOT EXISTS trace_spans (
    span_id   TEXT PRIMARY KEY,
    run_id    TEXT NOT NULL,
    parent_id TEXT,
    stage     TEXT,
    name      TEXT,
    start_time TEXT,
    end_time  TEXT,
    latency_ms REAL,
    status    TEXT,
    inputs    TEXT,
    outputs   TEXT,
    metrics   TEXT,
    error     TEXT,
    tags      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_spans_run ON trace_spans(run_id, start_time);

-- ---------------------------------------------------------------------
-- Diagnosis
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id      TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    experiment_id     TEXT,
    failure_detected  INTEGER DEFAULT 0,
    failure_class     TEXT,
    stage             TEXT,
    subtype           TEXT,
    severity          TEXT,
    confidence        REAL,
    tier              TEXT,
    rule_id           TEXT,
    recommended_recovery TEXT,
    recovery_ladder   TEXT,
    evidence          TEXT,
    rule_matches      TEXT,
    signals_used      TEXT,
    latency_ms        REAL,
    raw_llm_output    TEXT,
    validation_errors TEXT,
    repairs           TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_diagnoses_run  ON diagnoses(run_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_exp  ON diagnoses(experiment_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_cls  ON diagnoses(failure_class);

-- ---------------------------------------------------------------------
-- Fault injection
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fault_injections (
    fault_id              TEXT PRIMARY KEY,
    fault_type            TEXT NOT NULL,
    target_stage          TEXT,
    severity              TEXT,
    description           TEXT,
    expected_symptom      TEXT,
    expected_failure_class TEXT,
    expected_subtype      TEXT,
    injection_parameters  TEXT,
    active                INTEGER DEFAULT 0,
    experiment_id         TEXT,
    run_id                TEXT,
    created_at            TEXT NOT NULL,
    released_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_faults_type ON fault_injections(fault_type);
CREATE INDEX IF NOT EXISTS idx_faults_run  ON fault_injections(run_id);

-- ---------------------------------------------------------------------
-- Recovery
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recovery_attempts (
    attempt_id       TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    experiment_id    TEXT,
    attempt_index    INTEGER NOT NULL,
    action           TEXT NOT NULL,
    target_stage     TEXT,
    diagnosis_subtype TEXT,
    status           TEXT,
    rationale        TEXT,
    reused_stage     INTEGER DEFAULT 0,
    skipped_stages   TEXT,
    verification     TEXT,
    before_state     TEXT,
    after_state      TEXT,
    latency_ms       REAL,
    added_latency_ms REAL,
    tokens_used      INTEGER DEFAULT 0,
    cost             REAL DEFAULT 0,
    llm_calls        INTEGER DEFAULT 0,
    error            TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_recovery_run  ON recovery_attempts(run_id, attempt_index);
CREATE INDEX IF NOT EXISTS idx_recovery_act  ON recovery_attempts(action);

CREATE TABLE IF NOT EXISTS abstentions (
    abstention_id TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    reason        TEXT,
    failure_class TEXT,
    stage         TEXT,
    subtype       TEXT,
    attempted     TEXT,
    attempts_used INTEGER,
    budget_consumed TEXT,
    message       TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_abstentions_run ON abstentions(run_id);

-- ---------------------------------------------------------------------
-- Experiments & metrics
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id    TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT,
    dataset          TEXT,
    fault_types      TEXT,
    model            TEXT,
    retrieval_config TEXT,
    recovery_config  TEXT,
    repetitions      INTEGER DEFAULT 1,
    seed             INTEGER,
    mode             TEXT,
    config_path      TEXT,
    config_hash      TEXT,
    status           TEXT DEFAULT 'pending',
    result_label     TEXT,
    started_at       TEXT,
    finished_at      TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id     TEXT PRIMARY KEY,
    experiment_id TEXT,
    run_id        TEXT,
    name          TEXT NOT NULL,
    value         REAL,
    value_json    TEXT,
    group_name    TEXT,
    label         TEXT,
    seed          INTEGER,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_exp   ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name  ON metrics(name);
CREATE INDEX IF NOT EXISTS idx_metrics_group ON metrics(group_name);

-- ---------------------------------------------------------------------
-- Model + configuration registry (provenance)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    model_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    provider      TEXT NOT NULL,
    role          TEXT,
    version       TEXT,
    params        TEXT,
    cost_per_1k_in  REAL DEFAULT 0,
    cost_per_1k_out REAL DEFAULT 0,
    is_default    INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS configurations (
    configuration_id TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    kind             TEXT,
    description      TEXT,
    payload          TEXT,
    hash             TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments_config (
    experiment_config_id TEXT PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    description     TEXT,
    payload         TEXT,
    seed            INTEGER,
    created_at      TEXT NOT NULL
);