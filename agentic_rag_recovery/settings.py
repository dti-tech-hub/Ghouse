"""Configuration layer.

Everything tunable lives in YAML (``configs/default.yaml``) and is exposed
through :class:`Settings`.  Thresholds, budgets, model choices and path
locations are all overridable so that experiments are reproducible.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class ThresholdConfig:
    """Thresholds for the deterministic (Tier-1) detector."""

    # retrieval
    min_top_overlap: float = 0.34
    min_mean_overlap: float = 0.12
    useful_overlap: float = 0.20
    min_top_score: float = 0.12
    min_mean_score: float = 0.10
    min_score_gap: float = 0.01
    min_useful_chunk_ratio: float = 0.40
    max_duplicate_ratio: float = 0.60

    # ranking
    min_rerank_margin: float = 0.02
    max_rank_regression_rate: float = 0.34
    relevant_promotion_min_gain: float = 0.05

    # context
    min_context_tokens: int = 60
    max_context_tokens: int = 3000
    min_context_coverage: float = 0.50
    min_source_diversity: int = 1
    max_fragmentation_ratio: float = 0.50
    max_conflict_score: float = 0.35

    # generation / groundedness
    min_groundedness: float = 0.62
    min_claim_support: float = 0.34
    min_citation_coverage: float = 0.30
    max_unsupported_claims: int = 0
    min_answer_words: int = 12

    # query understanding
    min_query_words: int = 3
    max_query_words: int = 90
    ambiguous_entity_mentions: int = 3
    max_rewrite_drift: float = 0.62
    min_rewrite_change: float = 0.08

    # agent control
    max_repeat_actions: int = 1
    max_tool_calls: int = 12


@dataclass
class BudgetConfig:
    """Hard limits for the bounded recovery control loop."""

    max_recovery_attempts: int = 4
    max_total_latency_ms: int = 45_000
    max_llm_calls: int = 14
    max_token_budget: int = 24_000
    max_repeated_actions: int = 1
    max_stage_retries: int = 2


@dataclass
class ChunkingConfig:
    chunk_size: int = 320
    chunk_overlap: int = 60
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    min_chunk_chars: int = 30


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 12
    method: str = "hybrid"  # vector | bm25 | hybrid
    vector_backend: str = "auto"  # auto | faiss | numpy
    embedding_backend: str = "hashing"  # hashing | sentence_transformer
    embedding_dim: int = 384
    hybrid_alpha: float = 0.55  # weight on dense score
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_candidates: int = 10
    rerank_model: str = "lexical-cross"


@dataclass
class GenerationConfig:
    provider: str = "mock"  # mock | ollama | openai | claude
    model: str = "mock-deterministic-v1"
    temperature: float = 0.0
    max_tokens: int = 512
    ground_answer: bool = True
    cite_sources: bool = True
    require_grounding: bool = True
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout_s: float = 30.0


@dataclass
class DiagnosisConfig:
    rule_confidence_threshold: float = 0.55
    use_llm_fallback: bool = True
    llm_provider: Optional[str] = None  # defaults to generation provider
    llm_model: Optional[str] = None
    max_evidence_items: int = 8
    temperature: float = 0.0


@dataclass
class AgentConfig:
    max_steps: int = 8
    enable_decomposition: bool = True
    decomposition_min_words: int = 12
    enable_tools: bool = True
    tool_timeout_ms: int = 2_000


@dataclass
class PathsConfig:
    root: str = str(PROJECT_ROOT)
    data_dir: str = str(PROJECT_ROOT / "data")
    demo_documents: str = str(PROJECT_ROOT / "data" / "demo_documents")
    demo_questions: str = str(PROJECT_ROOT / "data" / "demo_questions")
    demo_ground_truth: str = str(PROJECT_ROOT / "data" / "demo_ground_truth")
    storage_dir: str = str(PROJECT_ROOT / "storage")
    index_dir: str = str(PROJECT_ROOT / "storage" / "index")
    upload_dir: str = str(PROJECT_ROOT / "storage" / "uploads")
    db_path: str = str(PROJECT_ROOT / "storage" / "agentic_rag.db")
    experiment_config_dir: str = str(PROJECT_ROOT / "experiments" / "configs")
    experiment_results_dir: str = str(PROJECT_ROOT / "experiments" / "results")


@dataclass
class Settings:
    """Root configuration object."""

    app_name: str = "Agentic RAG Failure Diagnosis and Recovery System"
    mode: str = "demo"  # demo | live
    seed: int = 42
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    labels: Dict[str, str] = field(default_factory=lambda: {"results": "DEMO / MOCK RESULTS"})
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def db_path(self) -> Path:
        return Path(self.paths.db_path)

    def index_dir(self) -> Path:
        return Path(self.paths.index_dir)

    def ensure_dirs(self) -> None:
        for key in ("storage_dir", "index_dir", "upload_dir"):
            Path(getattr(self.paths, key)).mkdir(parents=True, exist_ok=True)

    def copy(self) -> "Settings":
        return Settings.from_dict(self.to_dict())

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        data = copy.deepcopy(data or {})
        sections = {
            "thresholds": ThresholdConfig,
            "budget": BudgetConfig,
            "chunking": ChunkingConfig,
            "retrieval": RetrievalConfig,
            "generation": GenerationConfig,
            "diagnosis": DiagnosisConfig,
            "agent": AgentConfig,
            "paths": PathsConfig,
        }
        kwargs: Dict[str, Any] = {}
        for name, klass in sections.items():
            section = data.pop(name, {}) or {}
            valid = {k: v for k, v in section.items() if k in klass.__dataclass_fields__}
            kwargs[name] = klass(**valid)
        for key in ("app_name", "mode", "seed", "labels"):
            if key in data:
                kwargs[key] = data.pop(key)
        kwargs["raw"] = data
        instance = cls(**kwargs)
        instance.resolve_paths()
        return instance

    def resolve_paths(self) -> "Settings":
        """Resolve relative path entries against the project root."""
        base = Path(self.paths.root) if self.paths.root else PROJECT_ROOT
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        self.paths.root = str(base)
        for name in (
            "data_dir",
            "demo_documents",
            "demo_questions",
            "demo_ground_truth",
            "storage_dir",
            "index_dir",
            "upload_dir",
            "db_path",
            "experiment_config_dir",
            "experiment_results_dir",
        ):
            value = Path(getattr(self.paths, name))
            if not value.is_absolute():
                value = base / value
            setattr(self.paths, name, str(value))
        return self

    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None, **overrides: Any) -> "Settings":
        payload: Dict[str, Any] = {}
        default_path = PROJECT_ROOT / "configs" / "default.yaml"
        if default_path.exists():
            payload = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
        if path:
            p = Path(path)
            if p.exists():
                payload = _deep_merge(payload, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        env_path = os.environ.get("ARR_CONFIG_PATH")
        if env_path and Path(env_path).exists():
            payload = _deep_merge(payload, yaml.safe_load(Path(env_path).read_text(encoding="utf-8")) or {})
        if overrides:
            payload = _deep_merge(payload, overrides)
        settings = cls.from_dict(payload)

        # environment overrides (useful for the live-mode API)
        if os.environ.get("ARR_LLM_PROVIDER"):
            settings.generation.provider = os.environ["ARR_LLM_PROVIDER"]
        if os.environ.get("ARR_LLM_MODEL"):
            settings.generation.model = os.environ["ARR_LLM_MODEL"]
        if os.environ.get("ARR_LLM_BASE_URL"):
            settings.generation.base_url = os.environ["ARR_LLM_BASE_URL"]
        if os.environ.get("ARR_DB_PATH"):
            settings.paths.db_path = os.environ["ARR_DB_PATH"]
        if os.environ.get("ARR_MODE"):
            settings.mode = os.environ["ARR_MODE"]
        return settings


_settings: Optional[Settings] = None


def get_settings(reload: bool = False, **overrides: Any) -> Settings:
    global _settings
    if _settings is None or reload or overrides:
        _settings = Settings.load(**overrides)
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings


def settings_json(settings: Optional[Settings] = None) -> str:
    return json.dumps((settings or get_settings()).to_dict(), indent=2, default=str)