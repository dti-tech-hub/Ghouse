"""Datasets.

* ``data/demo_questions/`` + ``data/demo_ground_truth/`` form the **offline
  demonstration dataset** (always available, no download required).
* Public datasets (Natural Questions / HotpotQA / MS MARCO) are supported
  through a small loader that expects the original JSON(L) files placed under
  ``data/public_datasets/<name>/``. Nothing is fetched automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..settings import Settings, get_settings


@dataclass
class DemoQuestion:
    question_id: str
    question: str
    answer: str = ""
    answer_keywords: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    relevant_documents: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    is_complex: bool = False
    notes: str = ""
    expected_failure: Optional[str] = None
    paraphrase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "answer_keywords": self.answer_keywords,
            "source_document": self.source_document,
            "relevant_documents": self.relevant_documents,
            "category": self.category,
            "difficulty": self.difficulty,
            "is_complex": self.is_complex,
            "notes": self.notes,
            "expected_failure": self.expected_failure,
            "paraphrase": self.paraphrase,
        }


class DemoDataset:
    """The offline demonstration question set."""

    def __init__(self, questions: List[DemoQuestion]):
        self.questions = questions

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def by_id(self, question_id: str) -> Optional[DemoQuestion]:
        return next((q for q in self.questions if q.question_id == question_id), None)

    def by_category(self, category: str) -> List[DemoQuestion]:
        return [q for q in self.questions if q.category == category]

    def categories(self) -> List[str]:
        return sorted({q.category for q in self.questions})

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.questions),
            "categories": {c: len(self.by_category(c)) for c in self.categories()},
            "complex": sum(1 for q in self.questions if q.is_complex),
            "with_expected_failure": sum(1 for q in self.questions if q.expected_failure),
        }


def load_demo_dataset(settings: Optional[Settings] = None) -> DemoDataset:
    settings = settings or get_settings()
    root = Path(settings.paths.demo_questions)
    merged: Dict[str, Dict[str, Any]] = {}
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
            records = payload.get("questions", payload) if isinstance(payload, dict) else payload
            if isinstance(records, dict):
                records = [records]
            for record in records or []:
                if not isinstance(record, dict) or not record.get("question"):
                    continue
                record = dict(record)
                record.setdefault("question_id", record.get("id") or f"q{len(merged)+1:03d}")
                record.setdefault("answer", "")
                record.setdefault("answer_keywords", [])
                record.setdefault("relevant_documents", [])
                merged[record["question_id"]] = record
    if not merged:
        raise FileNotFoundError(
            f"no demo questions found under {root}. Expected *.json files with a 'questions' list."
        )
    questions = [
        DemoQuestion(
            question_id=str(record["question_id"]),
            question=str(record["question"]),
            answer=str(record.get("answer") or ""),
            answer_keywords=[str(k) for k in record.get("answer_keywords") or []],
            source_document=record.get("source_document"),
            relevant_documents=[str(d) for d in record.get("relevant_documents") or []],
            category=str(record.get("category") or "general"),
            difficulty=str(record.get("difficulty") or "medium"),
            is_complex=bool(record.get("is_complex")),
            notes=str(record.get("notes") or ""),
            expected_failure=record.get("expected_failure"),
            paraphrase=record.get("paraphrase"),
        )
        for record in merged.values()
    ]
    questions.sort(key=lambda q: q.question_id)
    return DemoDataset(questions)


# ---------------------------------------------------------------------------
# Public datasets (local files only - nothing is downloaded automatically)
# ---------------------------------------------------------------------------
PUBLIC_DATASETS = {
    "natural_questions": {
        "files": ["train.json", "dev.json", "validation.json"],
        "docs": "documents.json",
        "notes": "expects the simplified NQ release used by DPR",
    },
    "hotpotqa": {
        "files": ["train.json", "dev.json", "distractor/dev.json"],
        "notes": "expects HotpotQA json with 'context' field",
    },
    "ms_marco": {
        "files": ["triples.train.small.tsv", "dev.small.tsv"],
        "notes": "expects TSV: query, answer, passages",
    },
}


def dataset_available(name: str, settings: Optional[Settings] = None) -> Optional[Path]:
    settings = settings or get_settings()
    root = Path(settings.paths.data_dir) / "public_datasets" / name
    return root if root.exists() else None


def list_public_datasets(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    out: List[Dict[str, Any]] = []
    for name, meta in PUBLIC_DATASETS.items():
        path = dataset_available(name, settings)
        out.append(
            {
                "name": name,
                "available": bool(path),
                "path": str(path) if path else None,
                "expected_files": meta["files"],
                "notes": meta["notes"],
            }
        )
    return out


def load_public_dataset(name: str, limit: int = 200, settings: Optional[Settings] = None) -> List[DemoQuestion]:
    """Load a supported public dataset from local files."""
    settings = settings or get_settings()
    root = dataset_available(name, settings)
    if root is None:
        raise FileNotFoundError(
            f"public dataset '{name}' not present. Place its files under "
            f"{Path(settings.paths.data_dir) / 'public_datasets' / name}"
        )
    out: List[DemoQuestion] = []
    if name == "natural_questions":
        for path in _first_existing(root, PUBLIC_DATASETS[name]["files"]):
            for i, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
                if i >= limit:
                    break
                annotations = record.get("annotations") or [{}]
                long_answer = (annotations[0].get("long_answer") or {}).get("text", "")
                out.append(
                    DemoQuestion(
                        question_id=str(record.get("question", f"{path.stem}-{i}"))[:60],
                        question=str(record.get("question", "")),
                        answer=long_answer,
                        answer_keywords=_keywords(long_answer),
                        category="natural_questions",
                    )
                )
    elif name == "hotpotqa":
        for path in _first_existing(root, PUBLIC_DATASETS[name]["files"]):
            for i, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
                if i >= limit:
                    break
                answer = str(record.get("answer", ""))
                context = record.get("context") or [[], []]
                titles = [str(t) for t in (context[0] if context else [])]
                out.append(
                    DemoQuestion(
                        question_id=str(record.get("_id", f"{path.stem}-{i}")),
                        question=str(record.get("question", "")),
                        answer=answer,
                        answer_keywords=_keywords(answer),
                        relevant_documents=titles,
                        category="hotpotqa",
                        is_complex=True,
                    )
                )
    elif name == "ms_marco":
        for path in _first_existing(root, PUBLIC_DATASETS[name]["files"]):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if i >= limit:
                    break
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                answer = parts[1]
                out.append(
                    DemoQuestion(
                        question_id=f"{path.stem}-{i}",
                        question=parts[0],
                        answer=answer,
                        answer_keywords=_keywords(answer),
                        category="ms_marco",
                    )
                )
    return out


def _first_existing(root: Path, names: Iterable[str]) -> List[Path]:
    return [root / n for n in names if (root / n).exists()]


def _keywords(answer: str, limit: int = 6) -> List[str]:
    from ..textutil import tokenize

    return list(dict.fromkeys(t for t in tokenize(answer) if len(t) > 2))[:limit]