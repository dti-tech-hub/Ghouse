"""Deterministic custom metrics.

These are the *primary* evaluation metrics of the project.  External
libraries (RAGAS / DeepEval) are optional adapters layered on top; the
numbers in the database always come from these functions so that results are
reproducible offline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..textutil import (
    capitalized_terms,
    char_ngrams,
    coverage_of_terms,
    has_negation,
    jaccard,
    mean,
    numeric_values,
    overlap_coefficient,
    split_sentences,
    stem,
    tokenize,
)

CITATION_RE = re.compile(r"\[(S\d+)\]")


# ---------------------------------------------------------------------------
# Generation-side metrics
# ---------------------------------------------------------------------------
def split_claims(answer: str) -> List[str]:
    """Split an answer into atomic claims (sentences, citations stripped)."""
    if not answer:
        return []
    cleaned = CITATION_RE.sub("", answer)
    claims = [c.strip() for c in split_sentences(cleaned) if c.strip()]
    # a single-word claim is still a claim; dropping it would inflate groundedness
    return [c for c in claims if len(tokenize(c)) >= 1]


def extract_citations(answer: str) -> List[str]:
    return CITATION_RE.findall(answer or "")


def claim_support(claim: str, context: str) -> float:
    """Support of ``claim`` by ``context`` = max over context sentences.

    Uses a lexical-numeric hybrid so that a correct *number* or a correct
    entity counts as support even when phrasing differs.
    """
    if not claim.strip():
        return 0.0
    context_sentences = split_sentences(context) or ([context] if context else [])
    claim_tokens = set(tokenize(claim))
    claim_nums = numeric_values(claim)
    claim_caps = capitalized_terms(claim)
    if not context_sentences:
        return 0.0
    best = 0.0
    for sentence in context_sentences:
        sentence_tokens = set(tokenize(sentence))
        overlap = len(claim_tokens & sentence_tokens) / max(1, len(claim_tokens))
        ngram = _char_ngram_overlap(claim, sentence)
        numeric_bonus = 1.0 if claim_nums and claim_nums <= numeric_values(sentence) else 0.0
        entity_bonus = (
            0.25
            if claim_caps and (claim_caps & capitalized_terms(sentence))
            else 0.0
        )
        score = min(1.0, 0.55 * overlap + 0.30 * ngram + 0.40 * numeric_bonus + entity_bonus)
        best = max(best, score)
    return best


def _char_ngram_overlap(a: str, b: str, n: int = 4) -> float:
    ga = set(char_ngrams(a, n))
    if not ga:
        return 0.0
    gb = set(char_ngrams(b, n))
    if not gb:
        return 0.0
    return len(ga & gb) / len(ga)


def groundedness_score(
    question: str,
    answer: str,
    context: Sequence[str],
    citation_markers: Optional[Sequence[str]] = None,
    support_threshold: float = 0.34,
) -> Dict[str, Any]:
    """Fraction of answer claims that are entailed by the retrieved evidence."""
    claims = split_claims(answer)
    joined = "\n".join(c for c in context if c)
    if not claims:
        return {
            "groundedness": 0.0,
            "claims": [],
            "claim_scores": [],
            "unsupported_claims": [],
            "supported_count": 0,
            "claim_count": 0,
            "citation_coverage": 0.0,
            "citations": [],
            "cited_markers": [],
            "min_claim_support": 0.0,
        }
    scores = [claim_support(c, joined) for c in claims]
    unsupported = [c for c, s in zip(claims, scores) if s < support_threshold]
    groundedness = 1.0 - (len(unsupported) / len(claims))
    citations = extract_citations(answer)
    valid = set(citation_markers or [])
    cited = [c for c in citations if c in valid] if valid else citations
    coverage = len(set(cited)) / max(1, len(valid)) if valid else (1.0 if citations else 0.0)
    return {
        "groundedness": round(groundedness, 6),
        "claims": claims,
        "claim_scores": [round(s, 6) for s in scores],
        "unsupported_claims": unsupported,
        "supported_count": len(claims) - len(unsupported),
        "claim_count": len(claims),
        "citation_coverage": round(min(1.0, coverage), 6),
        "citations": citations,
        "cited_markers": sorted(set(cited)),
        "min_claim_support": round(min(scores), 6),
    }


def citation_coverage(answer: str, context_sources: Sequence[str]) -> float:
    """Fraction of retrieved sources that are actually cited in the answer."""
    if not context_sources:
        return 0.0
    cited = set(extract_citations(answer))
    used = sum(1 for i in range(1, len(context_sources) + 1) if f"S{i}" in cited)
    return round(used / len(context_sources), 6)


# ---------------------------------------------------------------------------
# Context-side metrics
# ---------------------------------------------------------------------------
def context_coverage(
    question: str, context: str, subqueries: Sequence[str] = (), constraints: Sequence[str] = ()
) -> float:
    """Coverage of the question's information need by the context.

    Averages three evidence sources, each of which must be satisfied:
    * the content terms of the question,
    * every explicit constraint extracted at query-understanding time,
    * every decomposed sub-question.
    """
    parts: List[float] = []
    base_terms = tokenize(question)
    if base_terms:
        parts.append(coverage_of_terms(context, base_terms))
    constraint_terms = [t for c in constraints for t in tokenize(c)]
    if constraint_terms:
        parts.append(coverage_of_terms(context, constraint_terms))
    for sq in subqueries:
        terms = tokenize(sq)
        if terms:
            parts.append(coverage_of_terms(context, terms))
    return round(mean(parts), 6) if parts else 1.0


QUANTITY_RE = re.compile(r"how many\s+((?:\w+\s+){0,2}?(?:minutes|hours|days|weeks|months|percent|units|attempts|retries|times|people))")


def constraint_coverage(question: str, constraints: Sequence[str], context: str) -> Tuple[float, List[str]]:
    """Which explicit constraints of the question the context does not cover.

    A ``how many <unit>`` constraint additionally requires the evidence to
    contain an actual quantity of that unit - the lexical unit alone is not
    enough to answer "how many minutes".
    """
    missing: List[str] = []
    scores: List[float] = []
    for constraint in constraints:
        quantity = QUANTITY_RE.search(constraint.lower())
        if quantity:
            unit = quantity.group(1).strip()
            pattern = re.compile(
                r"\b\d+(?:\.\d+)?\s*(?:\w+\s+){0,2}?" + re.escape(unit.split()[-1]) + r"\b"
            )
            hit = 1.0 if pattern.search(context.lower()) else 0.0
            scores.append(hit)
            if hit < 0.5:
                missing.append(constraint)
            continue
        terms = tokenize(constraint)
        if not terms:
            continue
        hit = coverage_of_terms(context, terms)
        scores.append(hit)
        if hit < 0.5:
            missing.append(constraint)
    coverage_value = round(mean(scores), 6) if scores else 1.0
    return coverage_value, missing


def _negates(sentence: str, shared_terms: Optional[Set[str]] = None) -> bool:
    """True when the sentence contains a negation applied to a content term."""
    tokens = tokenize(sentence, remove_stopwords=False, do_stem=False)
    negations = {"not", "no", "never", "cannot", "without", "neither", "nor", "n't", "unable"}
    for index, token in enumerate(tokens):
        if token in negations:
            window = {stem(t) for t in tokens[index + 1 : index + 5]}
            if shared_terms is None:
                return True
            if window & shared_terms:
                return True
    return False


def evidence_fragmentation(
    question: str,
    corpus_chunks: Sequence[Any],
    selected_ids: Sequence[str],
    document_texts: Optional[Dict[str, str]] = None,
    max_terms: int = 6,
) -> float:
    """Fraction of question-relevant sentences that the corpus itself splits.

    Sentences are taken from the **whole documents** (not from chunk text -
    otherwise the split we want to detect is invisible), and each one is checked
    for whether any single chunk can hold at least 70 percent of its content
    tokens.  If no chunk can, the evidence for that sentence is fragmented in
    the corpus, which is exactly what the CHUNK_EXPANSION recovery repairs.
    """
    if not corpus_chunks or not selected_ids:
        return 0.0
    selected = set(selected_ids)
    terms = [t for t in dict.fromkeys(tokenize(question)) if len(t) > 2][:max_terms]
    if not terms:
        return 0.0
    prepared = [(c, set(tokenize(c.content))) for c in corpus_chunks]
    split = 0
    total = 0
    per_term_cap = 4
    for term in terms:
        seen_for_term = 0
        for document_id, text in (document_texts or {}).items():
            if seen_for_term >= per_term_cap:
                break
            if term not in tokenize(text):
                continue
            for sentence in split_sentences(text):
                sentence_tokens = set(tokenize(sentence))
                if len(sentence_tokens) < 5 or term not in sentence_tokens:
                    continue
                total += 1
                seen_for_term += 1
                strong = [
                    c.chunk_id
                    for c, other in prepared
                    if c.document_id == document_id
                    and len(sentence_tokens & other) >= 0.7 * len(sentence_tokens)
                ]
                if not strong:
                    split += 1
                if seen_for_term >= per_term_cap:
                    break
    return round(split / total, 6) if total else 0.0


#: Function-like words that carry no topical content.  Contradiction detection
#: requires the sentences to agree on *content*, otherwise innocuous pairs such
#: as "must be acknowledged within 5 minutes" vs "must not retry more than 3
#: times" would be reported as contradictions.
FUNCTION_WORDS = {
    "must", "shall", "require", "requires", "required", "within", "after", "before",
    "at", "least", "most", "maximum", "more", "than", "every", "each", "all", "any",
    "only", "also", "which", "that", "these", "those", "are", "were", "being", "been",
    "have", "has", "had", "do", "does", "did", "make", "makes", "made", "use", "uses",
    "used", "take", "takes", "may", "can", "should", "will", "when", "while", "during",
    "per", "there", "their", "from", "into", "onto", "over", "under", "about", "above",
    "below", "between", "without", "unless", "whether", "either", "neither", "same",
    "other", "another", "such", "both", "each", "any", "all", "one", "two",
}

MIN_CONFLICT_SHARED_TOKENS = 3
MIN_CONFLICT_JACCARD = 0.34
MIN_NEGATION_MASK_JACCARD = 0.85
MIN_NEGATION_TOKENS = 3


def _content_tokens(sentence: str) -> Set[str]:
    return set(tokenize(sentence)) - FUNCTION_WORDS


def numeric_slots(sentence: str) -> Dict[Tuple[str, ...], Set[str]]:
    """Map each number in a sentence to the content tokens immediately before it."""
    tokens = tokenize(sentence)
    slots: Dict[Tuple[str, ...], Set[str]] = {}
    for index, token in enumerate(tokens):
        if not token.isdigit():
            continue
        context = tuple(t for t in tokens[max(0, index - 3) : index] if t not in FUNCTION_WORDS)
        slots.setdefault(context, set()).add(token)
    return slots


def _numeric_conflict(a: str, b: str) -> Optional[Dict[str, Any]]:
    """Two sentences contradict numerically only when exactly *one* quantity differs.

    "Severity 1 -> 5 minutes" vs "Severity 2 -> 15 minutes" differs in two slots
    and is therefore two different facts, not a contradiction; "retained for 35
    days" vs "retained for 20 days" differs in one slot and is a contradiction.
    """
    slots_a, slots_b = numeric_slots(a), numeric_slots(b)
    differing = [key for key in set(slots_a) | set(slots_b) if slots_a.get(key) != slots_b.get(key)]
    common = [key for key in set(slots_a) & set(slots_b) if slots_a.get(key) != slots_b.get(key)]
    if len(differing) == 1 and len(common) == 1:
        # A differing negation changes *which population* the statement is
        # about ("vendors that do not process data" vs "vendors that do"),
        # so differing numbers there are a contrast, not a contradiction.
        if _negates(a) != _negates(b):
            return None
        key = common[0]
        return {
            "type": "numeric",
            "values_a": sorted(slots_a.get(key, set())),
            "values_b": sorted(slots_b.get(key, set())),
            "slot": list(key),
        }
    return None


def _negation_conflict(a: str, b: str) -> Optional[Dict[str, Any]]:
    """Contradiction by negation: same statement, one version denies it."""
    neg_a, neg_b = _negates(a), _negates(b)
    if neg_a == neg_b:
        return None
    masked_a = set(tokenize(_mask_negations(a)))
    masked_b = set(tokenize(_mask_negations(b)))
    if len(masked_a) < MIN_NEGATION_TOKENS or len(masked_b) < MIN_NEGATION_TOKENS:
        return None
    jaccard = len(masked_a & masked_b) / len(masked_a | masked_b)
    if jaccard < MIN_NEGATION_MASK_JACCARD:
        return None
    shared = _content_tokens(a) & _content_tokens(b)
    if len(shared) < MIN_NEGATION_SHARED:
        return None
    return {"type": "negation", "masked_similarity": round(jaccard, 4)}


MIN_NEGATION_SHARED = 2


def _mask_negations(sentence: str) -> str:
    return re.sub(
        r"\b(?:not|no|never|cannot|n't|without|neither|nor|unable)\b", " ", sentence, flags=re.IGNORECASE
    )


def conflict_score(context_items: Sequence[Sequence[str]]) -> Tuple[float, List[Dict[str, Any]]]:
    """Detect contradictions in the assembled context.

    A pair is only flagged when the two sentences demonstrably make claims
    about the *same* statement: shared content tokens plus either a negation
    flip on an otherwise identical sentence, or exactly one differing quantity.
    """
    conflicts: List[Dict[str, Any]] = []
    sentences: List[Tuple[str, str]] = []
    for i, texts in enumerate(context_items, start=1):
        for sentence in split_sentences(" ".join(texts)):
            sentences.append((f"S{i}", sentence))
    if len(sentences) < 2:
        return 0.0, conflicts

    pairs_checked = 0
    for a in range(len(sentences)):
        for b in range(a + 1, len(sentences)):
            ma, sa = sentences[a]
            mb, sb = sentences[b]
            ta, tb = set(tokenize(sa)), set(tokenize(sb))
            if not ta or not tb:
                continue
            if len(ta & tb) < MIN_CONFLICT_SHARED_TOKENS:
                continue
            content_shared = _content_tokens(sa) & _content_tokens(sb)
            if len(content_shared) < 2:
                continue
            jaccard = len(ta & tb) / len(ta | tb)
            if jaccard < MIN_CONFLICT_JACCARD:
                continue
            pairs_checked += 1
            detail = _numeric_conflict(sa, sb) or _negation_conflict(sa, sb)
            if detail:
                conflicts.append(
                    {
                        **detail,
                        "marker_a": ma,
                        "marker_b": mb,
                        "shared_terms": sorted(content_shared)[:8],
                        "similarity": round(jaccard, 4),
                        "text_a": sa[:180],
                        "text_b": sb[:180],
                    }
                )
    # Scale: one *verified* contradiction is already a serious context defect
    # (0.5 = medium), two or more saturate the signal at 1.0.
    return round(min(1.0, 0.5 * len(conflicts)), 6), conflicts


# ---------------------------------------------------------------------------
# Retrieval-side metrics (require ground truth)
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Set[str], k: Optional[int] = None) -> float:
    if not gold_ids:
        return 0.0
    top = list(retrieved_ids)[: k or len(retrieved_ids)]
    return round(len(gold_ids & set(top)) / len(gold_ids), 6)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Set[str], k: Optional[int] = None) -> float:
    top = list(retrieved_ids)[: k or len(retrieved_ids)]
    if not top:
        return 0.0
    return round(len(gold_ids & set(top)) / len(top), 6)


def hit_rate(retrieved_ids: Sequence[str], gold_ids: Set[str]) -> float:
    return 1.0 if gold_ids & set(retrieved_ids) else 0.0


def mrr(retrieved_ids: Sequence[str], gold_ids: Set[str]) -> float:
    for position, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return round(1.0 / position, 6)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k."""
    if not gold_ids:
        return 0.0
    from math import log2

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, cid in enumerate(list(retrieved_ids)[:k], start=1)
        if cid in gold_ids
    )
    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(ideal_count))
    return round(dcg / idcg, 6) if idcg else 0.0


# ---------------------------------------------------------------------------
# Answer correctness (against a reference answer when available)
# ---------------------------------------------------------------------------
def answer_correctness(answer: str, reference: str, answer_keywords: Sequence[str] = ()) -> Dict[str, Any]:
    """Token-F1 against a reference answer plus keyword-hit evidence.

    This is the standard SQuAD-style proxy; it is deliberately reported next to
    the raw numbers so a reader can judge how strict it is.
    """
    if not reference:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": False, "keyword_coverage": 0.0}
    a_tokens = tokenize(answer)
    r_tokens = tokenize(reference)
    common: Dict[str, int] = {}
    for token in set(a_tokens):
        if token in r_tokens:
            common[token] = min(a_tokens.count(token), r_tokens.count(token))
    precision = recall = 0.0
    if a_tokens and r_tokens:
        precision = sum(common.values()) / len(a_tokens)
        recall = sum(common.values()) / len(r_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        f1 = 0.0
    keyword_coverage = coverage_of_terms(answer, answer_keywords) if answer_keywords else 0.0
    exact = normalize_answer(answer) == normalize_answer(reference)
    return {
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "exact_match": exact,
        "keyword_coverage": round(keyword_coverage, 6),
    }


def normalize_answer(text: str) -> str:
    tokens = [t for t in tokenize(text)]
    return " ".join(sorted(tokens))


def duplicate_ratio(items: Sequence[str]) -> float:
    if not items:
        return 0.0
    unique = len(set(items))
    return round(1.0 - unique / len(items), 6)


def source_diversity(items: Iterable[str]) -> int:
    return len({(item or "").split("/")[-1] for item in items})


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of duplicated n-grams - a degenerate-generation detector."""
    tokens = tokenize(text, remove_stopwords=False, do_stem=False)
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return round(1.0 - (len(set(grams)) / len(grams)), 6)