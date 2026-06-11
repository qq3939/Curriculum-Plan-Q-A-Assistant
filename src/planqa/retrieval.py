from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingProvider, normalize_matrix
from .pdf_index import IndexBundle, PdfChunk


@dataclass
class SearchResult:
    chunk: PdfChunk
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0


def search(index: IndexBundle, provider: EmbeddingProvider, query: str, top_k: int = 6) -> list[SearchResult]:
    if not query.strip() or len(index.chunks) == 0:
        return []
    query_vector = provider.embed_texts([query])
    query_vector = normalize_matrix(query_vector)[0]
    vector_scores = index.embeddings @ query_vector
    lexical_scores = np.asarray([_lexical_score(query, chunk) for chunk in index.chunks], dtype=np.float32)
    scores = (0.45 * vector_scores) + (0.55 * lexical_scores)
    top_k = min(max(top_k, 1), len(scores))
    indices = np.argpartition(scores, -top_k)[-top_k:]
    ordered = indices[np.argsort(scores[indices])[::-1]]
    return [
        SearchResult(
            chunk=index.chunks[int(idx)],
            score=float(scores[int(idx)]),
            vector_score=float(vector_scores[int(idx)]),
            lexical_score=float(lexical_scores[int(idx)]),
        )
        for idx in ordered
    ]


def source_payload(results: list[SearchResult]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for idx, result in enumerate(results, start=1):
        payload.append(
            {
                "id": f"S{idx}",
                "file": result.chunk.source_file,
                "page": result.chunk.page,
                "section": result.chunk.section_title,
                "text": result.chunk.text,
                "score": round(result.score, 4),
                "vector_score": round(result.vector_score, 4),
                "lexical_score": round(result.lexical_score, 4),
            }
        )
    return payload


def build_context(results: list[SearchResult], max_chars_per_source: int = 1800) -> str:
    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        chunk = result.chunk
        text = chunk.text
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."
        blocks.append(
            "\n".join(
                [
                    f"[S{idx}]",
                    f"文件: {chunk.source_file}",
                    f"页码: {chunk.page}",
                    f"章节: {chunk.section_title}",
                    "原文:",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def _lexical_score(query: str, chunk: PdfChunk) -> float:
    query_norm = _normalize(query)
    if not query_norm:
        return 0.0
    target = _normalize(" ".join([chunk.source_file, chunk.section_title, chunk.text]))
    terms = _query_terms(query_norm)
    if not terms:
        return 0.0

    total_weight = sum(_term_weight(term) for term in terms)
    if total_weight <= 0:
        return 0.0

    matched = 0.0
    for term in terms:
        if term in target:
            matched += _term_weight(term)
    base_score = matched / total_weight
    return min(base_score + _title_boost(query_norm, chunk), 1.0)


def _title_boost(query_norm: str, chunk: PdfChunk) -> float:
    candidates = [
        _title_core(chunk.section_title),
        _title_core(chunk.source_file.rsplit(".", 1)[0]),
    ]
    boost = 0.0
    for candidate in candidates:
        if len(candidate) < 4:
            continue
        if candidate in query_norm:
            boost = max(boost, 0.35)
        elif len(candidate) >= 6 and any(candidate[:length] in query_norm for length in range(4, len(candidate))):
            boost = max(boost, 0.18)
    return boost


def _normalize(value: str) -> str:
    value = value.lower()
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value))


def _title_core(value: str) -> str:
    normalized = _normalize(value)
    return re.sub(r"\d+", "", normalized)


def _query_terms(query_norm: str) -> list[str]:
    terms: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+", query_norm):
        token = match.group(0)
        if len(token) >= 2:
            terms.add(token)

    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", query_norm))
    for length in range(2, min(12, len(chinese)) + 1):
        for start in range(0, len(chinese) - length + 1):
            term = chinese[start : start + length]
            if term not in _STOP_TERMS:
                terms.add(term)
    return sorted(terms, key=lambda item: (-len(item), item))


def _term_weight(term: str) -> float:
    return float(len(term) ** 1.7)


_STOP_TERMS = {
    "什么",
    "多少",
    "哪些",
    "怎么",
    "一下",
    "可以",
    "这个",
    "学期",
    "专业",
    "课程",
    "培养",
    "计划",
    "要求",
}
