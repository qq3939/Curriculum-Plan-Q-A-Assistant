from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .embeddings import EmbeddingProvider
from .pdf_index import IndexBundle
from .retrieval import SearchResult, search


@dataclass(frozen=True)
class MatchedEntity:
    kind: str
    name: str
    score: float
    matched_by: str


@dataclass(frozen=True)
class IntentAnalysis:
    intent: str
    confidence: float
    normalized_query: str
    search_queries: list[str]
    entities: list[MatchedEntity] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def display_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "intent_label": INTENT_LABELS.get(self.intent, self.intent),
            "confidence": round(self.confidence, 3),
            "normalized_query": self.normalized_query,
            "search_queries": self.search_queries,
            "entities": [
                {
                    "kind": entity.kind,
                    "name": entity.name,
                    "score": round(entity.score, 3),
                    "matched_by": entity.matched_by,
                }
                for entity in self.entities
            ],
            "signals": self.signals,
        }


INTENT_LABELS = {
    "major_profile": "专业信息查询",
    "admission_category": "招生大类/专业归属查询",
    "credit_requirement": "学分/毕业要求查询",
    "course_plan": "课程/学期安排查询",
    "academic_advice": "学业建议/推荐",
    "policy_boundary": "实时或资料外信息",
    "general_qa": "培养计划问答",
}


_INTENT_RULES = {
    "academic_advice": ["推荐", "适合", "怎么选", "怎么安排", "规划", "轻松", "稳", "方向", "喜欢"],
    "course_plan": ["选课", "这学期", "下学期", "大一", "大二", "大三", "大四", "修什么", "要修啥", "修啥", "要修", "先修", "建议修读"],
    "credit_requirement": ["多少学分", "多少分", "几分", "拿多少分", "一共得", "学分", "毕业", "修满", "最低要求", "要求多少"],
    "admission_category": ["大类", "招生", "包含", "涵盖", "分流", "对应专业", "有哪些专业"],
    "major_profile": ["培养目标", "毕业要求", "主干课程", "专业介绍", "学什么", "就业", "方向"],
    "policy_boundary": ["名额", "余量", "时间冲突", "老师", "教室", "几点", "开课吗", "2026", "2024", "2023"],
}


_SYNONYM_MAP = {
    "计科": ["计算机科学与技术", "计算机", "软件开发", "软件工程", "硬件系统"],
    "计算机": ["计算机科学与技术", "数据科学与大数据技术", "人工智能", "智能科学与技术"],
    "机器人": ["机器人工程", "人工智能", "自动化", "机械设计制造及其自动化"],
    "ai": ["人工智能", "智能科学与技术", "数据科学与大数据技术"],
    "人工智能": ["人工智能", "智能科学与技术", "计算机科学与技术"],
    "大数据": ["数据科学与大数据技术", "计算机科学与技术"],
    "通信": ["通信工程", "电子信息工程"],
    "电信": ["电子信息工程", "通信工程", "电子科学与技术"],
    "自动控制": ["自动化", "电气工程及其自动化", "机器人工程"],
    "机械": ["机械设计制造及其自动化", "机器人工程", "车辆工程"],
    "车": ["车辆工程", "交通工程"],
    "新能源": ["新能源科学与工程", "储能科学与工程", "能源与动力工程"],
    "能动": ["能源与动力工程", "新能源科学与工程"],
    "生医": ["生物医学工程", "医学影像技术", "智能医学工程"],
    "通识": ["通识教育课程", "通识-思政类", "通识-英语类", "通识-体育类"],
    "公共课": ["通识教育课程", "最低要求42.5学分", "通识-思政类", "通识-英语类", "学科基础课程"],
    "大类课": ["学科基础课程(大类阶段)", "学科基础课程"],
    "学分够不够": ["学分结构及要求", "最低要求", "总学分"],
    "毕业要修": ["学分结构及要求", "毕业要求", "总学分"],
}

_SPECIFIC_ALIAS_MAP = {
    "计算机专业": ["计算机科学与技术"],
    "计算机大二": ["计算机科学与技术"],
    "计科专业": ["计算机科学与技术"],
    "计科": ["计算机科学与技术"],
    "智能制造类": ["工科试验班", "智能化制造类", "机器人工程", "机械设计制造及其自动化", "自动化"],
    "智能化制造类": ["工科试验班", "智能化制造类", "机器人工程", "机械设计制造及其自动化", "自动化"],
}


_FIELD_EXPANSIONS = {
    "学什么": ["主干课程", "课程设置", "培养目标"],
    "要修啥": ["课程设置", "建议修读学年学期", "学分结构及要求"],
    "这个学期": ["建议修读学年学期", "课程设置"],
    "毕业需要": ["毕业要求", "学分结构及要求", "总学分"],
    "适合我": ["培养目标", "主干课程", "毕业要求"],
    "有哪些": ["对应专业", "涵盖专业", "课程设置"],
    "大二": ["二/1", "二/2", "第二学年", "第3学期", "第4学期", "专业基础理论", "专业基础实践"],
    "第3学期": ["二/1", "第二学年第一学期", "专业基础理论", "专业基础实践"],
    "第4学期": ["二/2", "第二学年第二学期", "专业基础理论", "专业基础实践"],
    "第三学期": ["二/1", "第二学年第一学期", "专业基础理论", "专业基础实践"],
    "第四学期": ["二/2", "第二学年第二学期", "专业基础理论", "专业基础实践"],
}


def analyze_intent(index: IndexBundle, question: str) -> IntentAnalysis:
    normalized = _normalize_question(question)
    signals: list[str] = []
    intent_scores = {intent: 0.0 for intent in _INTENT_RULES}

    for intent, keywords in _INTENT_RULES.items():
        for keyword in keywords:
            if keyword.lower() in normalized:
                intent_scores[intent] += 1.0 + min(len(keyword), 6) / 10
                signals.append(f"{intent}:{keyword}")

    entities = _match_entities(index, question)
    if entities:
        intent_scores["major_profile"] += sum(0.4 for entity in entities if entity.kind == "专业")
        intent_scores["admission_category"] += sum(0.4 for entity in entities if entity.kind == "招生大类")
        signals.extend([f"{entity.kind}:{entity.name}" for entity in entities[:4]])

    if _asks_second_year(normalized) and _asks_course_arrangement(normalized):
        intent_scores["course_plan"] += 1.2
        signals.append("course_plan:second_year_course_arrangement")

    intent = max(intent_scores, key=intent_scores.get)
    if intent_scores[intent] <= 0:
        intent = "general_qa"
    confidence = _confidence(intent_scores.get(intent, 0.0), len(signals))

    queries = _build_search_queries(question, normalized, entities)
    return IntentAnalysis(
        intent=intent,
        confidence=confidence,
        normalized_query=normalized,
        search_queries=queries,
        entities=entities[:6],
        signals=signals[:12],
    )


def enhanced_search(
    index: IndexBundle,
    provider: EmbeddingProvider,
    question: str,
    top_k: int = 6,
) -> tuple[IntentAnalysis, list[SearchResult]]:
    analysis = analyze_intent(index, question)
    merged: dict[str, SearchResult] = {}
    query_count = max(len(analysis.search_queries), 1)

    for query_index, query in enumerate(analysis.search_queries):
        weight = 1.0 - min(query_index, 5) * 0.08
        for result in search(index, provider, query, top_k=max(top_k, 8)):
            key = result.chunk.chunk_id
            boosted = _boost_for_analysis(result, analysis, weight)
            previous = merged.get(key)
            if previous is None or boosted.score > previous.score:
                merged[key] = boosted

    if not merged:
        return analysis, []
    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    if analysis.intent == "course_plan" and _asks_second_year(analysis.normalized_query):
        scoped = _scope_to_matched_majors(ordered, analysis)
        if scoped:
            ordered = scoped
        major_count = sum(1 for entity in analysis.entities if entity.kind == "专业")
        top_k = max(top_k, min(12, major_count * 2))
    return analysis, ordered[:top_k]


def format_intent_context(analysis: IntentAnalysis) -> str:
    entities = "、".join(f"{entity.kind}:{entity.name}" for entity in analysis.entities) or "未识别到明确实体"
    queries = "；".join(analysis.search_queries)
    label = INTENT_LABELS.get(analysis.intent, analysis.intent)
    return "\n".join(
        [
            "意图分析:",
            f"- 识别意图: {label}",
            f"- 置信度: {analysis.confidence:.2f}",
            f"- 识别实体: {entities}",
            f"- 检索改写: {queries}",
        ]
    )


def _boost_for_analysis(result: SearchResult, analysis: IntentAnalysis, query_weight: float) -> SearchResult:
    boost = 0.0
    target = _compact(" ".join([result.chunk.source_file, result.chunk.section_title, result.chunk.text[:500]]))
    for entity in analysis.entities:
        if _compact(entity.name) in target:
            if entity.matched_by == "exact":
                entity_boost = 0.42
            elif entity.matched_by in _SPECIFIC_ALIAS_MAP:
                entity_boost = 0.32
            else:
                entity_boost = 0.18
            boost += entity_boost * entity.score
    if analysis.intent == "course_plan" and "建议修读" in result.chunk.text:
        boost += 0.08
    if analysis.intent == "course_plan" and _asks_second_year(analysis.normalized_query):
        if "二/1" in result.chunk.text or "二/2" in result.chunk.text:
            boost += 0.35
        if "专业基础理论" in result.chunk.text or "专业基础实践" in result.chunk.text:
            boost += 0.12
    if analysis.intent == "credit_requirement" and ("学分结构" in result.chunk.text or "最低要求" in result.chunk.text):
        boost += 0.08
    if analysis.intent == "admission_category" and ("对应专业" in result.chunk.section_title or "涵盖专业" in result.chunk.text):
        boost += 0.1
    if any(entity.kind == "课程模块" and entity.name in result.chunk.section_title for entity in analysis.entities):
        boost += 0.16

    return SearchResult(
        chunk=result.chunk,
        score=(result.score * query_weight) + boost,
        vector_score=result.vector_score,
        lexical_score=result.lexical_score,
    )


def _build_search_queries(question: str, normalized: str, entities: list[MatchedEntity]) -> list[str]:
    queries: list[str] = [question]

    expansions: list[str] = []
    for key, values in _SYNONYM_MAP.items():
        if key.lower() in normalized:
            expansions.extend(values)
    for key, values in _FIELD_EXPANSIONS.items():
        if key in normalized:
            expansions.extend(values)

    for entity in entities[:4]:
        expansions.append(entity.name)

    deduped_expansions = _dedupe(expansions)
    if deduped_expansions:
        queries.append(f"{question} {' '.join(deduped_expansions[:8])}")

    entity_query_limit = 6 if _asks_second_year(normalized) else 3
    for entity in entities[:entity_query_limit]:
        queries.append(f"{entity.name} {question}")
        if any(term in normalized for term in ["学什么", "课程", "选课", "这学期", "要修"]):
            queries.append(f"{entity.name} 课程设置 建议修读 学年学期")
        if any(term in normalized for term in ["毕业", "学分", "要求"]):
            queries.append(f"{entity.name} 学分结构及要求 毕业要求 总学分")
        if any(term in normalized for term in ["目标", "介绍", "适合", "方向"]):
            queries.append(f"{entity.name} 培养目标 主干课程 毕业要求")

    if any(term in normalized for term in ["大类", "分流", "包含", "涵盖"]):
        queries.append(f"{question} 2025级本科专业大类与对应专业一览表 招生大类 涵盖专业")
    if _asks_second_year(normalized):
        queries.append(f"{question} 二/1 二/2 第二学年 第3学期 第4学期 专业基础理论 专业基础实践 建议修读")

    return _dedupe(queries)[:10]


def _match_entities(index: IndexBundle, question: str) -> list[MatchedEntity]:
    normalized = _compact(question)
    candidates = _entity_candidates(index)
    matches: list[MatchedEntity] = []

    for kind, name in candidates:
        compact_name = _compact(name)
        if not compact_name or len(compact_name) < 2:
            continue
        if compact_name in normalized:
            matches.append(MatchedEntity(kind=kind, name=name, score=1.0, matched_by="exact"))
            continue
        ratio = difflib.SequenceMatcher(None, normalized, compact_name).ratio()
        partial = _partial_ratio(normalized, compact_name)
        score = max(ratio, partial)
        if score >= 0.72:
            matches.append(MatchedEntity(kind=kind, name=name, score=score, matched_by="fuzzy"))

    specific_alias_targets: set[str] = set()
    for alias, values in _SPECIFIC_ALIAS_MAP.items():
        if alias.lower() in question.lower() or _compact(alias) in normalized:
            for value in values:
                specific_alias_targets.add(value)
                matches.append(MatchedEntity(kind=_guess_kind(value), name=value, score=0.96, matched_by=alias))

    exact_major_names = {
        item.name
        for item in matches
        if item.kind == "专业" and item.matched_by in {"exact", *set(_SPECIFIC_ALIAS_MAP.keys())} and len(_compact(item.name)) >= 5
    }
    for alias, values in _SYNONYM_MAP.items():
        if alias.lower() in question.lower() or _compact(alias) in normalized:
            if specific_alias_targets and alias in {"计算机", "ai", "人工智能", "机器人", "机械"}:
                values = [value for value in values if value in specific_alias_targets]
            if any(_compact(alias) in _compact(name) for name in exact_major_names | specific_alias_targets):
                values = [value for value in values if value in exact_major_names]
            for value in values:
                matches.append(MatchedEntity(kind=_guess_kind(value), name=value, score=0.92, matched_by=alias))

    return _dedupe_entities(matches)


def _entity_candidates(index: IndexBundle) -> list[tuple[str, str]]:
    candidates: set[tuple[str, str]] = set()
    for chunk in index.chunks:
        section = chunk.section_title.split(">")[-1].strip()
        for name in re.findall(r"([\u4e00-\u9fffA-Za-z]+(?:\([\u4e00-\u9fffA-Za-z0-9]+\))?)\(\d{4}\)", section):
            candidates.add(("专业", name.strip()))
        if "通识教育课程" in section:
            candidates.add(("课程模块", "通识教育课程"))
        if "学科基础课程" in section:
            candidates.add(("课程模块", "学科基础课程"))
        if "专业大类与对应专业" in section:
            candidates.add(("招生大类", "2025级本科专业大类与对应专业一览表"))

    for keyword in ["工科试验班", "智能化制造类", "电子信息类", "经管类", "新闻传播学类", "设计学类"]:
        candidates.add(("招生大类", keyword))
    return sorted(candidates, key=lambda item: item[1])


def _guess_kind(name: str) -> str:
    if "课程" in name:
        return "课程模块"
    if "类" in name or "大类" in name:
        return "招生大类"
    return "专业"


def _confidence(score: float, signal_count: int) -> float:
    return min(0.35 + score * 0.16 + signal_count * 0.025, 0.98)


def _asks_second_year(normalized_question: str) -> bool:
    return any(term in normalized_question for term in ["大二", "第3学期", "第4学期", "第三学期", "第四学期", "二/1", "二/2"])


def _asks_course_arrangement(normalized_question: str) -> bool:
    return any(term in normalized_question for term in ["课程", "选课", "修", "学哪些", "会学", "要学", "学什么", "安排"])


def _scope_to_matched_majors(results: list[SearchResult], analysis: IntentAnalysis) -> list[SearchResult]:
    major_names = [_compact(entity.name) for entity in analysis.entities if entity.kind == "专业"]
    if not major_names:
        return results
    scoped: list[SearchResult] = []
    for result in results:
        target = _compact(" ".join([result.chunk.source_file, result.chunk.section_title, result.chunk.text[:500]]))
        if any(name in target for name in major_names):
            scoped.append(result)
    return scoped


def _normalize_question(question: str) -> str:
    value = question.strip().lower()
    value = value.replace("a1", "A1").replace("a2", "A2")
    value = re.sub(r"\s+", "", value)
    return value


def _compact(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value.lower()))


def _partial_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short in long:
        return 1.0
    if len(short) < 2:
        return 0.0
    best = 0.0
    window = len(short)
    for start in range(0, max(len(long) - window + 1, 1)):
        best = max(best, difflib.SequenceMatcher(None, short, long[start : start + window]).ratio())
    return best


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _dedupe_entities(items: list[MatchedEntity]) -> list[MatchedEntity]:
    best_by_name: dict[tuple[str, str], MatchedEntity] = {}
    for item in items:
        key = (item.kind, item.name)
        previous = best_by_name.get(key)
        if previous is None or item.score > previous.score:
            best_by_name[key] = item
    return sorted(best_by_name.values(), key=lambda item: item.score, reverse=True)
