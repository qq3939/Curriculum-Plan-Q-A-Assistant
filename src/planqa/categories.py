from __future__ import annotations

import re
from dataclasses import dataclass

from .intent import IntentAnalysis
from .pdf_index import IndexBundle


@dataclass(frozen=True)
class AdmissionCategory:
    name: str
    aliases: tuple[str, ...]
    majors: tuple[str, ...]
    source_file: str
    page: int


_CATEGORY_MARKERS = [
    ("工科试验班(智能化制造类)", "工科试验班\n(智能化制造类)"),
    ("工科试验班(电子与信息类)", "工科试验班\n(电子与信息类)"),
    ("理科试验班", "理科试验班"),
    ("经济管理试验班", "经济管理试验班"),
    ("新闻传播学类", "新闻传播学类"),
    ("设计学类", "设计学类"),
]

_INTEREST_KEYWORDS = {
    "机器人工程": ["机器人", "编程", "计算机", "人工智能", "控制", "嵌入式", "视觉", "软硬件", "智能制造"],
    "机械设计制造及其自动化": ["智能制造", "机械", "制造", "自动化", "建模", "机电", "装备", "机器人"],
    "工业设计": ["设计", "产品", "交互", "人机", "外观", "用户"],
    "车辆工程": ["车辆", "汽车", "智能车", "自动驾驶", "交通"],
    "过程装备与控制工程": ["装备", "控制", "过程", "化工", "机械"],
    "测控技术与仪器": ["测控", "传感", "仪器", "控制", "硬件"],
    "智能科学与技术": ["人工智能", "智能", "算法", "机器学习", "计算机"],
    "计算机科学与技术": ["计算机", "编程", "软件", "算法", "系统"],
    "数据科学与大数据技术": ["数据", "大数据", "机器学习", "算法", "计算机"],
    "人工智能": ["人工智能", "机器学习", "算法", "机器人", "计算机"],
    "自动化": ["控制", "自动化", "机器人", "智能制造", "电气"],
}


def get_admission_categories(index: IndexBundle) -> list[AdmissionCategory]:
    chunks = [
        chunk
        for chunk in index.chunks
        if chunk.page in {1, 11} and "2025 级本科专业大类与对应专业一览表" in chunk.text
    ]
    if not chunks:
        return []

    # The same table appears in multiple PDFs; prefer the standalone table, then de-duplicate by name.
    chunks.sort(key=lambda chunk: (0 if "专业大类与对应专业一览表.pdf" in chunk.source_file else 1, chunk.source_file))
    categories: dict[str, AdmissionCategory] = {}
    for chunk in chunks:
        for category in _parse_category_chunk(chunk.text, chunk.source_file, chunk.page):
            categories.setdefault(category.name, category)
    return list(categories.values())


def build_category_context(index: IndexBundle, question: str, analysis: IntentAnalysis) -> str:
    if analysis.intent not in {"academic_advice", "admission_category"}:
        return ""

    categories = get_admission_categories(index)
    matched = _match_category(categories, question, analysis)
    if matched is None:
        return ""

    candidates = _rank_category_majors(matched, question)
    out_of_category = _out_of_category_interest_hits(categories, matched, question)

    lines = [
        "结构化招生大类辅助信息：",
        f"识别到学生所属招生大类：{matched.name}（来源：{matched.source_file} p.{matched.page}）。",
        "该大类涵盖专业：",
        "、".join(matched.majors),
    ]
    if candidates:
        lines.append("\n根据用户兴趣词在本大类内的候选排序（仅作建议依据，不能替代官方分流规则）：")
        for major, score, keywords in candidates[:5]:
            reason = "、".join(keywords) if keywords else "与问题中的大类或专业名称匹配"
            lines.append(f"- {major}: 匹配 {reason}，score={score}")
    if out_of_category:
        lines.append("\n注意：以下兴趣相关专业不属于该招生大类，不能作为本大类内推荐结论：")
        for major, category_name in out_of_category:
            lines.append(f"- {major} 属于 {category_name}")
    return "\n".join(lines)


def _parse_category_chunk(text: str, source_file: str, page: int) -> list[AdmissionCategory]:
    positions: list[tuple[int, str, str]] = []
    for name, marker in _CATEGORY_MARKERS:
        position = text.find(marker)
        if position >= 0:
            positions.append((position, name, marker))
    note_position = text.find("注：")
    if note_position < 0:
        note_position = len(text)
    positions.sort(key=lambda item: item[0])

    categories: list[AdmissionCategory] = []
    for index, (position, name, marker) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else note_position
        segment = text[position + len(marker) : end]
        majors = _parse_majors(segment)
        if majors:
            categories.append(
                AdmissionCategory(
                    name=name,
                    aliases=_aliases_for(name),
                    majors=tuple(majors),
                    source_file=source_file,
                    page=page,
                )
            )
    return categories


def _parse_majors(segment: str) -> list[str]:
    compact = re.sub(r"\s+", "", segment)
    compact = compact.replace("○", "○")
    matches = re.findall(r"○\d+([^○]+)", compact)
    majors: list[str] = []
    for raw in matches:
        name = raw.strip("、，, ")
        name = re.sub(r"[、，,].*$", "", name)
        if name and name not in majors:
            majors.append(name)
    return majors


def _aliases_for(name: str) -> tuple[str, ...]:
    aliases = {name}
    if "(" in name and ")" in name:
        inner = name[name.find("(") + 1 : name.rfind(")")]
        aliases.add(inner)
        aliases.add(inner.replace("化", ""))
    aliases.add(name.replace("工科试验班", "").strip("()"))
    return tuple(alias for alias in aliases if alias)


def _match_category(
    categories: list[AdmissionCategory],
    question: str,
    analysis: IntentAnalysis,
) -> AdmissionCategory | None:
    compact_question = _compact(question)
    entity_names = {_compact(entity.name) for entity in analysis.entities if entity.kind == "招生大类"}
    best: AdmissionCategory | None = None
    best_score = 0
    for category in categories:
        score = 0
        for alias in category.aliases:
            compact_alias = _compact(alias)
            if compact_alias and compact_alias in compact_question:
                score = max(score, len(compact_alias))
            if compact_alias in entity_names:
                score = max(score, len(compact_alias) + 4)
        if score > best_score:
            best = category
            best_score = score
    return best


def _rank_category_majors(category: AdmissionCategory, question: str) -> list[tuple[str, int, list[str]]]:
    compact_question = _compact(question)
    ranked: list[tuple[str, int, list[str]]] = []
    for major in category.majors:
        keywords = _INTEREST_KEYWORDS.get(major, [])
        matched_keywords = [keyword for keyword in keywords if _compact(keyword) in compact_question]
        score = len(matched_keywords) * 3
        if _compact(major) in compact_question:
            score += 8
        if score:
            ranked.append((major, score, matched_keywords))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _out_of_category_interest_hits(
    categories: list[AdmissionCategory],
    matched_category: AdmissionCategory,
    question: str,
) -> list[tuple[str, str]]:
    compact_question = _compact(question)
    hits: list[tuple[str, str]] = []
    matched_majors = set(matched_category.majors)
    for category in categories:
        if category.name == matched_category.name:
            continue
        for major in category.majors:
            if major in matched_majors:
                continue
            keywords = _INTEREST_KEYWORDS.get(major, [])
            if _compact(major) in compact_question or any(_compact(keyword) in compact_question for keyword in keywords):
                hits.append((major, category.name))
    return hits[:8]


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()
