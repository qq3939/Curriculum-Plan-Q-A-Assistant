from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .categories import get_admission_categories
from .intent import IntentAnalysis
from .pdf_index import IndexBundle, PdfChunk
from .retrieval import SearchResult


@dataclass(frozen=True)
class CourseRecord:
    major: str
    code: str
    name: str
    credits: str
    exam_type: str
    term: str
    group: str
    source_file: str
    page: int
    section_title: str


def build_course_context(
    index: IndexBundle,
    question: str,
    analysis: IntentAnalysis,
    results: list[SearchResult],
    max_records: int = 80,
) -> str:
    if analysis.intent != "course_plan":
        return ""

    majors = _target_majors(analysis)
    category_name, category_majors = _target_category(index, question, analysis)
    if not majors and category_majors:
        majors = category_majors
    term_prefixes = _target_term_prefixes(question)
    if not majors and not term_prefixes:
        return ""

    if term_prefixes and not majors:
        return _broad_scope_context(index, term_prefixes)

    records = find_course_records(index, majors=majors, term_prefixes=term_prefixes)
    source_pages = {(result.chunk.source_file, result.chunk.page) for result in results}
    if source_pages and not majors:
        scoped = [record for record in records if (record.source_file, record.page) in source_pages]
        if scoped:
            records = scoped
    if not records:
        return ""

    total_records = len(records)
    total_majors = len({record.major for record in records})
    records = records[:max_records]
    lines = [
        "结构化课程表辅助信息：",
        "以下课程行由培养计划课程表文本抽取，用于减少漏课；引用仍以检索来源 [S] 和页面原文为准。",
    ]
    if category_name:
        lines.append(f"识别到招生大类：{category_name}；以下仅使用该大类涵盖专业的课程表。")
        lines.append("该大类目标专业：" + "、".join(majors))
    if total_records > max_records:
        lines.append(
            f"注意：匹配范围共 {total_majors} 个专业、{total_records} 条课程行；"
            f"当前上下文只放入前 {max_records} 条。回答时不要声称这是完整清单，应建议用户缩小到某个专业或分批查看。"
        )
    summaries = _term_summaries(records)
    current_key: tuple[str, str] | None = None
    for record in records:
        key = (record.major, record.term)
        if key != current_key:
            summary = summaries.get(key, "")
            suffix = f"（{summary}）" if summary else ""
            lines.append(f"\n{record.major} - 建议修读学年学期 {record.term}{suffix}")
            current_key = key
        lines.append(
            "- "
            f"{record.name}（{record.credits} 学分，{record.exam_type}，{record.group}，"
            f"{record.source_file} p.{record.page}）"
        )
    return "\n".join(lines)


def find_course_records(
    index: IndexBundle,
    majors: list[str] | None = None,
    term_prefixes: list[str] | None = None,
) -> list[CourseRecord]:
    target_majors = {_compact_major(major) for major in (majors or [])}
    target_terms = term_prefixes or []
    records: list[CourseRecord] = []
    seen: set[tuple[str, str, str, int]] = set()

    for chunk in index.chunks:
        major = _major_from_chunk(chunk)
        if not major:
            continue
        if target_majors and _compact_major(major) not in target_majors:
            continue
        for record in _parse_chunk_courses(chunk, major):
            if target_terms and not any(record.term.startswith(prefix) for prefix in target_terms):
                continue
            key = (record.major, record.code, record.term, record.page)
            if key not in seen:
                seen.add(key)
                records.append(record)

    return sorted(records, key=lambda item: (item.major, _term_sort_key(item.term), item.group, item.page, item.code))


def _parse_chunk_courses(chunk: PdfChunk, major: str) -> list[CourseRecord]:
    if "课程代码" not in chunk.text or "建议修读" not in chunk.text:
        return []

    records: list[CourseRecord] = []
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    group = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        group = _updated_group(line, group)
        code, first_name = _course_code_and_name(line)
        if not code:
            index += 1
            continue

        parsed = _parse_course_at(lines, index, code, first_name, chunk, major, group)
        if parsed is None:
            index += 1
            continue
        record, next_index = parsed
        records.append(record)
        index = max(next_index, index + 1)

    return records


def _parse_course_at(
    lines: list[str],
    start: int,
    code: str,
    first_name: str,
    chunk: PdfChunk,
    major: str,
    group: str,
) -> tuple[CourseRecord, int] | None:
    name_lines: list[str] = []
    if first_name:
        name_lines.append(first_name)

    cursor = start + 1
    while cursor < len(lines):
        value = lines[cursor]
        if _is_credit(value):
            break
        if _course_code_and_name(value)[0] or _is_table_heading(value):
            return None
        name_lines.append(value)
        cursor += 1

    if not name_lines or cursor >= len(lines) or not _is_credit(lines[cursor]):
        return None

    name = _clean_name("".join(name_lines))
    credits = lines[cursor]
    cursor += 1

    exam_index = None
    for lookahead in range(cursor, min(cursor + 8, len(lines))):
        if lines[lookahead] in {"考试", "考查"}:
            exam_index = lookahead
            break
    if exam_index is None or exam_index + 1 >= len(lines):
        return None

    exam_type = lines[exam_index]
    term = lines[exam_index + 1]
    if not _is_term(term):
        return None

    record = CourseRecord(
        major=major,
        code=code,
        name=name,
        credits=credits,
        exam_type=exam_type,
        term=term,
        group=_fallback_group(name, group),
        source_file=chunk.source_file,
        page=chunk.page,
        section_title=chunk.section_title,
    )
    return record, exam_index + 2


def _target_majors(analysis: IntentAnalysis) -> list[str]:
    majors: list[str] = []
    for entity in analysis.entities:
        if entity.kind == "专业" and entity.name not in majors:
            majors.append(entity.name)
    return majors


def _target_category(index: IndexBundle, question: str, analysis: IntentAnalysis) -> tuple[str, list[str]]:
    compact_question = _compact_major(question)
    entity_names = {_compact_major(entity.name) for entity in analysis.entities if entity.kind == "招生大类"}
    best_name = ""
    best_majors: list[str] = []
    best_score = 0
    for category in get_admission_categories(index):
        score = 0
        for alias in category.aliases:
            compact_alias = _compact_major(alias)
            if compact_alias and compact_alias in compact_question:
                score = max(score, len(compact_alias))
            if compact_alias in entity_names:
                score = max(score, len(compact_alias) + 4)
        if score > best_score:
            best_name = category.name
            best_majors = list(category.majors)
            best_score = score
    return best_name, best_majors


def _target_term_prefixes(question: str) -> list[str]:
    compact = re.sub(r"\s+", "", question)
    prefixes: list[str] = []
    mappings = [
        (["大一", "第一学年"], ["一/1", "一/2"]),
        (["大二", "第二学年", "第3学期", "第三学期", "第4学期", "第四学期"], ["二/1", "二/2"]),
        (["大三", "第三学年", "第5学期", "第五学期", "第6学期", "第六学期"], ["三/1", "三/2"]),
        (["大四", "第四学年", "第7学期", "第七学期", "第8学期", "第八学期"], ["四/1", "四/2"]),
    ]
    for needles, values in mappings:
        if any(needle in compact for needle in needles):
            prefixes.extend(values)
    for term in ["一/1", "一/2", "二/1", "二/2", "三/1", "三/2", "四/1", "四/2"]:
        if term in compact:
            prefixes.append(term)
    return _dedupe(prefixes)


def _major_from_chunk(chunk: PdfChunk) -> str:
    section = chunk.section_title.split(">")[-1].strip()
    match = re.search(r"(.+?)\s*\(\d{4}\)", section)
    if not match:
        return ""
    return match.group(1).strip()


def _updated_group(line: str, current: str) -> str:
    if line.startswith("(") and ("课程" in line or "模块" in line or "实践" in line or "理论" in line):
        return line
    if "最低要求" in line and len(line) <= 40 and not _course_code_and_name(line)[0]:
        return line
    return current


def _course_code_and_name(line: str) -> tuple[str, str]:
    exact = re.fullmatch(r"(\d{7,8}[A-Za-z]?)", line)
    if exact:
        return exact.group(1), ""
    inline = re.match(r"^(\d{7,8}[A-Za-z]?)\s+(.+)$", line)
    if inline:
        return inline.group(1), inline.group(2).strip()
    return "", ""


def _is_credit(line: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", line))


def _is_term(line: str) -> bool:
    return bool(re.fullmatch(r"[一二三四]/[12](?:\(短\d+\))?", line))


def _is_table_heading(line: str) -> bool:
    return line in {"课程代码", "课程名称", "学分", "总学时", "考核", "方式"} or "建议修读" in line


def _clean_name(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    return value.replace("（", "(").replace("）", ")")


def _fallback_group(course_name: str, group: str) -> str:
    if group:
        return group
    if any(term in course_name for term in ["实验", "实习", "课程设计", "实践"]):
        return "专业基础实践"
    return "课程设置"


def _term_summaries(records: list[CourseRecord]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[CourseRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.major, record.term)].append(record)

    summaries: dict[tuple[str, str], str] = {}
    for key, term_records in grouped.items():
        total_credits = sum(_credit_value(record.credits) for record in term_records)
        group_parts: list[str] = []
        by_group: dict[str, list[CourseRecord]] = defaultdict(list)
        for record in term_records:
            by_group[_group_kind(record.group, record.name)].append(record)
        for group_name in ["专业基础理论", "专业基础实践", "专业课程", "其他课程"]:
            group_records = by_group.get(group_name, [])
            if group_records:
                credits = sum(_credit_value(record.credits) for record in group_records)
                group_parts.append(f"{group_name}{len(group_records)}门/{_format_credits(credits)}学分")
        summaries[key] = (
            f"共{len(term_records)}门，{_format_credits(total_credits)}学分；"
            + "，".join(group_parts)
        )
    return summaries


def _group_kind(group: str, course_name: str) -> str:
    if "理论" in group:
        return "专业基础理论"
    if "实践" in group or any(term in course_name for term in ["实验", "实习", "课程设计"]):
        return "专业基础实践"
    if "专业课程" in group or "核心课程" in group or "选修模块" in group:
        return "专业课程"
    return "其他课程"


def _credit_value(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _format_credits(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _broad_scope_context(index: IndexBundle, term_prefixes: list[str]) -> str:
    records = find_course_records(index, term_prefixes=term_prefixes)
    if not records:
        return ""
    majors = sorted({record.major for record in records})
    term_label = "、".join(term_prefixes)
    return "\n".join(
        [
            "结构化课程范围提示：",
            f"培养计划的各专业课程表中有“建议修读学年学期”列；{term_label} 对应大二相关学期。",
            f"系统已能从 PDF 课程表抽取到 {len(majors)} 个专业、{len(records)} 条 {term_label} 课程行。",
            "当前问题没有识别到具体专业或招生大类，范围过大；不要回答“培养计划没有标注每门课程的建议修读学期”。",
            "应先说明资料里有明确学期标注，再请用户指定专业或招生大类，或说明可以按专业/大类分批列出。",
        ]
    )


def _compact_major(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def _term_sort_key(term: str) -> tuple[int, int, str]:
    year_order = {"一": 1, "二": 2, "三": 3, "四": 4}
    match = re.match(r"([一二三四])/([12])", term)
    if not match:
        return (9, 9, term)
    return (year_order.get(match.group(1), 9), int(match.group(2)), term)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
