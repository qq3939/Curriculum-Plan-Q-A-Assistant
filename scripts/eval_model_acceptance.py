from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.categories import build_category_context
from planqa.config import load_config
from planqa.courses import build_course_context
from planqa.embeddings import make_embedding_provider
from planqa.intent import enhanced_search
from planqa.llm import OpenAICompatibleChatClient
from planqa.pdf_index import build_index, is_index_current, load_index
from planqa.prompts import build_messages
from planqa.retrieval import build_context, source_payload


@dataclass(frozen=True)
class AcceptanceCase:
    id: str
    group: str
    question: str
    expected_all: tuple[str, ...] = ()
    expected_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    not_recommended: tuple[str, ...] = ()
    required_source_terms: tuple[str, ...] = ()
    requires_citation: bool = True
    requires_boundary: bool = False
    student_context: str = ""
    retrieval_question: str = ""
    history: tuple[dict[str, str], ...] = ()


@dataclass
class CaseResult:
    case_id: str
    group: str
    question: str
    passed: bool
    failures: list[dict[str, str]] = field(default_factory=list)
    intent: str = ""
    confidence: float = 0.0
    answer_preview: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)


CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        id="F01",
        group="single_fact",
        question="通识教育课程最低要求多少学分？",
        expected_all=("42.5",),
        required_source_terms=("通识教育课程",),
        forbidden=("差额", "可能缺项", "未列出"),
    ),
    AcceptanceCase(
        id="F02",
        group="single_fact",
        question="通识-思政类最低要求多少学分？",
        expected_all=("17",),
        required_source_terms=("通识教育课程", "思政"),
    ),
    AcceptanceCase(
        id="F03",
        group="single_fact",
        question="计算机科学与技术专业的培养目标是什么？",
        expected_all=("计算机科学与技术", "培养目标"),
        required_source_terms=("计算机科学与技术", "培养目标"),
    ),
    AcceptanceCase(
        id="F04",
        group="single_fact",
        question="计算机科学与技术专业总学分是多少？",
        expected_all=("158",),
        required_source_terms=("计算机科学与技术", "总学分"),
    ),
    AcceptanceCase(
        id="F05",
        group="single_fact",
        question="计算机科学与技术专业基础理论最低要求多少学分？",
        expected_all=("24",),
        required_source_terms=("计算机科学与技术", "专业基础理论"),
    ),
    AcceptanceCase(
        id="F06",
        group="single_fact",
        question="计算机科学与技术的主干课程有哪些？",
        expected_any=("数据结构", "计算机网络", "操作系统", "数据库"),
        required_source_terms=("计算机科学与技术",),
    ),
    AcceptanceCase(
        id="F07",
        group="single_fact",
        question="机器人工程专业的培养目标是什么？",
        expected_all=("机器人工程", "培养目标"),
        required_source_terms=("机器人工程", "培养目标"),
    ),
    AcceptanceCase(
        id="F08",
        group="single_fact",
        question="工科试验班电子与信息类包含哪些专业？",
        expected_all=("计算机科学与技术", "自动化"),
        required_source_terms=("电子与信息类", "涵盖专业"),
    ),
    AcceptanceCase(
        id="C01",
        group="course_plan",
        question="计算机大二会学哪些课程？",
        expected_all=("数据结构", "二/1", "二/2"),
        expected_any=("操作系统D", "计算机网络", "数据库原理"),
        required_source_terms=("计算机科学与技术", "二/1", "二/2"),
    ),
    AcceptanceCase(
        id="C02",
        group="course_plan",
        question="计科大二要修啥，别太官方",
        expected_all=("数据结构", "二/1", "二/2"),
        forbidden=("理论3门", "理论课3门", "理论（3门"),
        required_source_terms=("计算机科学与技术", "二/1", "二/2"),
    ),
    AcceptanceCase(
        id="C03",
        group="course_plan",
        question="计算机科学与技术第4学期有哪些课？",
        expected_all=("计算机网络", "数据库原理", "计算机组成", "操作系统D"),
        required_source_terms=("计算机科学与技术", "二/2"),
    ),
    AcceptanceCase(
        id="C04",
        group="course_plan",
        question="计算机科学与技术大二短学期有哪些课程？",
        expected_all=("电子实习A", "工程认识实习"),
        expected_any=("短3", "二/2(短3)"),
        required_source_terms=("计算机科学与技术", "短3"),
    ),
    AcceptanceCase(
        id="C05",
        group="course_plan",
        question="我是计算机专业大一第二学期，已修高数A1，推荐这学期重点关注哪些课？",
        expected_any=("高等数学A(2)", "线性代数B", "大学物理A(1)", "程序设计"),
        required_source_terms=("计算机科学与技术", "一/2"),
    ),
    AcceptanceCase(
        id="C06",
        group="course_plan",
        question="软件工程大二课程怎么安排？",
        expected_any=("二/1", "二/2", "数据结构", "软件"),
        required_source_terms=("软件工程",),
    ),
    AcceptanceCase(
        id="C07",
        group="course_plan",
        question="每个专业大二（第3、4学期）的完整课程清单是什么？培养计划里面不是写得很明白吗",
        expected_all=("建议修读学年学期",),
        expected_any=("范围", "指定", "专业", "招生大类", "分批"),
        forbidden=("培养计划没有标注", "没有标注每门课程", "无法找到每个专业大二"),
        requires_boundary=True,
        required_source_terms=("二/1", "二/2"),
    ),
    AcceptanceCase(
        id="A01",
        group="admission_category",
        question="工科试验班智能化制造类包含哪些专业？",
        expected_all=("机器人工程", "机械设计制造及其自动化", "材料科学与工程"),
        required_source_terms=("智能化制造类", "涵盖专业"),
    ),
    AcceptanceCase(
        id="A02",
        group="admission_category",
        question="计算机科学与技术属于哪个招生大类？",
        expected_all=("工科试验班", "电子与信息类"),
        required_source_terms=("电子与信息类", "计算机科学与技术"),
    ),
    AcceptanceCase(
        id="A03",
        group="admission_category",
        question="智能科学与技术属于哪个大类？",
        expected_all=("电子与信息类",),
        required_source_terms=("智能科学与技术", "电子与信息类"),
    ),
    AcceptanceCase(
        id="A04",
        group="admission_category",
        question="设计学类包含哪些专业？",
        expected_any=("环境设计", "视觉传达设计", "产品设计"),
        required_source_terms=("设计学类",),
    ),
    AcceptanceCase(
        id="R01",
        group="recommendation",
        question="我是智能化制造类，喜欢计算机和机器人，推荐什么专业？",
        expected_all=("机器人工程",),
        not_recommended=("计算机科学与技术", "人工智能", "自动化", "智能科学与技术", "数据科学与大数据技术"),
        forbidden=("往年经验", "分流竞争压力", "分流名额可能有限", "录取分数", "热门程度", "绩点要求", "志愿排序"),
        required_source_terms=("智能化制造类", "机器人工程"),
    ),
    AcceptanceCase(
        id="R02",
        group="recommendation",
        question="我是智能化制造类，喜欢产品设计和人机交互，推荐什么专业？",
        expected_any=("工业设计", "机器人工程", "机械设计制造及其自动化"),
        not_recommended=("计算机科学与技术", "人工智能", "自动化"),
        required_source_terms=("智能化制造类",),
    ),
    AcceptanceCase(
        id="R03",
        group="recommendation",
        question="我是电子与信息类，喜欢算法和数据，推荐什么专业？",
        expected_any=("计算机科学与技术", "数据科学与大数据技术", "人工智能", "智能科学与技术"),
        not_recommended=("机器人工程", "机械设计制造及其自动化"),
        required_source_terms=("电子与信息类",),
    ),
    AcceptanceCase(
        id="R04",
        group="recommendation",
        question="帮我随便推荐一个最好的专业。",
        expected_any=("年级", "招生大类", "专业", "兴趣", "强弱项", "画像"),
        requires_citation=False,
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="R05",
        group="recommendation",
        question="我想轻松一点选课，怎么安排比较稳？",
        expected_any=("年级", "专业", "当前学期", "已修课程", "压力"),
        requires_citation=False,
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="B01",
        group="boundary",
        question="这门课这个学期还有名额吗？",
        expected_any=("无法判断", "不能判断", "教务系统", "学院通知"),
        forbidden=("还有名额", "有余量", "名额充足"),
        requires_citation=False,
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="B02",
        group="boundary",
        question="2026级培养计划有什么变化？",
        expected_any=("2025", "无法判断", "未找到", "对应年级"),
        forbidden=("2026级明确", "变化如下", "新增了"),
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="B03",
        group="boundary",
        question="计算机网络这门课哪个老师讲得最好？",
        expected_any=("无法判断", "不能判断", "教务系统", "学院通知"),
        forbidden=("最好的是", "推荐老师"),
        requires_citation=False,
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="B04",
        group="boundary",
        question="智能化制造类分流竞争压力大吗？录取分数大概多少？",
        expected_any=("培养计划片段中未找到明确依据", "无法判断", "不能判断", "学院", "辅导员"),
        forbidden=("往年经验", "录取分数是", "竞争压力较大", "热门程度"),
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="B05",
        group="boundary",
        question="这些课时间会不会冲突？",
        expected_any=("无法判断", "不能判断", "实时课表", "教务系统"),
        forbidden=("一定不会冲突", "不会有冲突", "时间不冲突"),
        requires_citation=False,
        requires_boundary=True,
    ),
    AcceptanceCase(
        id="M01",
        group="followup",
        question="那大二呢？",
        retrieval_question="计算机科学与技术 那大二呢？",
        history=(
            {"role": "user", "content": "我想了解计算机科学与技术专业的课程安排。"},
            {"role": "assistant", "content": "可以，我会以计算机科学与技术培养计划为依据回答。"},
        ),
        expected_all=("数据结构", "二/1", "二/2"),
        required_source_terms=("计算机科学与技术", "二/1", "二/2"),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 30-case real-model acceptance suite.")
    parser.add_argument("--case", dest="case_ids", action="append", help="Run one case id; can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected cases.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failing case.")
    parser.add_argument(
        "--json-output",
        default=str(ROOT / "data" / "model_acceptance_latest.json"),
        help="Write machine-readable results here.",
    )
    args = parser.parse_args()

    config = load_config(ROOT)
    if not config.has_chat_config:
        print("OPENAI_API_KEY or OPENAI_CHAT_MODEL is not configured.")
        return 2

    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        index = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        index = load_index(config.index_dir)

    selected = list(CASES)
    if args.case_ids:
        wanted = {case_id.upper() for case_id in args.case_ids}
        selected = [case for case in selected if case.id.upper() in wanted]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        print("No acceptance cases selected.")
        return 2

    client = OpenAICompatibleChatClient(config)
    print(f"Running {len(selected)} model acceptance case(s) with temperature=0")
    print(f"Chat model: {config.chat_model}")
    print(f"Embedding: {'local-hash' if config.uses_local_embeddings else config.embedding_model}")

    results: list[CaseResult] = []
    for number, case in enumerate(selected, start=1):
        print(f"\n[{number}/{len(selected)}] {case.id} {case.question}")
        result = run_case(case, index, provider, client)
        results.append(result)
        if result.passed:
            print(f"PASS intent={result.intent} confidence={result.confidence:.2f}")
        else:
            print(f"FAIL intent={result.intent} confidence={result.confidence:.2f}")
            for failure in result.failures:
                print(f"- {failure['type']}: {failure['message']}")
            if args.stop_on_fail:
                break

    write_json(args.json_output, config, results)
    failed = [result for result in results if not result.passed]
    print("\nSummary:")
    print(f"- passed: {len(results) - len(failed)}")
    print(f"- failed: {len(failed)}")
    print(f"- json: {args.json_output}")
    if failed:
        print("Failed cases:", ", ".join(result.case_id for result in failed))
        return 1
    return 0


def run_case(case: AcceptanceCase, index, provider, client: OpenAICompatibleChatClient) -> CaseResult:
    query = case.retrieval_question or case.question
    try:
        analysis, results = enhanced_search(index, provider, query, top_k=8)
        context = "\n\n".join(
            part
            for part in [
                build_category_context(index, query, analysis),
                build_course_context(index, query, analysis, results),
                build_context(results),
            ]
            if part
        )
        messages = build_messages(
            case.question,
            context,
            chat_history=list(case.history),
            intent_context=analysis,
            student_context=case.student_context,
        )
        answer = client.complete(messages, temperature=0)
        source_items = source_payload(results)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            group=case.group,
            question=case.question,
            passed=False,
            failures=[{"type": "api_or_runtime_error", "message": str(exc)}],
        )

    failures = validate_case(case, answer, source_items)
    return CaseResult(
        case_id=case.id,
        group=case.group,
        question=case.question,
        passed=not failures,
        failures=failures,
        intent=analysis.intent,
        confidence=analysis.confidence,
        answer_preview=answer[:1200],
        sources=source_items[:5],
    )


def validate_case(
    case: AcceptanceCase,
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    source_text = "\n".join(
        f"{source.get('file')} {source.get('section')} {source.get('text')}" for source in sources
    )

    if case.requires_citation and not re.search(r"[\[【]S\d+(?:[^\]】]*)?[\]】]", answer):
        failures.append({"type": "citation_missing", "message": "answer has no [S#] citation"})

    for term in case.expected_all:
        if term not in answer:
            failures.append({"type": "content_missing", "message": f"answer missed required term: {term}"})

    if case.expected_any and not any(term in answer for term in case.expected_any):
        failures.append(
            {
                "type": "content_missing",
                "message": "answer missed all acceptable terms: " + " / ".join(case.expected_any),
            }
        )

    for term in case.required_source_terms:
        if term not in source_text and term not in answer:
            failures.append({"type": "retrieval_error", "message": f"sources missed expected term: {term}"})

    for term in case.forbidden:
        if has_forbidden_assertion(answer, term):
            failures.append({"type": "hallucination_or_overreach", "message": f"forbidden phrase found: {term}"})

    for major in case.not_recommended:
        if is_recommended(answer, major):
            failures.append({"type": "out_of_category_recommendation", "message": f"recommended forbidden major: {major}"})

    if case.requires_boundary and not has_boundary_language(answer):
        failures.append({"type": "boundary_failure", "message": "answer did not clearly refuse, narrow, or ask for missing info"})

    return failures


def is_recommended(answer: str, major: str) -> bool:
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if major not in line:
            continue
        if is_negative_major_mention(line):
            continue
        if line.startswith("|"):
            cells = [cell.strip("* `") for cell in line.strip("|").split("|")]
            recommendation_cells = cells[:2]
            if any(major in cell and has_recommendation_marker(cell) for cell in recommendation_cells):
                return True
            continue
        if re.search(rf"方案[一二三123][：:、\s-]{{0,4}}{re.escape(major)}", line):
            return True
        if re.search(rf"{re.escape(major)}[（(]?(首选|次选|推荐|优先)", line):
            return True
        if re.search(rf"(推荐|建议|首选|优先|可以考虑)(专业|方向|方案)?[：:、\s-]{{0,6}}{re.escape(major)}", line):
            return True
    return False


def has_forbidden_assertion(answer: str, term: str) -> bool:
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if term in line and not is_negative_context(line):
            return True
    return False


def has_recommendation_marker(value: str) -> bool:
    return any(marker in value for marker in ["方案", "推荐", "首选", "次选", "优先", "可以考虑", "适合"])


def is_negative_major_mention(line: str) -> bool:
    return any(marker in line for marker in ["不在本大类", "不能作为", "不属于", "无法作为", "不建议作为", "不是本大类"])


def is_negative_context(line: str) -> bool:
    return any(
        marker in line
        for marker in [
            "未包含",
            "未提供",
            "未找到",
            "未检索",
            "没有",
            "不涉及",
            "不判断",
            "不能判断",
            "无法判断",
            "无法依据",
            "无法给出",
            "不能给出",
            "不会包含",
            "请勿依赖",
        ]
    )


def has_boundary_language(answer: str) -> bool:
    return any(
        term in answer
        for term in [
            "无法判断",
            "不能判断",
            "未找到明确依据",
            "培养计划片段中未找到",
            "请告诉我",
            "需要你提供",
            "指定",
            "范围",
            "教务系统",
            "学院通知",
            "导师",
            "辅导员",
        ]
    )


def write_json(path_value: str, config, results: list[CaseResult]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chat_model": config.chat_model,
        "embedding_model": "local-hash" if config.uses_local_embeddings else config.embedding_model,
        "temperature": 0,
        "total": len(results),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "results": [
            {
                "case_id": result.case_id,
                "group": result.group,
                "question": result.question,
                "passed": result.passed,
                "failures": result.failures,
                "intent": result.intent,
                "confidence": round(result.confidence, 3),
                "answer_preview": result.answer_preview,
                "sources": result.sources,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
