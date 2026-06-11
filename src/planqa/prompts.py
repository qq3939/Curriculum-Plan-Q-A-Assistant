from __future__ import annotations


SYSTEM_PROMPT = """你是一个面向本科学生的培养计划学业问答助手。

你的资料依据只来自当前检索到的培养计划片段。你可以回答培养计划问题，也可以给学业相关建议，但必须遵守：
1. 使用简体中文，先给结论，再解释依据。
2. 培养计划明确写到的内容必须标注引用编号，例如 [S1]。
3. 学业建议必须明确区分“资料明确写明”和“基于资料与学生情况的建议”。
4. 专业推荐只在学生所属招生大类内推荐。缺少年级、招生大类/专业、兴趣、强弱项或限制条件时，先追问最少数量的关键问题。
5. 选课建议只基于建议修读学期、课程类别、学分结构和已修课程；不能判断实时课表、容量、时间冲突、教师、教室或临时调整。
6. 如果资料不足以支持回答，要直接说明“培养计划片段中未找到明确依据”，并提醒以教务系统、学院通知、导师或辅导员意见为准。
7. 不要编造课程、学分、政策、年份变化或实时信息。
8. 用户问“大二”“第3学期”“第4学期”时，优先查找课程表中的“二/1”“二/2”，它们就是培养计划中的建议修读学年学期。
9. 如果检索片段已经包含“二/1”“二/2”课程表，不要回答“没有找到按学期排列的课程表”。
10. 回答要像学业顾问：清楚、克制、可执行。不要为了显得完整而扩大到没有依据的专业、课程或年份。
11. 只回答用户实际问到的范围。用户只问最低学分、培养目标、包含哪些专业等单点问题时，不要主动展开无关模块、不要自行做复杂加总；例如只问“通识教育课程最低要求多少学分”时，只给 42.5 学分和来源即可。
12. 除非用户明确要求核算总学分，否则不要对不同课程模块做额外加总；如果资料片段没有完整覆盖所有子项，不要推断“差额”或“可能缺项”。
13. 如果上下文包含“结构化课程表辅助信息”，课程门数和学分统计必须以其中的统计为准，不要重新估算。

推荐类回答优先给 1 到 3 个方案。每个方案包含适合人群、推荐理由、注意风险和依据引用。
课程清单类回答优先按学期或课程类别整理；如果跨页来源共同构成清单，要合并说明。"""


def build_messages(
    user_question: str,
    context: str,
    chat_history: list[dict[str, str]],
    intent_context: object | None = None,
    student_context: str | None = None,
    max_history_messages: int = 8,
) -> list[dict[str, str]]:
    recent_history = chat_history[-max_history_messages:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    intent_text = ""
    if intent_context is not None:
        try:
            from .intent import format_intent_context

            intent_text = format_intent_context(intent_context)
        except Exception:
            intent_text = ""

    content_parts: list[str] = []
    if student_context:
        content_parts.append(
            "以下是学生在本次会话中提供的画像，只能用于当前回答，不要声称这些信息来自培养计划：\n"
            f"{student_context}"
        )
    if intent_text:
        content_parts.append(intent_text)
    content_parts.append(
        "以下是从培养计划 PDF 检索出的资料片段。回答必须先基于这些片段；如果给建议，要明确区分资料依据与建议。\n\n"
        f"{context or '没有检索到相关培养计划片段。'}"
    )
    messages.append({"role": "user", "content": "\n\n".join(content_parts)})

    for message in recent_history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_question})
    return messages


def fallback_answer(question: str, has_results: bool) -> str:
    question_text = question.strip()
    if any(term in question_text for term in ["名额", "余量", "时间冲突", "几点上课", "老师", "教室"]):
        return (
            "当前没有配置云端模型。我可以检索培养计划原文，但不能判断实时课表、课程容量、"
            "时间冲突、教师安排或教室信息。请以教务系统和学院通知为准。"
        )
    if "2026" in question_text or "2024" in question_text or "2023" in question_text:
        return (
            "当前资料源是 2025 级培养计划。没有配置云端模型时，我无法进一步判断其他年级的变化；"
            "请以对应年级培养计划、教务系统和学院通知为准。"
        )
    if any(term in question_text for term in ["推荐", "选哪些课", "选课", "专业", "怎么安排"]):
        return (
            "当前没有配置云端模型，因此我先展示检索到的培养计划片段。个性化学业建议通常还需要年级、"
            "招生大类或专业、当前学期、已修课程、兴趣方向和学分压力偏好。"
        )
    if has_results:
        return "当前没有配置云端模型，因此我先展示检索到的培养计划原文片段，供你核对。"
    return "当前没有配置云端模型，也没有检索到明确相关的培养计划片段。"
