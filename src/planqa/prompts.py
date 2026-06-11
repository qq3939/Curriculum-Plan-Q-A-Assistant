from __future__ import annotations


SYSTEM_PROMPT = """你是一个面向本科学生的培养计划学业问答助手。

你的资料依据只来自当前提供的培养计划片段。你可以回答培养计划问题，也可以给学业相关建议，但必须遵守：
1. 回答使用简体中文，面向学生，尽量解释清楚。
2. 培养计划明文写到的内容，要标注引用编号，例如 [S1]。
3. 学业建议可以基于资料和学生情况合理归纳，但必须明确这是“建议”，不能伪装成官方规定。
4. 专业推荐第一版只做学生所属招生大类内推荐；如果不知道学生所属大类、兴趣、强弱项或限制，先追问。
5. 选课建议只基于培养计划中的建议修读学期、课程类别、学分结构和学生已修情况；不能判断实时课表、容量、时间冲突、教师安排或教务系统临时调整。
6. 如果资料不足以支持回答，要直接说明没有找到明确依据，并建议以教务系统、学院通知、导师或辅导员意见为准。
7. 不要编造课程、学分、政策、年份变化或实时信息。

回答学业建议时，优先给 1 到 3 个方案，每个方案包含适合人群、理由、风险或注意点、依据引用。
如果用户问题信息不足，不要强行推荐，先提出最少数量的关键追问。"""


def build_messages(
    user_question: str,
    context: str,
    chat_history: list[dict[str, str]],
    max_history_messages: int = 8,
) -> list[dict[str, str]]:
    recent_history = chat_history[-max_history_messages:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "user",
            "content": (
                "以下是从培养计划 PDF 检索出的资料片段。回答只能依据这些片段；如果要给建议，"
                "请明确区分资料依据与建议。\n\n"
                f"{context or '没有检索到相关资料片段。'}"
            ),
        }
    )
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
            "当前没有配置云端模型。我能检索培养计划原文，但不能判断实时课表、课程容量、"
            "时间冲突、教师安排或教室信息。请以教务系统和学院通知为准。"
        )
    if "2026" in question_text or "2024" in question_text or "2023" in question_text:
        return (
            "当前资料源是 2025 级培养计划。没有配置云端模型时，我无法进一步判断其他年级的变化；"
            "请以对应年级培养计划、教务系统和学院通知为准。"
        )
    if any(term in question_text for term in ["推荐", "选哪些课", "选课", "专业", "怎么安排"]):
        return (
            "当前没有配置云端模型，因此我先展示检索到的培养计划片段。若要生成个性化学业建议，"
            "请配置 API；建议类问题通常还需要说明年级、招生大类或专业、当前学期、已修课程、"
            "兴趣方向和学分压力偏好。"
        )
    if has_results:
        return "当前没有配置云端模型，因此我先展示检索到的培养计划原文片段，供你核对。"
    return "当前没有配置云端模型，也没有检索到明确相关的培养计划片段。"
