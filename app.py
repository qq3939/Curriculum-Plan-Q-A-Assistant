from __future__ import annotations

import html
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from planqa.categories import build_category_context
from planqa.config import load_config
from planqa.courses import build_course_context
from planqa.embeddings import make_embedding_provider
from planqa.intent import enhanced_search
from planqa.llm import OpenAICompatibleChatClient
from planqa.pdf_index import build_index, is_index_current, load_index
from planqa.prompts import build_messages, fallback_answer
from planqa.retrieval import build_context, source_payload


WELCOME_MESSAGE = "你好，今天想先查哪一块培养计划？"

EXAMPLE_PROMPTS = [
    "计算机大二会学哪些课程",
    "通识教育课程最低要求多少学分？",
    "工科试验班智能化制造类包含哪些专业？",
    "我是智能化制造类，喜欢计算机和机器人，推荐什么专业？",
]

PROFILE_FIELDS = {
    "year": "年级",
    "category": "招生大类",
    "major": "专业",
    "term": "当前学期",
    "completed": "已修课程",
    "interests": "兴趣方向",
    "strengths": "强弱项",
    "load": "学分压力偏好",
}


@st.cache_resource(show_spinner="正在读取培养计划索引...")
def load_or_build_index(root_dir: str, provider_signature: str):
    config = load_config(Path(root_dir))
    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider_signature):
        return build_index(config.corpus_dir, config.index_dir, provider)
    return load_index(config.index_dir)


def main() -> None:
    st.set_page_config(page_title="培养计划学业问答助手", layout="wide")
    apply_theme()
    init_session()

    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    chat_client = OpenAICompatibleChatClient(config)

    try:
        index = load_or_build_index(str(ROOT), provider.signature)
    except Exception as exc:
        st.error(f"索引加载失败：{exc}")
        return

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if pending_prompt:
        run_answer_question(pending_prompt, current_profile(), index, provider, chat_client)
        st.rerun()

    render_top_bar(index, chat_client, current_profile())
    main_col, dock_col = st.columns([0.64, 0.36], gap="large")

    with main_col:
        render_answer_timeline()
        prompt = render_command_center()

    with dock_col:
        profile = render_profile_inspector(config, provider, chat_client)
        render_evidence_dock(latest_assistant_message())
        render_profile_controls(config, provider)

    if prompt:
        run_answer_question(prompt, profile, index, provider, chat_client)
        st.rerun()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-app: #f8fafc;
            --bg-card: #ffffff;
            --bg-subtle: #f1f5f9;
            --bg-fact: #f0fdf4;
            --bg-advice: #eff6ff;
            --bg-risk: #fff7ed;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-crisp: #e2e8f0;
            --border-soft: #f1f5f9;
            --fact: #166534;
            --advice: #1e40af;
            --risk: #c2410c;
            --radius: 6px;
            --shadow-card: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg-app);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
                "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
        }

        .main .block-container {
            max-width: 98% !important;
            padding: 1.05rem 1.35rem 2rem !important;
        }

        #MainMenu, footer, header [data-testid="stToolbar"] {
            visibility: hidden;
        }

        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            border: 1px solid var(--border-crisp);
            border-radius: var(--radius);
            background: rgba(255, 255, 255, 0.86);
            padding: 12px 16px;
            margin-bottom: 14px;
            box-shadow: var(--shadow-card);
        }

        .product-title {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0;
        }

        .product-subtitle {
            color: var(--text-secondary);
            font-size: 12px;
            margin-top: 2px;
        }

        .top-tags, .quick-tags, .profile-tags, .source-pills {
            display: flex;
            gap: 7px;
            flex-wrap: wrap;
            align-items: center;
        }

        .top-tag, .quick-tag, .profile-tag, .source-pill {
            border: 1px solid var(--border-crisp);
            border-radius: 4px;
            background: var(--bg-subtle);
            color: #334155;
            font-size: 12px;
            line-height: 1.35;
            padding: 3px 7px;
            white-space: nowrap;
        }

        .workspace-label {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin: 0 0 9px;
        }

        .empty-workspace, .answer-card, .profile-panel, .source-dock, .intent-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-crisp);
            border-radius: var(--radius);
            box-shadow: var(--shadow-card);
        }

        .empty-workspace {
            padding: 22px 22px 20px;
            margin-bottom: 14px;
        }

        .empty-title {
            font-size: 20px;
            line-height: 1.45;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .empty-copy {
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.65;
            max-width: 720px;
        }

        .user-query {
            border-left: 3px solid #64748b;
            padding: 8px 0 8px 12px;
            margin: 0 0 12px;
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 600;
        }

        .answer-card {
            padding: 18px 20px 20px;
            margin-bottom: 22px;
        }

        .card-timestamp {
            color: var(--text-muted);
            font-size: 11px;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .answer-body {
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.68;
        }

        .answer-body h2, .answer-body h3 {
            font-size: 16px;
            line-height: 1.5;
            margin: 15px 0 8px;
        }

        .answer-body p {
            margin: 8px 0;
        }

        .answer-body ul {
            margin: 6px 0 10px 1.15rem;
            padding: 0;
        }

        .answer-body li {
            margin: 4px 0;
        }

        .answer-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }

        .answer-table th,
        .answer-table td {
            border: 1px solid var(--border-crisp);
            padding: 6px 7px;
            text-align: left;
            vertical-align: top;
            word-break: break-word;
        }

        .answer-table th {
            background: var(--bg-subtle);
            color: var(--text-primary);
            font-weight: 700;
        }

        .answer-body strong {
            color: var(--text-primary);
            font-weight: 700;
        }

        .fact-block, .advice-block, .risk-block {
            border-radius: 0 var(--radius) var(--radius) 0;
            padding: 12px 13px;
            margin: 12px 0;
            font-size: 13px;
            line-height: 1.6;
        }

        .fact-block {
            background: var(--bg-fact);
            border-left: 3px solid var(--fact);
        }

        .advice-block {
            background: var(--bg-advice);
            border-left: 3px solid var(--advice);
        }

        .risk-block {
            background: var(--bg-risk);
            border-left: 3px solid var(--risk);
        }

        .block-title {
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .profile-panel, .source-dock, .intent-panel {
            padding: 14px 15px;
            margin-bottom: 13px;
        }

        .panel-title {
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-soft);
            padding-bottom: 7px;
            margin-bottom: 10px;
            font-size: 13px;
            font-weight: 700;
        }

        .profile-row, .debug-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: var(--text-secondary);
            font-size: 12px;
            line-height: 1.45;
            padding: 4px 0;
        }

        .profile-row strong, .debug-row strong {
            color: var(--text-primary);
            font-weight: 650;
            text-align: right;
            max-width: 68%;
            word-break: break-word;
        }

        .profile-progress {
            height: 6px;
            border-radius: 999px;
            background: var(--border-soft);
            overflow: hidden;
            margin: 9px 0 10px;
        }

        .profile-progress span {
            display: block;
            height: 100%;
            background: var(--text-primary);
        }

        .dock-note {
            color: var(--text-muted);
            font-size: 12px;
            line-height: 1.55;
        }

        .evidence-quote {
            color: var(--text-secondary);
            border-left: 2px solid #cbd5e1;
            padding-left: 10px;
            margin: 8px 0 10px;
            font-size: 12px;
            line-height: 1.58;
            white-space: pre-wrap;
        }

        div[data-testid="stForm"] {
            background: var(--bg-card);
            border: 1px solid var(--border-crisp);
            border-radius: var(--radius);
            box-shadow: var(--shadow-card);
            padding: 12px 13px 10px;
            margin-top: 14px;
        }

        div[data-testid="stButton"] > button {
            border-radius: 4px !important;
            border: 1px solid var(--border-crisp);
            background: var(--bg-card);
            color: var(--text-primary);
            min-height: 34px;
            box-shadow: none;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #94a3b8;
            background: var(--bg-subtle);
            color: var(--text-primary);
        }

        div[data-testid="stButton"] > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: var(--text-primary);
            border-color: var(--text-primary);
            color: white !important;
        }

        div[data-testid="stButton"] > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] * {
            color: white !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div {
            border-radius: 4px !important;
            border-color: var(--border-crisp) !important;
            background: white !important;
            color: var(--text-primary) !important;
            font-size: 13px !important;
        }

        label, [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary);
        }

        .stTextInput label, .stTextArea label, .stSelectbox label {
            color: var(--text-secondary) !important;
            font-size: 12px !important;
            font-weight: 650 !important;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding: .75rem .8rem 1.5rem !important;
            }
            .top-bar {
                align-items: flex-start;
                flex-direction: column;
            }
            .answer-card, .empty-workspace {
                padding: 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "sources": [], "intent": None}
        ]
    st.session_state.setdefault("pending_prompt", None)
    for key in PROFILE_FIELDS:
        st.session_state.setdefault(f"profile_{key}", "")


def render_top_bar(index, chat_client, profile: dict[str, str]) -> None:
    file_count = index.manifest.get("file_count", 0)
    chunk_count = index.manifest.get("chunk_count", 0)
    model_state = "可生成回答" if chat_client.available else "仅显示检索"
    profile_hint = profile.get("major") or profile.get("category") or profile.get("year") or "未设定画像"
    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <div class="product-title">🎓 2025级培养计划学业问答助手</div>
                <div class="product-subtitle">Asymmetric Academic Workspace · 本地 PDF 知识库</div>
            </div>
            <div class="top-tags">
                <span class="top-tag">PDF {html.escape(str(file_count))}</span>
                <span class="top-tag">片段 {html.escape(str(chunk_count))}</span>
                <span class="top-tag">模型 {html.escape(model_state)}</span>
                <span class="top-tag">👤 {html.escape(profile_hint)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_timeline() -> None:
    st.markdown('<div class="workspace-label">Main Timeline</div>', unsafe_allow_html=True)
    real_messages = [
        message for message in st.session_state.messages if message.get("content") != WELCOME_MESSAGE
    ]
    if not real_messages:
        render_empty_workspace()
        return

    for message in real_messages:
        role = message.get("role", "")
        content = str(message.get("content", ""))
        if role == "user":
            st.markdown(
                f'<div class="user-query">▶ Q: {html.escape(content)}</div>',
                unsafe_allow_html=True,
            )
        elif role == "assistant":
            time_label = html.escape(str(message.get("time") or "AI 学业顾问报告"))
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="card-timestamp">{time_label}</div>
                    {semantic_answer_html(content)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_workspace() -> None:
    st.markdown(
        """
        <div class="empty-workspace">
            <div class="workspace-label">Ready</div>
            <div class="empty-title">你好，我是你的 AI 学业顾问。</div>
            <div class="empty-copy">
                已加载 2025 级培养计划资料。你可以直接问课程、学分、专业方向、建议修读学期，
                我会把培养计划明文依据和基于画像的建议分开呈现。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_command_center() -> str | None:
    st.markdown('<div class="workspace-label">Quick Prompts</div>', unsafe_allow_html=True)
    columns = st.columns(len(EXAMPLE_PROMPTS))
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with columns[index]:
            if st.button(prompt, key=f"example_prompt_{index}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

    with st.form("command_center", clear_on_submit=True):
        prompt = st.text_area(
            "键入你的学业疑问",
            key="command_prompt",
            placeholder="例如：我是计算机专业大一第二学期，已修高数A1，这学期重点关注哪些课？",
            height=82,
        )
        submitted = st.form_submit_button("开始分析", type="primary", use_container_width=True)
    if submitted and prompt.strip():
        return prompt.strip()
    return None


def run_answer_question(prompt: str, profile: dict[str, str], index, provider, chat_client) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("正在检索培养计划并生成学业诊断..."):
        analysis, results = enhanced_search(index, provider, prompt, top_k=8)
        sources = source_payload(results)
        category_context = build_category_context(index, prompt, analysis)
        course_context = build_course_context(index, prompt, analysis, results)
        base_context = build_context(results)
        context = "\n\n".join(part for part in [category_context, course_context, base_context] if part)
        student_context = format_student_context(profile)

    if chat_client.available:
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages
            if item["role"] in {"user", "assistant"}
        ]
        messages = build_messages(
            prompt,
            context,
            history,
            intent_context=analysis,
            student_context=student_context,
        )
        try:
            answer = chat_client.complete(messages)
        except Exception as exc:
            answer = f"调用模型失败：{exc}\n\n下面先展示检索到的培养计划来源。"
    else:
        answer = fallback_answer(prompt, has_results=bool(results))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "intent": analysis.display_dict(),
            "time": "AI 学业顾问报告 · " + datetime.now().strftime("%H:%M"),
        }
    )


def render_profile_inspector(config, provider, chat_client) -> dict[str, str]:
    profile = current_profile()
    filled = sum(1 for value in profile.values() if value.strip())
    width = int(filled / len(PROFILE_FIELDS) * 100)
    tags = [
        profile.get("year") or "年级未填",
        profile.get("category") or profile.get("major") or "专业未填",
        profile.get("load") or "压力偏好未填",
    ]
    st.markdown(
        f"""
        <div class="profile-panel">
            <div class="panel-title">👤 学生当前画像</div>
            <div class="profile-tags">{"".join(f'<span class="profile-tag">{html.escape(tag)}</span>' for tag in tags)}</div>
            <div class="profile-progress"><span style="width:{width}%"></span></div>
            <div class="profile-row"><span>完整度</span><strong>{filled}/{len(PROFILE_FIELDS)}</strong></div>
            <div class="profile-row"><span>模型状态</span><strong>{'已连接' if chat_client.available else '仅检索'}</strong></div>
            <div class="profile-row"><span>向量</span><strong>{'本地' if config.uses_local_embeddings else html.escape(provider.signature)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        if st.button("新对话", type="primary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE, "sources": [], "intent": None}
            ]
            st.rerun()
    with right:
        if st.button("清空画像", use_container_width=True):
            for key in PROFILE_FIELDS:
                st.session_state[f"profile_{key}"] = ""
            st.rerun()

    return current_profile()


def render_profile_controls(config, provider) -> None:
    st.markdown('<div class="workspace-label">Profile Controls</div>', unsafe_allow_html=True)
    st.text_input("年级", key="profile_year", placeholder="大一 / 大二 / 2025级")
    st.text_input("招生大类", key="profile_category", placeholder="工科试验班(智能化制造类)")
    st.text_input("专业", key="profile_major", placeholder="计算机科学与技术")
    st.text_input("当前学期", key="profile_term", placeholder="大一第二学期")
    st.text_area("已修课程", key="profile_completed", placeholder="高数A(1)、线性代数B", height=64)
    st.text_area("兴趣方向", key="profile_interests", placeholder="机器人、算法、嵌入式", height=64)
    st.text_area("强弱项", key="profile_strengths", placeholder="数学强，物理一般", height=64)
    st.selectbox(
        "学分压力偏好",
        ["", "稳一点", "均衡", "可以挑战", "希望轻松"],
        key="profile_load",
    )

    if st.button("重建索引", use_container_width=True):
        load_or_build_index.clear()
        with st.spinner("正在重建索引..."):
            build_index(config.corpus_dir, config.index_dir, provider)
        st.success("索引已重建")
        st.rerun()


def render_evidence_dock(message: dict[str, Any] | None) -> None:
    sources = (message or {}).get("sources") or []
    analysis = (message or {}).get("intent")
    if not sources:
        st.markdown(
            '<div class="source-dock">'
            '<div class="panel-title">📄 当前回答依据</div>'
            '<div class="dock-note">提出一个问题后，这里会固定显示文件、页码、章节与原文片段。</div>'
            "</div>"
            '<div class="intent-panel">'
            '<div class="panel-title">🔎 检索复盘</div>'
            '<div class="dock-note">意图、实体和检索改写会在回答生成后展示。</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    source_html = []
    for source in sources[:5]:
        source_html.append(
            '<div class="source-pills">'
            f'<span class="source-pill">{html.escape(str(source["id"]))}</span>'
            f'<span class="source-pill">p.{html.escape(str(source["page"]))}</span>'
            "</div>"
            f'<div class="debug-row"><span>文件</span><strong>{html.escape(str(source["file"]))}</strong></div>'
            f'<div class="debug-row"><span>章节</span><strong>{html.escape(str(source["section"]))}</strong></div>'
            f'<div class="evidence-quote">{html.escape(str(source["text"])[:520])}</div>'
        )
    st.markdown(
        '<div class="source-dock">'
        '<div class="panel-title">📄 当前回答依据</div>'
        f'{"".join(source_html)}'
        "</div>",
        unsafe_allow_html=True,
    )
    render_intent_panel(analysis)


def render_intent_panel(analysis: dict[str, Any] | None) -> None:
    if not analysis:
        return
    entities = analysis.get("entities") or []
    queries = analysis.get("search_queries") or []
    entity_text = "、".join(
        f"{entity.get('kind')}:{entity.get('name')}" for entity in entities[:4]
    ) or "未识别"
    query_text = queries[0] if queries else "无"
    st.markdown(
        '<div class="intent-panel">'
        '<div class="panel-title">🔎 检索复盘</div>'
        f'<div class="debug-row"><span>识别意图</span><strong>{html.escape(str(analysis.get("intent_label") or analysis.get("intent")))}</strong></div>'
        f'<div class="debug-row"><span>置信度</span><strong>{html.escape(str(analysis.get("confidence")))}</strong></div>'
        f'<div class="debug-row"><span>实体</span><strong>{html.escape(entity_text)}</strong></div>'
        f'<div class="evidence-quote">{html.escape(query_text)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


SECTION_ALIASES = {
    "fact": ("培养计划明文依据", "培养计划明确写明", "资料明确写明", "明文依据", "官方依据"),
    "advice": ("基于画像与资料的建议", "基于资料与学生情况的建议", "规划建议", "个性化建议", "学业建议", "推荐方案"),
    "risk": ("注意风险与下一步", "注意风险", "风险提示", "资料不足", "下一步"),
}


def semantic_answer_html(answer: str) -> str:
    sections = split_answer_sections(answer)
    has_structured_sections = any(sections[key] for key in ("fact", "advice", "risk"))
    blocks: list[str] = []

    if sections["lead"]:
        blocks.append(f'<div class="answer-body">{markdownish_to_html("\n".join(sections["lead"]))}</div>')

    if has_structured_sections:
        blocks.extend(
            render_answer_block(key, "\n".join(lines))
            for key, lines in sections.items()
            if key in {"fact", "advice", "risk"} and lines
        )
        return "\n".join(blocks)

    inferred_key = infer_answer_kind(answer)
    if inferred_key:
        return render_answer_block(inferred_key, answer)
    return f'<div class="answer-body">{markdownish_to_html(answer)}</div>'


def split_answer_sections(answer: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"lead": [], "fact": [], "advice": [], "risk": []}
    current = "lead"
    for raw_line in answer.strip().splitlines():
        section_key = classify_section_heading(raw_line)
        if section_key:
            current = section_key
            continue
        sections[current].append(raw_line)
    return {key: trim_blank_lines(lines) for key, lines in sections.items()}


def classify_section_heading(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line:
        return None
    clean = re.sub(r"^\s{0,3}#{1,4}\s+", "", line)
    clean = re.sub(r"^\d+[.、]\s*", "", clean)
    clean = clean.strip(" -*_【】[]：:")
    heading_like = line.startswith("#") or len(clean) <= 28
    if not heading_like:
        return None
    for key, aliases in SECTION_ALIASES.items():
        if any(alias in clean for alias in aliases):
            return key
    return None


def trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def infer_answer_kind(answer: str) -> str | None:
    if any(term in answer for term in ["实时", "无法判断", "未找到明确依据", "以教务系统", "注意"]):
        return "risk"
    if any(term in answer for term in ["建议", "推荐", "适合", "方案"]):
        return "advice"
    if any(term in answer for term in ["培养计划", "资料明确", "[S", "【S", "学分", "课程"]):
        return "fact"
    return None


def render_answer_block(kind: str, text: str) -> str:
    title_map = {
        "fact": "📌 培养计划明文依据",
        "advice": "💡 基于画像与资料的建议",
        "risk": "⚠ 注意风险与下一步",
    }
    class_map = {"fact": "fact-block", "advice": "advice-block", "risk": "risk-block"}
    return (
        f'<div class="{class_map[kind]}">'
        f'<div class="block-title">{title_map[kind]}</div>'
        f"{markdownish_to_html(text)}"
        "</div>"
    )


def markdownish_to_html(text: str) -> str:
    lines = text.strip().splitlines()
    html_lines: list[str] = []
    list_tag: str | None = None
    index = 0

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            html_lines.append(f"</{list_tag}>")
            list_tag = None

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if not line:
            close_list()
            index += 1
            continue
        if line.startswith("|"):
            close_list()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            html_lines.append(markdown_table_to_html(table_lines))
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            close_list()
            level = "h2" if len(heading.group(1)) <= 2 else "h3"
            html_lines.append(f"<{level}>{format_inline(heading.group(2))}</{level}>")
            index += 1
            continue
        if re.fullmatch(r"-{3,}", line):
            close_list()
            html_lines.append("<hr>")
            index += 1
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            if list_tag != "ul":
                close_list()
                html_lines.append("<ul>")
                list_tag = "ul"
            html_lines.append(f"<li>{format_inline(bullet.group(1))}</li>")
            index += 1
            continue
        numbered = re.match(r"^\d+[.、]\s+(.+)$", line)
        if numbered:
            if list_tag != "ol":
                close_list()
                html_lines.append("<ol>")
                list_tag = "ol"
            html_lines.append(f"<li>{format_inline(numbered.group(1))}</li>")
            index += 1
            continue
        close_list()
        html_lines.append(f"<p>{format_inline(line)}</p>")
        index += 1
    close_list()
    return "\n".join(html_lines)


def markdown_table_to_html(lines: list[str]) -> str:
    rows = [split_table_row(line) for line in lines]
    rows = [row for row in rows if row]
    if not rows:
        return ""
    has_separator = len(rows) > 1 and all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in rows[1])
    header = rows[0] if has_separator else []
    body_rows = rows[2:] if has_separator else rows
    parts = ['<table class="answer-table">']
    if header:
        parts.append("<thead><tr>")
        parts.extend(f"<th>{format_inline(cell)}</th>" for cell in header)
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in body_rows:
        parts.append("<tr>")
        parts.extend(f"<td>{format_inline(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_inline(value: str) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def latest_assistant_message() -> dict[str, Any] | None:
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant" and message.get("sources"):
            return message
    return None


def current_profile() -> dict[str, str]:
    return {
        key: str(st.session_state.get(f"profile_{key}", "") or "").strip()
        for key in PROFILE_FIELDS
    }


def format_student_context(profile: dict[str, str]) -> str:
    lines = []
    for key, label in PROFILE_FIELDS.items():
        value = profile.get(key, "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
