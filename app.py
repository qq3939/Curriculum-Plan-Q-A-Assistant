from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
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

    with st.sidebar:
        profile = render_sidebar(config, provider, chat_client)

    try:
        index = load_or_build_index(str(ROOT), provider.signature)
    except Exception as exc:
        st.error(f"索引加载失败：{exc}")
        return

    render_shell_header(index, chat_client)
    pending_prompt = st.session_state.pop("pending_prompt", None)
    render_chat_history()
    if not pending_prompt:
        render_examples()

    prompt = pending_prompt or st.chat_input("向培养计划提问")
    if not prompt:
        return

    answer_question(prompt, profile, index, provider, chat_client)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-muted: #f7f7f8;
            --surface-rail: #f3f3f1;
            --line: #e4e4e7;
            --text: #202123;
            --muted: #6b7280;
            --accent: #10a37f;
            --accent-soft: #e7f6f1;
            --warning-soft: #fff7e6;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--surface-muted);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: var(--surface-rail);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--text);
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 1.1rem;
            padding-bottom: 8rem;
        }

        .app-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 10px 0 18px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 22px;
        }

        .app-title {
            font-size: 18px;
            font-weight: 650;
            letter-spacing: 0;
        }

        .app-subtitle {
            margin-top: 4px;
            color: var(--muted);
            font-size: 13px;
        }

        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .status-pill {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
            color: var(--muted);
            font-size: 12px;
            padding: 6px 9px;
            white-space: nowrap;
        }

        .status-pill strong {
            color: var(--text);
            font-weight: 600;
        }

        .rail-title {
            font-size: 17px;
            font-weight: 700;
            margin: 6px 0 16px;
        }

        .rail-note {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
            margin: -4px 0 12px;
        }

        .quick-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin: 6px 0 18px;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--text);
            min-height: 38px;
            transition: border-color .12s ease, background .12s ease;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
            color: var(--text);
        }

        div[data-testid="stButton"] > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: var(--text);
            border-color: var(--text);
            color: #fff !important;
        }

        div[data-testid="stButton"] > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] * {
            color: #fff !important;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            padding: 0.55rem 0;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 15px;
            line-height: 1.72;
        }

        [data-testid="stChatInput"] {
            max-width: 900px;
            margin: 0 auto;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 8px;
            border-color: var(--line);
            min-height: 52px;
        }

        .source-heading {
            color: var(--muted);
            font-size: 13px;
            font-weight: 650;
            margin-top: 12px;
            margin-bottom: 4px;
        }

        .source-meta {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 8px;
        }

        .profile-filled {
            border: 1px solid #cde8de;
            background: var(--accent-soft);
            border-radius: 8px;
            padding: 9px 10px;
            font-size: 12px;
            color: var(--text);
        }

        .profile-empty {
            border: 1px solid #f1ddae;
            background: var(--warning-soft);
            border-radius: 8px;
            padding: 9px 10px;
            font-size: 12px;
            color: var(--text);
        }

        @media (max-width: 760px) {
            .app-topbar {
                align-items: flex-start;
                flex-direction: column;
            }
            .status-row {
                justify-content: flex-start;
            }
            .quick-grid {
                grid-template-columns: 1fr;
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


def render_sidebar(config, provider, chat_client) -> dict[str, str]:
    st.markdown('<div class="rail-title">培养计划助手</div>', unsafe_allow_html=True)

    if st.button("新对话", type="primary", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "sources": [], "intent": None}
        ]
        st.rerun()

    if st.button("清空画像", use_container_width=True):
        for key in PROFILE_FIELDS:
            st.session_state[f"profile_{key}"] = ""
        st.rerun()

    st.divider()
    st.subheader("学生画像")
    st.markdown(
        '<div class="rail-note">只保存在当前浏览器会话，用来让推荐和选课建议更贴近你。</div>',
        unsafe_allow_html=True,
    )

    st.text_input("年级", key="profile_year", placeholder="例如：大一 / 大二 / 2025级")
    st.text_input("招生大类", key="profile_category", placeholder="例如：工科试验班(智能化制造类)")
    st.text_input("专业", key="profile_major", placeholder="例如：计算机科学与技术")
    st.text_input("当前学期", key="profile_term", placeholder="例如：大一第二学期")
    st.text_area("已修课程", key="profile_completed", placeholder="例如：高数A(1)、线性代数B", height=78)
    st.text_area("兴趣方向", key="profile_interests", placeholder="例如：机器人、算法、嵌入式", height=78)
    st.text_area("强弱项", key="profile_strengths", placeholder="例如：数学强，物理一般", height=78)
    st.selectbox(
        "学分压力偏好",
        ["", "稳一点", "均衡", "可以挑战", "希望轻松"],
        key="profile_load",
    )

    profile = current_profile()
    filled = sum(1 for value in profile.values() if value.strip())
    profile_class = "profile-filled" if filled >= 3 else "profile-empty"
    st.markdown(
        f'<div class="{profile_class}">画像完整度：{filled}/{len(PROFILE_FIELDS)}。'
        f' 推荐类问题通常至少需要年级、专业或大类、兴趣方向。</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("运行状态")
    pdf_count = len(list(config.corpus_dir.glob("*.pdf"))) if config.corpus_dir.exists() else 0
    st.write(f"资料：{pdf_count} 个 PDF")
    st.write(f"模型：{'已连接' if chat_client.available else '仅检索'}")
    st.write(f"向量：{'本地' if config.uses_local_embeddings else provider.signature}")

    left, right = st.columns(2)
    with left:
        if st.button("重建索引", use_container_width=True):
            load_or_build_index.clear()
            with st.spinner("正在重建索引..."):
                build_index(config.corpus_dir, config.index_dir, provider)
            st.success("索引已重建")
            st.rerun()
    with right:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE, "sources": [], "intent": None}
            ]
            st.rerun()

    return profile


def render_shell_header(index, chat_client) -> None:
    file_count = index.manifest.get("file_count", 0)
    chunk_count = index.manifest.get("chunk_count", 0)
    model_state = "可生成回答" if chat_client.available else "仅显示检索"
    st.markdown(
        f"""
        <div class="app-topbar">
            <div>
                <div class="app-title">培养计划学业问答助手</div>
                <div class="app-subtitle">2025 级培养计划 · 本地检索 · 会话内画像</div>
            </div>
            <div class="status-row">
                <div class="status-pill">资料 <strong>{html.escape(str(file_count))}</strong></div>
                <div class="status-pill">片段 <strong>{html.escape(str(chunk_count))}</strong></div>
                <div class="status-pill">模型 <strong>{html.escape(model_state)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_intent(message.get("intent"))
                render_sources(message.get("sources", []))


def render_examples() -> None:
    if len(st.session_state.messages) > 1:
        return

    st.markdown('<div class="quick-grid">', unsafe_allow_html=True)
    columns = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with columns[index % 2]:
            if st.button(prompt, key=f"example_prompt_{index}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def answer_question(prompt: str, profile: dict[str, str], index, provider, chat_client) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在查找培养计划依据..."):
            analysis, results = enhanced_search(index, provider, prompt, top_k=8)
            sources = source_payload(results)
            context = build_context(results)
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

        st.markdown(answer)
        intent_payload = analysis.display_dict()
        render_intent(intent_payload)
        render_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "intent": intent_payload}
        )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown('<div class="source-heading">来源</div>', unsafe_allow_html=True)
    for source in sources:
        label = f"[{source['id']}] {source['file']} · p.{source['page']} · {source['section']}"
        with st.expander(label):
            st.markdown(
                f"<div class=\"source-meta\">综合分数 {source['score']} · 向量 {source.get('vector_score')} · 关键词 {source.get('lexical_score')}</div>",
                unsafe_allow_html=True,
            )
            st.text(str(source["text"])[:2200])


def render_intent(analysis: dict[str, Any] | None) -> None:
    if not analysis:
        return
    with st.expander("检索详情"):
        st.write(f"识别意图：`{analysis.get('intent_label') or analysis.get('intent')}`")
        st.write(f"置信度：`{analysis.get('confidence')}`")

        entities = analysis.get("entities") or []
        if entities:
            st.write("匹配实体：")
            for entity in entities:
                st.write(
                    f"- {entity.get('kind')}: {entity.get('name')} "
                    f"(score={entity.get('score')}, by={entity.get('matched_by')})"
                )

        queries = analysis.get("search_queries") or []
        if queries:
            st.write("检索改写：")
            for query in queries:
                st.write(f"- {query}")


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
