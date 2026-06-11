from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.embeddings import make_embedding_provider
from planqa.llm import OpenAICompatibleChatClient
from planqa.pdf_index import build_index, is_index_current, load_index
from planqa.prompts import build_messages, fallback_answer
from planqa.retrieval import build_context, search, source_payload


@st.cache_resource(show_spinner="正在读取培养计划索引...")
def load_or_build_index(root_dir: str, provider_signature: str):
    config = load_config(Path(root_dir))
    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider_signature):
        return build_index(config.corpus_dir, config.index_dir, provider)
    return load_index(config.index_dir)


def render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    st.markdown("**来源**")
    for source in sources:
        label = f"[{source['id']}] {source['file']} - p.{source['page']} - {source['section']}"
        with st.expander(label):
            st.caption(
                f"综合分数: {source['score']} | 向量: {source.get('vector_score')} | 关键词: {source.get('lexical_score')}"
            )
            st.text(str(source["text"]))


def main() -> None:
    st.set_page_config(page_title="培养计划学业问答助手", layout="wide")
    st.title("培养计划学业问答助手")

    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    chat_client = OpenAICompatibleChatClient(config)

    with st.sidebar:
        st.subheader("资料状态")
        st.write(f"资料目录: `{config.corpus_dir.name}`")
        if config.corpus_dir.exists():
            st.write(f"PDF 数量: {len(list(config.corpus_dir.glob('*.pdf')))}")
        else:
            st.error("未找到培养计划资料目录。")

        st.subheader("模型状态")
        if config.has_api_key:
            st.success("已配置 API Key")
            st.write(f"Chat: `{config.chat_model}`")
            st.write(f"Embedding: `{config.embedding_model}`")
        else:
            st.warning("未配置 API Key，当前仅显示检索片段。")

        if st.button("重建索引"):
            load_or_build_index.clear()
            with st.spinner("正在重建索引..."):
                build_index(config.corpus_dir, config.index_dir, provider)
            st.success("索引已重建。")
            st.rerun()

        if st.button("清空会话"):
            st.session_state.messages = []
            st.rerun()

    try:
        index = load_or_build_index(str(ROOT), provider.signature)
    except Exception as exc:
        st.error(f"索引加载失败: {exc}")
        return

    with st.sidebar:
        st.subheader("索引状态")
        st.write(f"文件数: {index.manifest.get('file_count')}")
        st.write(f"片段数: {index.manifest.get('chunk_count')}")
        st.write(f"索引方式: `{index.manifest.get('embedding_signature')}`")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "你好，我可以根据当前培养计划回答学业问题。推荐类问题请告诉我年级、"
                    "招生大类或专业、当前学期、已修课程和兴趣方向。"
                ),
                "sources": [],
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    prompt = st.chat_input("请输入你的培养计划或学业规划问题")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在查找培养计划依据..."):
            results = search(index, provider, prompt, top_k=6)
            sources = source_payload(results)
            context = build_context(results)

        if chat_client.available:
            history = [
                {"role": item["role"], "content": item["content"]}
                for item in st.session_state.messages
                if item["role"] in {"user", "assistant"}
            ]
            messages = build_messages(prompt, context, history)
            try:
                answer = chat_client.complete(messages)
            except Exception as exc:
                answer = f"调用模型失败: {exc}\n\n下面先展示检索到的培养计划来源。"
        else:
            answer = fallback_answer(prompt, has_results=bool(results))

        st.markdown(answer)
        render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
