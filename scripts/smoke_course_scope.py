from __future__ import annotations

import sys
from pathlib import Path

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
from planqa.retrieval import build_context


def main() -> int:
    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        index = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        index = load_index(config.index_dir)

    question = "每个专业大二（第3、4学期）的完整课程清单是什么？培养计划里面不是写得很明白吗"
    analysis, results = enhanced_search(index, provider, question, top_k=8)
    context = "\n\n".join(
        part
        for part in [
            build_category_context(index, question, analysis),
            build_course_context(index, question, analysis, results),
            build_context(results),
        ]
        if part
    )
    messages = build_messages(question, context, chat_history=[], intent_context=analysis)
    answer = OpenAICompatibleChatClient(config).complete(messages)

    print("Question:", question)
    print(f"Intent: {analysis.intent} confidence={analysis.confidence:.2f}")
    print("Answer:", answer[:1200])

    if "培养计划没有标注" in answer or "没有标注每门课程" in answer:
        raise RuntimeError("Scope answer incorrectly claimed the curriculum plan lacks term labels.")
    if "建议修读学年学期" not in answer:
        raise RuntimeError("Scope answer did not mention the curriculum plan's term label.")
    if not any(term in answer for term in ["范围", "专业", "招生大类", "分批", "指定"]):
        raise RuntimeError("Scope answer did not ask the user to narrow or batch the course-list request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
