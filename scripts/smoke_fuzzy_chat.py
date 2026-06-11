from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.courses import build_course_context
from planqa.embeddings import make_embedding_provider
from planqa.intent import enhanced_search
from planqa.llm import OpenAICompatibleChatClient
from planqa.pdf_index import build_index, is_index_current, load_index
from planqa.prompts import build_messages
from planqa.retrieval import build_context, source_payload


def main() -> int:
    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        index = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        index = load_index(config.index_dir)

    question = "计算机大二会学哪些课程"
    analysis, results = enhanced_search(index, provider, question, top_k=5)
    joined_sources = "\n".join(f"{result.chunk.section_title} {result.chunk.text[:400]}" for result in results)
    if "计算机科学与技术" not in joined_sources or "二/1" not in joined_sources or "二/2" not in joined_sources:
        raise RuntimeError("Fuzzy retrieval did not find 计算机科学与技术 second-year course-table rows.")
    course_context = build_course_context(index, question, analysis, results)
    context = "\n\n".join(part for part in [course_context, build_context(results)] if part)
    messages = build_messages(question, context, chat_history=[], intent_context=analysis)
    answer = OpenAICompatibleChatClient(config).complete(messages)
    print("Question:", question)
    print(f"Intent: {analysis.intent} confidence={analysis.confidence:.2f}")
    print("Queries:", analysis.search_queries[:4])
    print("Answer:", answer[:900])
    print("Sources:")
    for source in source_payload(results):
        print(f"- {source['id']} {source['file']} p.{source['page']} {source['section']}")
    if "数据结构" not in answer or "二/1" not in answer or "二/2" not in answer:
        raise RuntimeError("Fuzzy answer did not list second-year computer-science courses.")
    if "理论（3门" in answer or "理论课3门" in answer or "理论3门" in answer:
        raise RuntimeError("Fuzzy answer miscounted the second-year theory courses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
