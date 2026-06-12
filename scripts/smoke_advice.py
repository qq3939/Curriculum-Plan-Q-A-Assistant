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

    question = "我是智能化制造类，喜欢计算机和机器人，推荐什么专业？"
    analysis, results = enhanced_search(index, provider, question, top_k=8)
    category_context = build_category_context(index, question, analysis)
    context = "\n\n".join(part for part in [category_context, build_context(results)] if part)
    messages = build_messages(question, context, chat_history=[], intent_context=analysis)
    answer = OpenAICompatibleChatClient(config).complete(messages)

    print("Question:", question)
    print(f"Intent: {analysis.intent} confidence={analysis.confidence:.2f}")
    print("Answer:", answer[:1200])
    print("Sources:")
    for source in source_payload(results):
        print(f"- {source['id']} {source['file']} p.{source['page']} {source['section']}")

    if "机器人工程" not in answer:
        raise RuntimeError("Advice answer did not recommend 机器人工程 for the stated interests.")
    forbidden_phrases = [
        "推荐自动化",
        "首选自动化",
        "推荐计算机科学与技术",
        "推荐人工智能",
        "首选人工智能",
        "往年经验",
        "分流竞争压力",
        "录取分数",
        "热门程度",
    ]
    if any(phrase in answer for phrase in forbidden_phrases):
        raise RuntimeError("Advice answer recommended a major outside 智能化制造类.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
