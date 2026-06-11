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

    question = "通识教育课程最低要求多少学分？请给出来源。"
    analysis, results = enhanced_search(index, provider, question, top_k=4)
    context = build_context(results)
    messages = build_messages(question, context, chat_history=[], intent_context=analysis)
    answer = OpenAICompatibleChatClient(config).complete(messages)
    print("Question:", question)
    print(f"Intent: {analysis.intent} confidence={analysis.confidence:.2f}")
    print("Answer:", answer[:800])
    print("Sources:")
    for source in source_payload(results):
        print(f"- {source['id']} {source['file']} p.{source['page']} {source['section']}")
    if "42.5" not in answer:
        raise RuntimeError("Smoke answer did not include the expected 42.5 credit value.")
    if "差额" in answer or "可能存在" in answer or "未列出" in answer:
        raise RuntimeError("Smoke answer added unsupported credit-subtotal speculation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
