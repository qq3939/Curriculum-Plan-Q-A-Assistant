from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.embeddings import OpenAIEmbeddingProvider
from planqa.llm import OpenAICompatibleChatClient


def main() -> int:
    config = load_config(ROOT)
    if not config.has_api_key:
        print("OPENAI_API_KEY is not configured.")
        return 2

    print(f"Base URL: {config.base_url}")
    print(f"Chat model: {config.chat_model}")
    print(f"Embedding model: {config.embedding_model}")

    if config.uses_local_embeddings:
        print("Embedding check skipped: using local-hash retrieval vectors.")
    else:
        embedding_provider = OpenAIEmbeddingProvider(config)
        vectors = embedding_provider.embed_texts(["培养计划学业问答助手 API 连通性检查"])
        if vectors.shape[0] != 1:
            raise RuntimeError("Embedding API returned an unexpected result.")
        print(f"Embedding check passed: dim={vectors.shape[1]}")

    chat_client = OpenAICompatibleChatClient(config)
    answer = chat_client.complete(
        [
            {"role": "system", "content": "你只需要简短回答。"},
            {"role": "user", "content": "请回复：API已接通"},
        ],
        temperature=0,
    )
    print(f"Chat check passed: {answer[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
