from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.embeddings import make_embedding_provider
from planqa.pdf_index import build_index


def main() -> int:
    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    print(f"Corpus: {config.corpus_dir}")
    print(f"Index: {config.index_dir}")
    print(f"Embedding provider: {provider.signature}")
    bundle = build_index(config.corpus_dir, config.index_dir, provider)
    print(f"Indexed files: {bundle.manifest['file_count']}")
    print(f"Indexed chunks: {bundle.manifest['chunk_count']}")
    for item in bundle.manifest["files"]:
        print(f"- {item['name']}: {item['pages']} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
