from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.embeddings import LocalHashEmbeddingProvider
from planqa.pdf_index import build_index, is_index_current, load_index
from planqa.intent import enhanced_search


EXPECTED_PAGES = {
    "2025级本科专业大类与对应专业一览表.pdf": 1,
    "2025级本科培养计划-上册.pdf": 480,
    "2025级本科培养计划-下册.pdf": 515,
    "培养计划阅读指南.pdf": 3,
    "学科基础课程(大类阶段).pdf": 4,
    "通识教育课程.pdf": 4,
}

SAMPLES = [
    "通识教育课程最低要求多少学分？",
    "计算机科学与技术专业的培养目标是什么？",
    "工科试验班智能化制造类包含哪些专业？",
    "我是智能化制造类，喜欢计算机和机器人，推荐什么专业？",
    "我是计算机专业大一第二学期，已修高数A1，推荐这学期重点关注哪些课？",
    "2026级培养计划有什么变化？",
]


def main() -> int:
    config = load_config(ROOT)
    provider = LocalHashEmbeddingProvider()
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        bundle = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        bundle = load_index(config.index_dir)

    files = {item["name"]: item["pages"] for item in bundle.manifest["files"]}
    assert files == EXPECTED_PAGES, f"Unexpected PDF page map: {files}"
    assert bundle.manifest["file_count"] == 6
    assert bundle.manifest["chunk_count"] == len(bundle.chunks) > 0
    assert len(bundle.embeddings) == len(bundle.chunks)

    for chunk in bundle.chunks:
        assert chunk.source_file
        assert chunk.page >= 1
        assert chunk.section_title
        assert chunk.text

    print("Index checks passed.")
    for sample in SAMPLES:
        analysis, results = enhanced_search(bundle, provider, sample, top_k=3)
        assert results, f"No retrieval results for {sample}"
        print(f"\nQ: {sample}")
        print(f"  intent={analysis.intent} confidence={analysis.confidence:.2f}")
        for result in results:
            chunk = result.chunk
            snippet = " ".join(chunk.text.split())[:120]
            print(f"  score={result.score:.4f} {chunk.source_file} p.{chunk.page} {chunk.section_title}")
            print(f"  {snippet}")
    print("\nSample retrieval checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
