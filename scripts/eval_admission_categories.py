from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.categories import build_category_context, get_admission_categories
from planqa.config import load_config
from planqa.embeddings import LocalHashEmbeddingProvider
from planqa.intent import enhanced_search
from planqa.pdf_index import build_index, is_index_current, load_index


def main() -> int:
    config = load_config(ROOT)
    provider = LocalHashEmbeddingProvider()
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        index = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        index = load_index(config.index_dir)

    categories = get_admission_categories(index)
    by_name = {category.name: category for category in categories}
    target = by_name["工科试验班(智能化制造类)"]
    assert len(target.majors) == 20
    assert "机器人工程" in target.majors
    assert "机械设计制造及其自动化" in target.majors
    assert "自动化" not in target.majors
    assert "计算机科学与技术" not in target.majors
    assert "人工智能" not in target.majors

    question = "我是智能化制造类，喜欢计算机和机器人，推荐什么专业？"
    analysis, results = enhanced_search(index, provider, question, top_k=8)
    context = build_category_context(index, question, analysis)
    assert "结构化招生大类辅助信息" in context
    assert "机器人工程" in context
    assert "工科试验班(智能化制造类)" in context
    assert "自动化 属于 工科试验班(电子与信息类)" in context

    print("Admission category checks passed.")
    print(f"{target.name}: {len(target.majors)} majors")
    print("、".join(target.majors))
    print("Context preview:")
    print(context[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
