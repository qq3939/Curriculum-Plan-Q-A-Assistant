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
from planqa.courses import build_course_context, find_course_records
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

    records = find_course_records(index, majors=["计算机科学与技术"], term_prefixes=["二/1", "二/2"])
    names = {record.name for record in records}
    expected_names = {
        "离散数学",
        "JAVA编程与开发",
        "概率论与数理统计B",
        "数据结构",
        "大学物理实验(1)",
        "数据结构实验",
        "JAVA编程与开发实验",
        "数据库原理(双语)",
        "计算机组成",
        "操作系统D",
        "计算机网络",
        "数据库原理实验",
        "计算机组成实验",
        "计算机网络实验",
        "操作系统实验",
        "工程认识实习",
        "电子实习A",
    }
    missing = sorted(expected_names - names)
    assert not missing, f"Missing CS second-year courses: {missing}"
    assert len(records) == len(expected_names), f"Unexpected CS second-year course count: {len(records)}"
    assert {record.page for record in records} == {159, 160}

    question = "计算机大二会学哪些课程"
    analysis, results = enhanced_search(index, provider, question, top_k=8)
    context = build_course_context(index, question, analysis, results)
    assert "结构化课程表辅助信息" in context
    assert "离散数学" in context
    assert "数据库原理(双语)" in context
    assert "二/1" in context and "二/2" in context
    assert "p.159" in context and "p.160" in context

    print("Course table extraction checks passed.")
    print(f"计算机科学与技术大二课程数: {len(records)}")
    for record in records:
        print(f"- {record.term} {record.name} {record.credits}学分 {record.group} p.{record.page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
