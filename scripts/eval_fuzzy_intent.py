from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planqa.config import load_config
from planqa.embeddings import make_embedding_provider
from planqa.intent import enhanced_search
from planqa.pdf_index import build_index, is_index_current, load_index


FUZZY_CASES = [
    {
        "question": "计科大二要修啥，别太官方",
        "must_match": ["计算机科学与技术"],
        "expected_intents": {"course_plan", "academic_advice", "major_profile"},
    },
    {
        "question": "公共课一共得拿多少分",
        "must_match": ["通识教育课程", "42.5"],
        "expected_intents": {"credit_requirement", "course_plan", "general_qa"},
    },
    {
        "question": "智能制造类，喜欢机器人和写代码，分流选啥更贴",
        "must_match": ["机器人工程"],
        "expected_intents": {"academic_advice", "admission_category", "major_profile"},
    },
    {
        "question": "毕业到底要修够几分",
        "must_match": ["学分"],
        "expected_intents": {"credit_requirement"},
    },
    {
        "question": "工科大类有哪些专业能选",
        "must_match": ["工科试验班", "涵盖专业"],
        "expected_intents": {"admission_category"},
    },
]


def main() -> int:
    config = load_config(ROOT)
    provider = make_embedding_provider(config)
    if not is_index_current(config.corpus_dir, config.index_dir, provider.signature):
        index = build_index(config.corpus_dir, config.index_dir, provider)
    else:
        index = load_index(config.index_dir)

    for case in FUZZY_CASES:
        analysis, results = enhanced_search(index, provider, case["question"], top_k=6)
        assert results, f"No results for fuzzy question: {case['question']}"
        joined = "\n".join(
            f"{result.chunk.source_file} {result.chunk.section_title} {result.chunk.text[:800]}" for result in results
        )
        missing = [needle for needle in case["must_match"] if needle not in joined]
        assert not missing, f"{case['question']} missing expected signals: {missing}"
        assert analysis.intent in case["expected_intents"], (
            f"{case['question']} intent={analysis.intent}, expected one of {case['expected_intents']}"
        )
        print(f"\nQ: {case['question']}")
        print(f"  intent={analysis.intent} confidence={analysis.confidence:.2f}")
        print(f"  queries={analysis.search_queries[:3]}")
        for result in results[:3]:
            print(f"  {result.score:.4f} {result.chunk.source_file} p.{result.chunk.page} {result.chunk.section_title}")
    print("\nFuzzy intent checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
