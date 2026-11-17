from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from hermes_agent import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDING_CACHE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    load_documents,
    read_text_file,
    resolve_docs_dir,
    retrieve,
)


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("evaluation_questions.json")


def normalized_name(value: str) -> str:
    return unicodedata.normalize("NFKC", Path(value).stem).lower()


def evaluate(
    mode: str,
    questions: list[dict],
    chunks,
    top_k: int,
    cache_path: Path,
    embedding_model: str,
) -> dict:
    hit_counts = {1: 0, 3: 0, 5: 0}
    evidence_hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    evidence_reciprocal_rank_sum = 0.0
    rows = []
    for item in questions:
        results = retrieve(
            item["question"],
            chunks,
            limit=top_k,
            search_mode=mode,
            cache_path=cache_path,
            embedding_model=embedding_model,
        )
        actual = [chunk.file_path.name for chunk in results]
        expected = {normalized_name(name) for name in item["expected_files"]}
        relevant_rank = next(
            (
                rank
                for rank, name in enumerate(actual, start=1)
                if normalized_name(name) in expected
            ),
            None,
        )
        for cutoff in hit_counts:
            hit_counts[cutoff] += int(
                relevant_rank is not None and relevant_rank <= cutoff
            )
        if relevant_rank is not None:
            reciprocal_rank_sum += 1 / relevant_rank
        evidence_terms = [
            unicodedata.normalize("NFKC", term).lower()
            for term in item.get("evidence_terms", [])
        ]
        evidence_rank = None
        if evidence_terms:
            for cutoff in range(1, len(results) + 1):
                context_text = unicodedata.normalize(
                    "NFKC",
                    "\n".join(chunk.text for chunk in results[:cutoff]),
                ).lower()
                if all(term in context_text for term in evidence_terms):
                    evidence_rank = cutoff
                    break
            for cutoff in evidence_hit_counts:
                evidence_hit_counts[cutoff] += int(
                    evidence_rank is not None and evidence_rank <= cutoff
                )
            if evidence_rank is not None:
                evidence_reciprocal_rank_sum += 1 / evidence_rank
        rows.append(
            {
                "question": item["question"],
                "category": item.get("category", "未分類"),
                "rank": relevant_rank,
                "evidence_rank": evidence_rank,
                "expected": item["expected_files"],
                "actual": actual,
            }
        )
    return {
        "mode": mode,
        "total": len(questions),
        "hit_at": {
            str(cutoff): hit_counts[cutoff] / len(questions) if questions else 0.0
            for cutoff in hit_counts
        },
        "mrr": reciprocal_rank_sum / len(questions) if questions else 0.0,
        "evidence_hit_at": {
            str(cutoff): (
                evidence_hit_counts[cutoff] / len(questions) if questions else 0.0
            )
            for cutoff in evidence_hit_counts
        },
        "evidence_mrr": (
            evidence_reciprocal_rank_sum / len(questions) if questions else 0.0
        ),
        "results": rows,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="会議ログ検索のTop-k命中率を比較する")
    parser.add_argument(
        "--mode",
        choices=[
            "keyword",
            "local",
            "hybrid",
            "gbrain",
            "enhanced",
            "both",
            "all",
        ],
        default="both",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="集計値だけを表示する",
    )
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_PATH,
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    args = parser.parse_args()

    questions = json.loads(read_text_file(args.questions))
    chunks = load_documents(resolve_docs_dir(args.docs_dir, args.config))
    if args.mode == "both":
        modes = ["keyword", "enhanced"]
    elif args.mode == "all":
        modes = ["keyword", "local", "hybrid", "gbrain", "enhanced"]
    else:
        modes = [args.mode]
    reports = [
        evaluate(
            mode,
            questions,
            chunks,
            args.top_k,
            args.embedding_cache,
            args.embedding_model,
        )
        for mode in modes
    ]

    for report in reports:
        print(
            f"\n[{report['mode']}] "
            f"File Top-1/3/5="
            f"{report['hit_at']['1']:.1%}/"
            f"{report['hit_at']['3']:.1%}/"
            f"{report['hit_at']['5']:.1%} "
            f"Evidence Top-1/3/5="
            f"{report['evidence_hit_at']['1']:.1%}/"
            f"{report['evidence_hit_at']['3']:.1%}/"
            f"{report['evidence_hit_at']['5']:.1%} "
            f"File MRR={report['mrr']:.3f} "
            f"Evidence MRR={report['evidence_mrr']:.3f}"
        )
        if args.summary_only:
            continue
        for row in report["results"]:
            status = f"rank={row['rank']}" if row["rank"] else "MISS"
            evidence_status = (
                f"evidence={row['evidence_rank']}"
                if row["evidence_rank"]
                else "evidence=MISS"
            )
            print(
                f"- {status}, {evidence_status} "
                f"[{row['category']}]: {row['question']}"
            )
            print(f"  上位: {', '.join(row['actual']) or 'なし'}")


if __name__ == "__main__":
    main()
