from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from hermes_agent import (
    DEFAULT_CONFIG_PATH, DEFAULT_GBRAIN_HOME, SEARCH_LOGGER,
    _PARALLEL_GBRAIN_SLOT,
    is_advice_question, load_documents, normalize_text, prepare_advice_search,
    read_text_file, resolve_docs_dir, retrieve_gbrain, retrieve_keyword, retrieve_parallel,
)
from slack_bot import SlackBotSettings, SlackHermesBot


class SearchStatus(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.status = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.status = record.getMessage()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and bounded parallel GBrain search")
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--gbrain-home", type=Path, default=Path(os.getenv("GBRAIN_HOME", str(DEFAULT_GBRAIN_HOME))))
    parser.add_argument("--prime", action="store_true", help="Populate query cache with up to 30 seconds per query")
    parser.add_argument("--api", action="store_true", help="Also generate answers through Gemini")
    args = parser.parse_args()
    os.environ["GBRAIN_HOME"] = str(args.gbrain_home.resolve())
    docs_dir = resolve_docs_dir(args.docs_dir, args.config)
    chunks = load_documents(docs_dir)
    questions = json.loads(read_text_file(args.questions)) if args.questions else [
        {"question": "経理採用を上手く行かせるには"},
        {"question": "エンジニアの選考フローどうしたら"},
    ]
    questions = questions[:args.count]
    status = SearchStatus()
    SEARCH_LOGGER.setLevel(logging.INFO)
    SEARCH_LOGGER.addHandler(status)
    bot = SlackHermesBot(SlackBotSettings(docs_dir, search_mode="parallel")) if args.api else None
    if bot:
        bot.refresh_documents()
    records = []
    result_limit = bot.settings.top_k if bot else 5
    try:
        for index, item in enumerate(questions):
            question = item["question"]
            query, scoped = prepare_advice_search(question, chunks) if is_advice_question(question) else (question, chunks)
            prime_result = None
            if args.prime:
                started = time.monotonic()
                try:
                    semantic = retrieve_gbrain(query, scoped, max(result_limit * 2, 10), args.gbrain_home, timeout_seconds=30)
                    prime_result = {"seconds": round(time.monotonic() - started, 3), "matches": len(semantic)}
                except Exception as exc:
                    prime_result = {"seconds": round(time.monotonic() - started, 3), "error_type": type(exc).__name__}
                print(json.dumps({"question_number": index + 1, "prime": prime_result}), flush=True)
            started = time.monotonic()
            keyword = retrieve_keyword(query, scoped, result_limit)
            keyword_seconds = time.monotonic() - started
            started = time.monotonic()
            parallel = retrieve_parallel(query, scoped, result_limit, gbrain_home=args.gbrain_home)
            row = {
                "question_number": index + 1,
                "keyword_seconds": round(keyword_seconds, 3),
                "parallel_seconds": round(time.monotonic() - started, 3),
                "status": status.status,
                "same_top1": bool(keyword and parallel and keyword[0].file_path == parallel[0].file_path),
            }
            # Keep sequential measurements independent of timeout cleanup from the last query.
            cleanup_started = time.monotonic()
            if not _PARALLEL_GBRAIN_SLOT.acquire(timeout=10):
                raise RuntimeError("GBrain timeout cleanup did not finish")
            _PARALLEL_GBRAIN_SLOT.release()
            row["cleanup_seconds"] = round(time.monotonic() - cleanup_started, 3)
            expected = {normalize_text(Path(name).stem).lower() for name in item.get("expected_files", [])}
            if expected:
                for mode, results in (("keyword", keyword), ("parallel", parallel)):
                    row[f"{mode}_rank"] = next((rank for rank, chunk in enumerate(results, 1) if normalize_text(chunk.file_path.stem).lower() in expected), None)
                    evidence_terms = [normalize_text(term).lower() for term in item.get("evidence_terms", [])]
                    row[f"{mode}_evidence_rank"] = next((
                        cutoff for cutoff in range(1, len(results) + 1)
                        if evidence_terms and all(
                            term in normalize_text("\n".join(chunk.text for chunk in results[:cutoff])).lower()
                            for term in evidence_terms
                        )
                    ), None)
            if bot:
                started = time.monotonic()
                try:
                    answer = bot.answer(question, "benchmark", str(index))
                    row["answer_chars"] = len(answer)
                    row["answer_has_citations"] = "[S" in answer
                    row["answer_timed_out"] = "時間内に完了しません" in answer
                except Exception as exc:
                    row["answer_error_type"] = type(exc).__name__
                row["answer_seconds"] = round(time.monotonic() - started, 3)
                row["answer_search_status"] = status.status
            records.append(row)
            print(json.dumps(row), flush=True)
        summary = {
            "questions": len(records),
            "gbrain_used": sum("gbrain=used " in row["status"] for row in records),
        }
        scored = [row for row in records if "keyword_rank" in row]
        if scored:
            summary["scored_questions"] = len(scored)
            for mode in ("keyword", "parallel"):
                for metric in ("rank", "evidence_rank"):
                    name = "evidence_" if metric == "evidence_rank" else ""
                    summary[f"{mode}_{name}top1"] = sum(row.get(f"{mode}_{metric}") == 1 for row in scored)
                    summary[f"{mode}_{name}top5"] = sum(0 < (row.get(f"{mode}_{metric}") or 0) <= 5 for row in scored)
        print(json.dumps(summary), flush=True)
    finally:
        SEARCH_LOGGER.removeHandler(status)
        if _PARALLEL_GBRAIN_SLOT.acquire(timeout=10):
            _PARALLEL_GBRAIN_SLOT.release()
        if bot:
            bot.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
