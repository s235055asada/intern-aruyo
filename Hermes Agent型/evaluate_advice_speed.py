from __future__ import annotations

import argparse
import json
import time

from hermes_agent import prepare_advice_search, resolve_docs_dir, retrieve_keyword
from slack_bot import SlackBotSettings, SlackHermesBot, format_slack_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="相談質問の検索・回答時間を評価")
    parser.add_argument("--api", action="store_true", help="Gemini実APIを2回呼ぶ")
    args = parser.parse_args()
    bot = SlackHermesBot(SlackBotSettings(resolve_docs_dir(None)))
    try:
        started = time.monotonic()
        bot.refresh_documents()
        print(json.dumps({
            "startup_seconds": round(time.monotonic() - started, 3),
            "files": len({chunk.file_path for chunk in bot.chunks}),
            "chunks": len(bot.chunks),
        }, ensure_ascii=False), flush=True)
        for index, question in enumerate((
            "経理採用を上手く行かせるには",
            "エンジニアの選考フローどうしたら",
        )):
            started = time.monotonic()
            search_question, scoped = prepare_advice_search(question, bot.chunks)
            contexts = retrieve_keyword(search_question, scoped, limit=6)
            result = {
                "question": question,
                "search_seconds": round(time.monotonic() - started, 3),
                "topic_chunks": len(scoped),
                "source_files": [chunk.file_path.name for chunk in contexts],
            }
            if args.api:
                started = time.monotonic()
                try:
                    answer = bot.answer(question, "evaluation", str(index))
                    result["answer"] = format_slack_answer(answer)
                    result["timed_out"] = "時間内に完了しません" in answer
                except Exception as exc:
                    result["error_type"] = type(exc).__name__
                result["response_seconds"] = round(time.monotonic() - started, 3)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        bot.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
