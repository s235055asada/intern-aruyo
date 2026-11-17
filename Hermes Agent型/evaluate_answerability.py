from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_agent import (
    DEFAULT_CONFIG_PATH,
    answerability_failure_reason,
    load_documents,
    read_text_file,
    resolve_docs_dir,
    retrieve,
)


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("answerability_questions.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="回答可能性の事前判定を評価する")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    questions = json.loads(read_text_file(args.questions))
    chunks = load_documents(resolve_docs_dir(args.docs_dir, args.config))
    correct = 0

    for item in questions:
        contexts = retrieve(
            item["question"],
            chunks,
            limit=args.top_k,
            search_mode="keyword",
        )
        reason = answerability_failure_reason(item["question"], contexts)
        predicted_answerable = reason is None
        matched = predicted_answerable == item["answerable"]
        correct += int(matched)
        status = "PASS" if matched else "FAIL"
        prediction = "回答可能" if predicted_answerable else f"拒否: {reason}"
        print(f"[{status}] {prediction} | {item['question']}")

    accuracy = correct / len(questions) if questions else 0.0
    print(f"\nAccuracy: {correct}/{len(questions)} ({accuracy:.1%})")
    raise SystemExit(0 if correct == len(questions) else 1)


if __name__ == "__main__":
    main()
