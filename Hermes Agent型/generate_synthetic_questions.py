from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

from hermes_agent import (
    DEFAULT_CONFIG_PATH,
    canonical_document_id,
    get_gemini_api_key,
    load_documents,
    post_gemini_json,
    read_text_file,
    resolve_docs_dir,
)


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("synthetic_questions.json")
EXISTING_QUESTION_PATHS = (
    Path(__file__).with_name("accuracy_questions.json"),
    Path(__file__).with_name("accuracy_holdout_questions.json"),
)


def select_source_chunks(chunks, count: int) -> list:
    used_files: set[str] = set()
    for path in EXISTING_QUESTION_PATHS:
        for item in json.loads(read_text_file(path)):
            used_files.update(
                canonical_document_id(name) for name in item["expected_files"]
            )

    grouped = {}
    for chunk in chunks:
        if canonical_document_id(chunk.file_path.name) in used_files:
            continue
        grouped.setdefault(chunk.file_path, []).append(chunk)

    signal_terms = (
        "決定",
        "対応",
        "担当",
        "確認",
        "必要",
        "送る",
        "面接",
        "採用",
        "求人",
        "候補者",
        "スカウト",
    )

    def content_score(chunk) -> tuple[int, int]:
        signals = sum(chunk.text.count(term) for term in signal_terms)
        return signals, min(len(chunk.text), 700)

    representatives = [
        max(document_chunks, key=content_score)
        for document_chunks in grouped.values()
    ]
    representatives.sort(
        key=lambda chunk: hashlib.sha256(
            str(chunk.file_path).encode("utf-8")
        ).hexdigest()
    )
    return representatives[:count]


def generate_questions(sources: list, model: str) -> list[dict]:
    source_text = "\n\n".join(
        f"<SOURCE id=\"D{index}\">\n{chunk.text[:900]}\n</SOURCE>"
        for index, chunk in enumerate(sources, start=1)
    )
    prompt = f"""
以下は互いに異なる会議ログの抜粋です。検索システムの未知質問評価用に、
各SOURCEにつき日本語の質問を1問作ってください。

条件:
- 質問は、そのSOURCEだけで具体的に答えられること。
- 元文の表現をそのまま並べず、自然な言い換えにすること。
- 会議名、ファイル名、SOURCE IDを質問文へ含めないこと。
- 日時を問う質問は作らないこと。
- 一般論ではなく、実際に話された方針、理由、手順、担当、注意事項を問うこと。
- evidence_termsには、答えの判定に必要で、SOURCE本文へ完全一致する3文字以上の
  文字列を2個入れること。表記を変更しないこと。
- SOURCEに明確な質問を作れない場合は、そのSOURCEを出力しないこと。

JSON配列だけを返してください。各要素は次の形式です。
{{"source_id":"D1","question":"...","evidence_terms":["...","..."]}}

{source_text}
""".strip()
    model_name = model.removeprefix("models/")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:"
        f"generateContent?key={get_gemini_api_key()}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    result = post_gemini_json(url, payload, timeout=180)
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def validate_questions(generated: list[dict], sources: list) -> list[dict]:
    source_by_id = {
        f"D{index}": chunk for index, chunk in enumerate(sources, start=1)
    }
    validated: list[dict] = []
    seen_questions: set[str] = set()
    for item in generated:
        source = source_by_id.get(str(item.get("source_id", "")))
        question = str(item.get("question", "")).strip()
        terms = item.get("evidence_terms", [])
        if source is None or not question or len(terms) != 2:
            continue
        normalized_text = normalize(source.text)
        normalized_terms = [normalize(str(term).strip()) for term in terms]
        if any(len(term) < 3 or term not in normalized_text for term in normalized_terms):
            continue
        question_key = re.sub(r"\s+", "", normalize(question)).lower()
        if question_key in seen_questions:
            continue
        title_id = canonical_document_id(source.file_path.stem)
        if len(title_id) >= 5 and title_id in canonical_document_id(question):
            continue
        seen_questions.add(question_key)
        validated.append(
            {
                "category": "疑似未知質問",
                "question": question,
                "expected_files": [source.file_path.name],
                "evidence_terms": normalized_terms,
                "source_chunk": source.chunk_index,
            }
        )
    return validated


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="未使用会議ログから疑似評価質問を作る")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    chunks = load_documents(resolve_docs_dir(args.docs_dir, args.config))
    sources = select_source_chunks(chunks, args.count)
    generated = generate_questions(sources, args.model)
    validated = validate_questions(generated, sources)
    args.output.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"疑似質問を{len(validated)}件保存しました: {args.output} "
        f"(候補SOURCE: {len(sources)}件)"
    )


if __name__ == "__main__":
    main()
