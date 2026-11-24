from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
IGNORED_DOCUMENT_NAMES = {"_batch_log.txt"}
DEFAULT_SYSTEM_PATH = Path(__file__).with_name("system.md")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_GOAL_PATH = Path(__file__).parent.parent / "目標１.txt"
DEFAULT_EMBEDDING_CACHE_PATH = Path(__file__).parent / ".cache" / "gemini_embeddings.json"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_LOCAL_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_LOCAL_EMBEDDING_CACHE_PATH = (
    Path(__file__).parent / ".cache" / "local_chunk_embeddings.json"
)
DEFAULT_GBRAIN_HOME = Path(__file__).parent / ".gbrain"
DEFAULT_GBRAIN_QUERY_CACHE_PATH = (
    Path(__file__).parent / ".cache" / "gbrain_queries.json"
)
_LOCAL_EMBEDDING_MODEL_INSTANCE = None
_PARALLEL_GBRAIN_SLOT = threading.BoundedSemaphore(1)
_GBRAIN_CACHE_LOCK = threading.Lock()
SEARCH_LOGGER = logging.getLogger("hermes.search")


@dataclass(frozen=True)
class DocumentMetadata:
    meeting_name: str
    meeting_date: str | None
    speakers: tuple[str, ...]
    source: str | None


@dataclass(frozen=True)
class DocumentChunk:
    file_path: Path
    chunk_index: int
    text: str
    metadata: DocumentMetadata
    score: float = 0.0
    keyword_score: float = 0.0
    semantic_score: float = 0.0

    @property
    def cache_key(self) -> str:
        content = f"{self.file_path.resolve()}\0{self.chunk_index}\0{self.text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_system_prompt(system_path: Path = DEFAULT_SYSTEM_PATH) -> str:
    return read_text_file(system_path)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    return json.loads(read_text_file(config_path))


def resolve_docs_dir(cli_docs_dir: Path | None, config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    if cli_docs_dir is not None:
        return cli_docs_dir

    config = load_config(config_path)
    docs_dir = config.get("local_docs_dir")
    if docs_dir:
        docs_path = Path(docs_dir)
        if docs_path.is_absolute():
            return docs_path
        return (config_path.parent / docs_path).resolve()

    return Path("docs")


def normalize_query(text: str) -> list[str]:
    text = normalize_text(text)
    known_terms = [
        "オンボーディング",
        "オンボ",
        "受け入れ",
        "受入",
        "TODO",
        "ToDo",
        "担当",
        "期限",
    ]
    matched_terms = [term for term in known_terms if normalize_text(term).lower() in text.lower()]

    removable_phrases = [
        "について",
        "ください",
        "教えて",
        "どうなった",
        "あります",
        "ですか",
        "ました",
        "会議",
        "内容",
        "決まったこと",
        "決まった",
        "決定したこと",
        "決定",
    ]
    cleaned = text
    for phrase in removable_phrases:
        cleaned = cleaned.replace(phrase, " ")

    tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥ー]{2,}", cleaned.lower())
    stop_words = {
        "について",
        "ください",
        "教えて",
        "どうなった",
        "あります",
        "ですか",
        "ました",
        "会議",
        "内容",
        "決まった",
        "決定",
    }
    terms = matched_terms + [token for token in tokens if token not in stop_words]
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        expanded.extend(expand_query_term(term))

    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in expanded:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def expand_query_term(term: str) -> list[str]:
    aliases = {
        "オンボーディング": ["オンボ", "オンボ", "onboarding", "受け入れ", "受入"],
        "オンボ": ["オンボーディング", "オンボ", "onboarding", "受け入れ", "受入"],
        "受け入れ": ["受入", "オンボ", "オンボーディング"],
        "受入": ["受け入れ", "オンボ", "オンボーディング"],
        "todo": ["to do", "アクション", "宿題", "やること", "ネクスト"],
        "担当": ["担当者", "オーナー"],
        "期限": ["期日", "締切", "いつまで"],
    }
    return aliases.get(term, [])


def split_document(text: str, max_chars: int = 700, overlap_chars: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            period = text.rfind("。", start, end)
            split_at = max(newline, period)
            if split_at > start + max_chars // 2:
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def extract_metadata(path: Path, text: str) -> DocumentMetadata:
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    meeting_name = title_match.group(1).strip() if title_match else path.stem

    date_patterns = [
        r"(?mi)^-\s*Date:\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2}(?:\s+\d{1,2}:\d{2})?)",
        r"(?<!\d)(\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\d)",
        r"(?<!\d)(\d{4}年\d{1,2}月\d{1,2}日)(?!\d)",
    ]
    meeting_date = None
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            meeting_date = match.group(1)
            break

    source_match = re.search(r"(?mi)^-\s*Source:\s*(.+?)\s*$", text)
    source = source_match.group(1).strip() if source_match else None

    speakers: list[str] = []
    speaker_patterns = [
        r"(?m)^###\s+(?:\d{1,2}:\d{2}(?::\d{2})?\s+)(.+?)\s*$",
        r"(?m)^##\s+(.+?)\s+`\d{1,2}:\d{2}(?::\d{2})?`\s*$",
    ]
    for pattern in speaker_patterns:
        for speaker in re.findall(pattern, text):
            cleaned = speaker.strip()
            if cleaned and cleaned not in speakers:
                speakers.append(cleaned)

    return DocumentMetadata(
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        speakers=tuple(speakers),
        source=source,
    )


def load_documents(docs_dir: Path) -> list[DocumentChunk]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"文書フォルダが見つかりません: {docs_dir}")

    paths = [
        path
        for path in sorted(docs_dir.rglob("*"))
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.name.lower() not in IGNORED_DOCUMENT_NAMES
        )
    ]
    extension_priority = {".md": 0, ".markdown": 1, ".txt": 2}
    unique_paths: dict[tuple[Path, str], Path] = {}
    for path in paths:
        key = (path.parent.resolve(), normalize_text(path.stem).lower())
        current = unique_paths.get(key)
        if current is None or extension_priority.get(path.suffix.lower(), 9) < (
            extension_priority.get(current.suffix.lower(), 9)
        ):
            unique_paths[key] = path

    chunks: list[DocumentChunk] = []
    for path in sorted(unique_paths.values()):
        text = read_text_file(path)
        metadata = extract_metadata(path, text)
        for index, chunk_text in enumerate(split_document(text), start=1):
            chunks.append(
                DocumentChunk(
                    file_path=path,
                    chunk_index=index,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

    return chunks


def score_chunk(question_terms: list[str], chunk: DocumentChunk) -> float:
    normalized_name = normalize_text(chunk.file_path.name).lower()
    normalized_meeting = normalize_text(chunk.metadata.meeting_name).lower()
    normalized_text = normalize_text(chunk.text).lower()
    score = 0.0
    for term in question_terms:
        normalized_term = normalize_text(term).lower()
        score += normalized_text.count(normalized_term) * max(1, min(len(normalized_term), 8))
        score += normalized_name.count(normalized_term) * max(12, min(len(normalized_term) * 8, 64))
        score += normalized_meeting.count(normalized_term) * max(8, min(len(normalized_term) * 6, 48))
    return score


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9_]+|[ぁ-んァ-ン一-龥ー]+", normalized)
    result: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[a-z0-9_]+", token):
            result.append(token)
            continue
        if len(token) <= 4:
            result.append(token)
        for size in (2, 3):
            result.extend(
                token[index : index + size]
                for index in range(max(0, len(token) - size + 1))
            )
    return result


def search_aliases(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    synonym_groups = [
        ("オンボーディング", "オンボ", "受け入れ", "受入"),
        ("応募者", "候補者", "エントリー"),
        ("重複応募", "重複", "二重エントリー", "二重", "同一候補者"),
        ("レコード", "候補者データ"),
        ("消す", "削除"),
        ("人工知能", "AI"),
        ("スキル共有", "スキル", "GitHub", "共有方法"),
        ("録音ボット", "Notta"),
        ("その場で", "リアルタイム"),
        ("連休", "ゴールデンウィーク", "GW"),
        ("承認", "OK", "確認"),
        ("本公開", "公開設定"),
        ("公開状態", "限定公開", "公開設定"),
        ("止める", "停止", "ストップ"),
        ("取引先", "クライアント"),
        ("競合", "他社", "スカウト20件以上"),
        ("不利", "ビハインド", "年収"),
        ("管理職クラス以外", "上位レイヤー以外", "プレイヤー"),
        ("エンジニア", "実装"),
        ("出勤形態", "勤務形態", "出社", "必要に応じて出社"),
        ("勤務時間", "フレックス", "基本フレックス"),
        ("母集団の規模", "必要母集団", "母集団"),
        ("選考スピード", "先行スピード"),
        ("分かりやすく", "具体例"),
        ("依頼", "お願い", "頼む"),
        ("選考フロー", "先行フロー", "専攻フロー"),
        ("選考管理", "先行管理"),
        ("選考設定", "先行設定", "専攻設定"),
        ("選考体験", "先行体験"),
        ("選考依頼", "先行依頼"),
        ("書類選考", "書類先行"),
        ("面接官", "先行感", "先行缶", "選考官"),
        ("母集団形成", "母集団女性"),
        ("辞退率", "事態の率", "自体の率"),
    ]
    additions: list[str] = []
    for group in synonym_groups:
        if any(normalize_text(term).lower() in normalized for term in group):
            additions.extend(group)
    return list(dict.fromkeys(additions))


def expand_search_text(text: str) -> str:
    additions = search_aliases(text)
    return f"{text} {' '.join(additions)}" if additions else text


@lru_cache(maxsize=4096)
def cached_search_tokens(text: str) -> Counter[str]:
    return Counter(tokenize_for_bm25(text))


def bm25_scores(question: str, chunks: list[DocumentChunk]) -> list[float]:
    if not chunks:
        return []
    query_terms = Counter(tokenize_for_bm25(expand_search_text(question)))
    normalized_question = normalize_text(question).lower()
    boosted_aliases = {
        "オンボ",
        "重複応募",
        "同一候補者",
        "候補者データ",
        "削除",
        "github",
        "共有方法",
        "notta",
        "リアルタイム",
        "ゴールデンウィーク",
        "公開設定",
        "限定公開",
        "スカウト20件以上",
        "ビハインド",
        "年収",
        "プレイヤー",
        "実装",
        "必要に応じて出社",
        "基本フレックス",
        "必要母集団",
        "先行スピード",
        "具体例",
        "お願い",
        "先行フロー",
        "先行管理",
        "先行設定",
        "先行体験",
        "先行依頼",
        "書類先行",
        "先行感",
        "先行缶",
        "母集団女性",
        "事態の率",
    }
    all_boosted_aliases = [
        normalize_text(alias).lower()
        for alias in search_aliases(question)
        if normalize_text(alias).lower() in boosted_aliases
    ]
    aliases = [
        alias
        for alias in all_boosted_aliases
        if alias not in normalized_question
    ]
    if not query_terms:
        return [0.0] * len(chunks)

    generic_person_terms = {"候補者", "応募者", "担当者", "責任者"}
    person_names = {
        name
        for name in re.findall(r"([一-龯々]{1,4})(?:氏|さん)", normalized_question)
        if name not in generic_person_terms
    }

    document_tokens = [
        cached_search_tokens(
            f"{chunk.file_path.stem} {chunk.metadata.meeting_name} {chunk.text}"
        )
        for chunk in chunks
    ]
    document_frequencies: Counter[str] = Counter()
    for chunk, tokens in zip(chunks, document_tokens, strict=True):
        document_frequencies.update(set(tokens))

    average_length = (
        sum(sum(tokens.values()) for tokens in document_tokens) / len(document_tokens)
    )
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    document_count = len(document_tokens)
    for chunk, tokens in zip(chunks, document_tokens, strict=True):
        frequencies = tokens
        length_normalization = 1 - b + b * sum(tokens.values()) / max(average_length, 1)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / (frequency + k1 * length_normalization)
                * min(query_frequency, 2)
            )
        searchable_text = normalize_text(
            f"{chunk.file_path.stem} {chunk.metadata.meeting_name} {chunk.text}"
        ).lower()
        title_text = normalize_text(
            f"{chunk.file_path.stem} {chunk.metadata.meeting_name}"
        ).lower()
        matched_aliases = [alias for alias in aliases if alias in searchable_text]
        score += sum(
            min(len(alias), 8) * 10.0
            for alias in matched_aliases
        )
        if len(matched_aliases) >= 2:
            score += 100.0 * (len(matched_aliases) - 1)
        score += 60.0 * sum(alias in title_text for alias in all_boosted_aliases)
        for person_name in person_names:
            if person_name in searchable_text:
                score += 120.0
                if any(term in searchable_text for term in ("依頼", "お願い", "頼む")):
                    score += 160.0
        scores.append(score)
    return scores


def add_adjacent_chunk_scores(
    scores: list[float],
    chunks: list[DocumentChunk],
    weight: float = 0.35,
) -> list[float]:
    score_by_location = {
        (chunk.file_path, chunk.chunk_index): score
        for chunk, score in zip(chunks, scores, strict=True)
    }
    adjusted: list[float] = []
    for chunk, score in zip(chunks, scores, strict=True):
        neighbor_score = max(
            score_by_location.get(
                (chunk.file_path, chunk.chunk_index + offset),
                0.0,
            )
            for offset in (-1, 1)
        )
        adjusted.append(score + neighbor_score * weight)
    return adjusted


def title_match_score(question_terms: list[str], chunk: DocumentChunk) -> float:
    title = normalize_text(
        f"{chunk.file_path.stem} {chunk.metadata.meeting_name}"
    ).lower()
    meaningful_terms = {
        normalize_text(term).lower()
        for term in question_terms
        if len(normalize_text(term)) >= 2
    }
    if not meaningful_terms:
        return 0.0
    matched_length = sum(len(term) for term in meaningful_terms if term in title)
    longest = max(len(term) for term in meaningful_terms)
    return min(1.0, matched_length / max(longest, 1))


def retrieve_keyword(question: str, chunks: list[DocumentChunk], limit: int = 6) -> list[DocumentChunk]:
    raw_scores = add_adjacent_chunk_scores(bm25_scores(question, chunks), chunks)
    scored = [
        DocumentChunk(
            file_path=chunk.file_path,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            metadata=chunk.metadata,
            score=score,
            keyword_score=score,
        )
        for chunk, score in zip(chunks, raw_scores, strict=True)
    ]
    scored = [chunk for chunk in scored if chunk.score > 0]
    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    normalized_question = normalize_text(question).lower()
    title_focus_terms = (
        ("オンボ",) if "オンボーディング" in normalized_question else ()
    )
    if title_focus_terms:
        title_focused = [
            chunk
            for chunk in scored
            if any(
                term
                in normalize_text(
                    f"{chunk.file_path.stem} {chunk.metadata.meeting_name}"
                ).lower()
                for term in title_focus_terms
            )
        ]
        if title_focused:
            scored = title_focused
    diverse: list[DocumentChunk] = []
    seen_files: set[Path] = set()
    for chunk in scored:
        if chunk.file_path in seen_files:
            continue
        seen_files.add(chunk.file_path)
        diverse.append(expand_chunk_context(chunk, chunks))
        if len(diverse) >= limit:
            break
    return diverse


def get_gemini_api_key() -> str:
    api_key = (
        get_env_or_windows_user_env("GEMINI_API_KEY")
        or get_env_or_windows_user_env("GOOGLE_API_KEY")
        or load_gemini_key_from_goal_file()
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY または GOOGLE_API_KEY が設定されていません。")
    return api_key


def post_gemini_json(url: str, payload: dict, timeout: int = 60, retries: int = 4) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(f"Gemini API error {exc.code}: {body}") from exc
            time.sleep(20 if exc.code == 429 else 2**attempt)
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Gemini API connection error: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError("Gemini APIへの接続に失敗しました。")


def embed_texts_with_gemini(
    texts: list[str],
    task_type: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[list[float]]:
    if not texts:
        return []

    model_name = model.removeprefix("models/")
    resource_name = f"models/{model_name}"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/{resource_name}:"
        f"batchEmbedContents?key={get_gemini_api_key()}"
    )
    requests = []
    for text in texts:
        requests.append(
            {
                "model": resource_name,
                "content": {"parts": [{"text": text}]},
                "embedContentConfig": {
                    "taskType": task_type,
                    "outputDimensionality": dimensions,
                    "autoTruncate": True,
                },
            }
        )
    result = post_gemini_json(url, {"requests": requests}, timeout=120)
    embeddings = [
        embedding.get("values", [])
        for embedding in result.get("embeddings", [])
    ]
    if len(embeddings) != len(texts) or any(not values for values in embeddings):
        raise RuntimeError(f"Gemini Embeddingsの応答件数が不正です: {len(embeddings)}/{len(texts)}")
    return embeddings


def load_embedding_cache(
    cache_path: Path,
    model: str,
    dimensions: int,
) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(read_text_file(cache_path))
    except (json.JSONDecodeError, OSError):
        return {}
    if (
        payload.get("model") != model
        or payload.get("dimensions") != dimensions
        or payload.get("scope") != "document-v1"
    ):
        return {}
    return payload.get("embeddings", {})


def save_embedding_cache(
    cache_path: Path,
    embeddings: dict[str, list[float]],
    model: str,
    dimensions: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dimensions": dimensions,
        "scope": "document-v1",
        "embeddings": embeddings,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)


def build_embedding_index(
    chunks: list[DocumentChunk],
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    batch_size: int = 8,
) -> dict[str, list[float]]:
    cached = load_embedding_cache(cache_path, model, dimensions)
    grouped: dict[Path, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.file_path, []).append(chunk)

    documents: list[tuple[str, str]] = []
    for path, document_chunks in grouped.items():
        content_hash = hashlib.sha256(
            "\0".join(chunk.cache_key for chunk in document_chunks).encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(
            f"{path.resolve()}\0{content_hash}".encode("utf-8")
        ).hexdigest()
        metadata = document_chunks[0].metadata
        if len(document_chunks) <= 4:
            samples = document_chunks
        else:
            indexes = {
                0,
                len(document_chunks) // 3,
                (len(document_chunks) * 2) // 3,
                len(document_chunks) - 1,
            }
            samples = [document_chunks[index] for index in sorted(indexes)]
        representative_text = (
            f"会議名: {metadata.meeting_name}\n"
            f"日付: {metadata.meeting_date or '不明'}\n"
            f"発言者: {', '.join(metadata.speakers)}\n"
            + "\n".join(chunk.text[:150] for chunk in samples)
        )
        documents.append((cache_key, representative_text))

    active_keys = {key for key, _ in documents}
    cached = {key: value for key, value in cached.items() if key in active_keys}
    missing = [(key, text) for key, text in documents if key not in cached]

    if missing:
        print(
            f"Gemini Embeddingsを更新中: {len(missing)}文書",
            file=sys.stderr,
        )
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        texts = [text for _, text in batch]
        vectors = embed_texts_with_gemini(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            model=model,
            dimensions=dimensions,
        )
        for (cache_key, _), vector in zip(batch, vectors, strict=True):
            cached[cache_key] = vector
        save_embedding_cache(cache_path, cached, model, dimensions)
        print(
            f"  {min(start + len(batch), len(missing))}/{len(missing)}",
            file=sys.stderr,
        )
        if start + len(batch) < len(missing):
            time.sleep(8)
    return cached


def document_embedding_keys(chunks: list[DocumentChunk]) -> dict[Path, str]:
    grouped: dict[Path, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.file_path, []).append(chunk)
    keys: dict[Path, str] = {}
    for path, document_chunks in grouped.items():
        content_hash = hashlib.sha256(
            "\0".join(chunk.cache_key for chunk in document_chunks).encode("utf-8")
        ).hexdigest()
        keys[path] = hashlib.sha256(
            f"{path.resolve()}\0{content_hash}".encode("utf-8")
        ).hexdigest()
    return keys


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def get_local_embedding_model(model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
    global _LOCAL_EMBEDDING_MODEL_INSTANCE
    if _LOCAL_EMBEDDING_MODEL_INSTANCE is not None:
        return _LOCAL_EMBEDDING_MODEL_INSTANCE
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "fastembedがインストールされていません。"
            "`pip install -r requirements.txt`を実行してください。"
        ) from exc
    _LOCAL_EMBEDDING_MODEL_INSTANCE = TextEmbedding(model_name=model_name)
    return _LOCAL_EMBEDDING_MODEL_INSTANCE


def build_local_chunk_index(
    chunks: list[DocumentChunk],
    cache_path: Path = DEFAULT_LOCAL_EMBEDDING_CACHE_PATH,
    model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
) -> dict[str, list[float]]:
    cached: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            payload = json.loads(read_text_file(cache_path))
            if (
                payload.get("model") == model_name
                and payload.get("scope") == "local-chunk-v1"
            ):
                cached = payload.get("embeddings", {})
        except (json.JSONDecodeError, OSError):
            cached = {}

    active_keys = {chunk.cache_key for chunk in chunks}
    cached = {key: value for key, value in cached.items() if key in active_keys}
    missing = [chunk for chunk in chunks if chunk.cache_key not in cached]
    if missing:
        print(
            f"ローカルEmbeddingを更新中: {len(missing)}チャンク",
            file=sys.stderr,
        )
        model = get_local_embedding_model(model_name)
        texts = [
            (
                f"会議名: {chunk.metadata.meeting_name}\n"
                f"本文: {chunk.text}"
            )
            for chunk in missing
        ]
        vectors = list(model.embed(texts, batch_size=32))
        for chunk, vector in zip(missing, vectors, strict=True):
            cached[chunk.cache_key] = vector.tolist()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "scope": "local-chunk-v1",
                    "embeddings": cached,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    return cached


def rerank_chunks(candidates: list[DocumentChunk], limit: int) -> list[DocumentChunk]:
    selected: list[DocumentChunk] = []
    remaining = list(candidates)
    while remaining and len(selected) < limit:
        eligible = [
            chunk
            for chunk in remaining
            if sum(chosen.file_path == chunk.file_path for chosen in selected) < 3
        ]
        if not eligible:
            eligible = remaining

        def rerank_score(chunk: DocumentChunk) -> float:
            same_file_penalty = sum(
                0.10 for chosen in selected if chosen.file_path == chunk.file_path
            )
            adjacent_penalty = sum(
                0.12
                for chosen in selected
                if chosen.file_path == chunk.file_path
                and abs(chosen.chunk_index - chunk.chunk_index) <= 1
            )
            return chunk.score - same_file_penalty - adjacent_penalty

        best = max(eligible, key=rerank_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve_local(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
    cache_path: Path = DEFAULT_LOCAL_EMBEDDING_CACHE_PATH,
    model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
) -> list[DocumentChunk]:
    embeddings = build_local_chunk_index(
        chunks,
        cache_path=cache_path,
        model_name=model_name,
    )
    model = get_local_embedding_model(model_name)
    query_vector = next(model.query_embed(question)).tolist()
    terms = normalize_query(question) or [question.lower()]
    raw_keyword_scores = add_adjacent_chunk_scores(
        bm25_scores(question, chunks),
        chunks,
    )
    max_keyword = max(raw_keyword_scores, default=0.0)

    candidates: list[DocumentChunk] = []
    for chunk, keyword_score in zip(chunks, raw_keyword_scores, strict=True):
        semantic_score = cosine_similarity(
            query_vector,
            embeddings.get(chunk.cache_key, []),
        )
        normalized_keyword = keyword_score / max_keyword if max_keyword else 0.0
        title_score = title_match_score(terms, chunk)
        score = (
            semantic_score * 0.55
            + normalized_keyword * 0.40
            + title_score * 0.05
        )
        candidates.append(
            DocumentChunk(
                file_path=chunk.file_path,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                metadata=chunk.metadata,
                score=score,
                keyword_score=normalized_keyword,
                semantic_score=semantic_score,
            )
        )
    candidates.sort(key=lambda chunk: chunk.score, reverse=True)
    return rerank_chunks(candidates, limit)


def retrieve_hybrid(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[DocumentChunk]:
    embeddings = build_embedding_index(
        chunks,
        cache_path=cache_path,
        model=embedding_model,
        dimensions=dimensions,
    )
    query_vector = embed_texts_with_gemini(
        [question],
        task_type="RETRIEVAL_QUERY",
        model=embedding_model,
        dimensions=dimensions,
    )[0]
    terms = normalize_query(question) or [question.lower()]
    raw_keyword_scores = bm25_scores(question, chunks)
    max_keyword = max(raw_keyword_scores, default=0.0)
    embedding_keys = document_embedding_keys(chunks)

    candidates: list[DocumentChunk] = []
    for chunk, keyword_score in zip(chunks, raw_keyword_scores, strict=True):
        semantic_score = cosine_similarity(
            query_vector,
            embeddings.get(embedding_keys[chunk.file_path], []),
        )
        normalized_keyword = keyword_score / max_keyword if max_keyword else 0.0
        title_score = title_match_score(terms, chunk)
        hybrid_score = (
            semantic_score * 0.30
            + normalized_keyword * 0.60
            + title_score * 0.10
        )
        candidates.append(
            DocumentChunk(
                file_path=chunk.file_path,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                metadata=chunk.metadata,
                score=hybrid_score,
                keyword_score=normalized_keyword,
                semantic_score=semantic_score,
            )
        )

    candidates.sort(key=lambda chunk: chunk.score, reverse=True)
    return rerank_chunks(candidates, limit)


def canonical_document_id(value: str | Path) -> str:
    normalized = normalize_text(str(value)).lower()
    normalized = re.sub(r"\.(?:md|markdown|txt)$", "", normalized)
    return "".join(character for character in normalized if character.isalnum())


def parse_gbrain_search_output(output: str) -> list[tuple[float, str, str]]:
    results: list[tuple[float, str, str]] = []
    current: tuple[float, str, list[str]] | None = None
    result_pattern = re.compile(r"^\[([0-9.]+)\]\s+(.+?)\s+--\s*(.*)$")
    ignored_prefixes = ("UPGRADE_AVAILABLE", "gbrain ")

    for line in output.splitlines():
        match = result_pattern.match(line)
        if match:
            if current is not None:
                score, slug, text_parts = current
                results.append((score, slug, "\n".join(text_parts).strip()))
            current = (float(match.group(1)), match.group(2), [match.group(3)])
            continue
        if current is not None and not line.startswith(ignored_prefixes):
            current[2].append(line)

    if current is not None:
        score, slug, text_parts = current
        results.append((score, slug, "\n".join(text_parts).strip()))
    return results


def find_gbrain_executable() -> str:
    executable = shutil.which("gbrain")
    if executable:
        return executable
    if os.name == "nt":
        candidate = Path.home() / ".bun" / "bin" / "gbrain.exe"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "GBrainが見つかりません。公式手順でGBrainをインストールしてください。"
    )


def text_ngrams(text: str, size: int = 3) -> set[str]:
    normalized = "".join(
        character
        for character in normalize_text(text).lower()
        if not character.isspace()
    )
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def match_gbrain_result(
    slug: str,
    snippet: str,
    chunks: list[DocumentChunk],
) -> DocumentChunk | None:
    slug_id = canonical_document_id(slug)
    by_file: dict[Path, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.file_path, []).append(chunk)

    matching_files = [
        path
        for path in by_file
        if slug_id.endswith(canonical_document_id(path.stem))
        or canonical_document_id(path.stem).endswith(slug_id)
    ]
    if not matching_files:
        return None

    snippet_ngrams = text_ngrams(snippet)

    def match_score(chunk: DocumentChunk) -> float:
        if not snippet_ngrams:
            return 0.0
        overlap = len(snippet_ngrams & text_ngrams(chunk.text))
        return overlap / len(snippet_ngrams)

    candidates = [chunk for path in matching_files for chunk in by_file[path]]
    return max(candidates, key=match_score)


def expand_chunk_context(
    source: DocumentChunk,
    chunks: list[DocumentChunk],
    radius: int = 1,
) -> DocumentChunk:
    nearby = [
        chunk
        for chunk in chunks
        if chunk.file_path == source.file_path
        and abs(chunk.chunk_index - source.chunk_index) <= radius
    ]
    nearby.sort(
        key=lambda chunk: (
            chunk.chunk_index != source.chunk_index,
            abs(chunk.chunk_index - source.chunk_index),
            chunk.chunk_index,
        )
    )
    text = "\n\n".join(
        f"[元chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in nearby
    )
    return DocumentChunk(
        file_path=source.file_path,
        chunk_index=source.chunk_index,
        text=text,
        metadata=source.metadata,
        score=source.score,
        keyword_score=source.keyword_score,
        semantic_score=source.semantic_score,
    )


def retrieve_gbrain(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
    gbrain_home: Path = DEFAULT_GBRAIN_HOME,
    timeout_seconds: float = 30,
    query_cache_path: Path = DEFAULT_GBRAIN_QUERY_CACHE_PATH,
) -> list[DocumentChunk]:
    cache_key = gbrain_query_cache_key(question, chunks, limit)
    cached_output = load_gbrain_query_cache(query_cache_path).get(cache_key)
    if cached_output:
        return gbrain_matches_from_output(cached_output, chunks, limit)

    environment = os.environ.copy()
    environment["GBRAIN_HOME"] = str(gbrain_home.resolve())
    if not environment.get("GOOGLE_GENERATIVE_AI_API_KEY"):
        environment["GOOGLE_GENERATIVE_AI_API_KEY"] = get_gemini_api_key()

    command = [
        find_gbrain_executable(),
        "search",
        question,
        "--limit",
        str(limit),
        "--mode",
        "conservative",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if os.name == "nt":
            # Bun launchers may create child processes holding the PGlite DB open.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=3, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            process.kill()
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(
            f"GBrain検索が{timeout_seconds}秒以内に完了しませんでした。"
        ) from exc
    if process.returncode != 0:
        error = stderr.strip() or stdout.strip()
        raise RuntimeError(f"GBrain検索に失敗しました: {error}")

    matched = gbrain_matches_from_output(stdout, chunks, limit)
    if matched:
        with _GBRAIN_CACHE_LOCK:
            cache = load_gbrain_query_cache(query_cache_path)
            cache[cache_key] = stdout
            save_gbrain_query_cache(query_cache_path, cache)
    return matched


def gbrain_query_cache_key(
    question: str,
    chunks: list[DocumentChunk],
    limit: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(normalize_text(question).encode("utf-8"))
    digest.update(f"\0{limit}\0".encode("ascii"))
    for chunk in chunks:
        digest.update(chunk.cache_key.encode("ascii"))
    return digest.hexdigest()


def load_gbrain_query_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(read_text_file(path))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, str)
    }


def save_gbrain_query_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def gbrain_matches_from_output(
    output: str,
    chunks: list[DocumentChunk],
    limit: int,
) -> list[DocumentChunk]:

    matched: list[DocumentChunk] = []
    for score, slug, snippet in parse_gbrain_search_output(output):
        source = match_gbrain_result(slug, snippet, chunks)
        if source is None:
            continue
        matched.append(
            expand_chunk_context(
                DocumentChunk(
                file_path=source.file_path,
                chunk_index=source.chunk_index,
                text=source.text,
                metadata=source.metadata,
                score=score,
                semantic_score=score,
                ),
                chunks,
            )
        )
    return matched[:limit]


def retrieve_enhanced(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
) -> list[DocumentChunk]:
    candidate_limit = max(limit * 4, 20)
    keyword_results = retrieve_keyword(question, chunks, candidate_limit)
    try:
        gbrain_results = retrieve_gbrain(question, chunks, max(limit * 2, 10))
    except RuntimeError:
        gbrain_results = []

    return fuse_search_results(keyword_results, gbrain_results, chunks, limit)


def retrieve_parallel(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
    timeout_seconds: float = 2.5,
    gbrain_home: Path | None = None,
) -> list[DocumentChunk]:
    if timeout_seconds < 0:
        raise ValueError("GBrain待機時間は0以上で指定してください。")
    started = time.monotonic()
    deadline = started + timeout_seconds
    completed: queue.Queue = queue.Queue(maxsize=1)
    admitted = timeout_seconds > 0 and _PARALLEL_GBRAIN_SLOT.acquire(blocking=False)

    def semantic_search() -> None:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                completed.put(("timeout", []))
                return
            results = retrieve_gbrain(
                question,
                chunks,
                max(limit * 2, 10),
                gbrain_home=gbrain_home or Path(os.getenv("GBRAIN_HOME", str(DEFAULT_GBRAIN_HOME))),
                timeout_seconds=remaining,
            )
            status = "used" if results else "no_match"
            completed.put((status, results))
        except Exception as exc:
            # Exceptions may contain API credentials or external command output.
            completed.put((f"unavailable:{type(exc).__name__}", []))
        finally:
            _PARALLEL_GBRAIN_SLOT.release()

    if admitted:
        threading.Thread(target=semantic_search, daemon=True, name="gbrain-search").start()
    keyword_results = retrieve_keyword(question, chunks, max(limit * 4, 20))
    status = "busy" if timeout_seconds else "disabled"
    semantic_results: list[DocumentChunk] = []
    if admitted:
        try:
            status, semantic_results = completed.get(timeout=max(0, deadline - time.monotonic()))
            if time.monotonic() > deadline:
                status, semantic_results = "timeout", []
        except queue.Empty:
            status = "timeout"
    SEARCH_LOGGER.info("parallel gbrain=%s elapsed=%.3fs", status, time.monotonic() - started)
    if not semantic_results:
        return keyword_results[:limit]
    return fuse_search_results(keyword_results, semantic_results, chunks, limit)


def fuse_search_results(
    keyword_results: list[DocumentChunk],
    gbrain_results: list[DocumentChunk],
    chunks: list[DocumentChunk],
    limit: int,
) -> list[DocumentChunk]:
    keyword_by_file: dict[Path, tuple[int, DocumentChunk]] = {}
    for chunk in keyword_results:
        if chunk.file_path not in keyword_by_file:
            keyword_by_file[chunk.file_path] = (
                len(keyword_by_file) + 1,
                chunk,
            )
    gbrain_by_file = {
        chunk.file_path: (rank, chunk)
        for rank, chunk in enumerate(gbrain_results, start=1)
    }

    fused: list[DocumentChunk] = []
    for file_path in keyword_by_file.keys() | gbrain_by_file.keys():
        keyword_entry = keyword_by_file.get(file_path)
        gbrain_entry = gbrain_by_file.get(file_path)
        score = 0.0
        text_parts: list[str] = []
        source = None
        if keyword_entry is not None:
            keyword_rank, keyword_chunk = keyword_entry
            score += 1.2 / (20 + keyword_rank)
            expanded_keyword = expand_chunk_context(keyword_chunk, chunks)
            text_parts.append(expanded_keyword.text)
            source = keyword_chunk
        if gbrain_entry is not None:
            gbrain_rank, gbrain_chunk = gbrain_entry
            score += 1.0 / (20 + gbrain_rank)
            if gbrain_chunk.text not in text_parts:
                text_parts.append(gbrain_chunk.text)
            source = source or gbrain_chunk
        if source is None:
            continue
        fused.append(
            DocumentChunk(
                file_path=source.file_path,
                chunk_index=source.chunk_index,
                text="\n\n".join(text_parts),
                metadata=source.metadata,
                score=score,
                keyword_score=(
                    keyword_entry[1].score if keyword_entry is not None else 0.0
                ),
                semantic_score=(
                    gbrain_entry[1].score if gbrain_entry is not None else 0.0
                ),
            )
        )
    fused.sort(key=lambda chunk: (-chunk.score, str(chunk.file_path), chunk.chunk_index))
    primary_file = keyword_results[0].file_path if keyword_results else None
    if len(gbrain_results) >= 2 and keyword_results:
        semantic_top = gbrain_results[0]
        semantic_margin = semantic_top.score - gbrain_results[1].score
        keyword_top_score = keyword_results[0].score
        semantic_keyword_match = next(
            (
                chunk
                for chunk in keyword_results[:5]
                if chunk.file_path == semantic_top.file_path
            ),
            None,
        )
        if (
            semantic_keyword_match is not None
            and keyword_top_score > 0
            and semantic_keyword_match.score >= keyword_top_score * 0.65
            and semantic_margin >= 0.03
        ):
            primary_file = semantic_top.file_path
    if primary_file is not None:
        primary_index = next(
            (
                index
                for index, chunk in enumerate(fused)
                if chunk.file_path == primary_file
            ),
            None,
        )
        if primary_index not in {None, 0}:
            fused.insert(0, fused.pop(primary_index))
    return fused[:limit]


def retrieve(
    question: str,
    chunks: list[DocumentChunk],
    limit: int = 6,
    search_mode: str = "keyword",
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    gbrain_timeout: float = 2.5,
) -> list[DocumentChunk]:
    if search_mode == "keyword":
        results = retrieve_keyword(question, chunks, limit)
    elif search_mode == "local":
        results = retrieve_local(question, chunks, limit=limit)
    elif search_mode == "gbrain":
        results = retrieve_gbrain(question, chunks, limit=limit)
    elif search_mode == "enhanced":
        results = retrieve_enhanced(question, chunks, limit=limit)
    elif search_mode == "parallel":
        results = retrieve_parallel(question, chunks, limit=limit, timeout_seconds=gbrain_timeout)
    elif search_mode == "hybrid":
        results = retrieve_hybrid(
            question,
            chunks,
            limit=limit,
            cache_path=cache_path,
            embedding_model=embedding_model,
        )
    else:
        raise ValueError(f"未対応の検索方式です: {search_mode}")

    if is_summary_question(question):
        return expand_summary_contexts(question, results, chunks, limit=limit)
    return results


SUMMARY_QUESTION_MARKERS = (
    "要約",
    "まとめ",
    "会話内容",
    "会話の内容",
    "議論内容",
    "議論の内容",
    "何を話",
    "何が話",
)


def is_summary_question(question: str) -> bool:
    normalized = unicodedata.normalize("NFKC", question).replace(" ", "")
    return any(marker in normalized for marker in SUMMARY_QUESTION_MARKERS)


def requested_meeting_sequence(question: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", question)
    match = re.search(r"(?:第\s*)?(\d+)\s*回目", normalized)
    if match:
        return match.group(1)
    match = re.search(r"-(\d+)(?!\d)", normalized)
    return match.group(1) if match else None


def meeting_family_key(path: Path) -> str:
    stem = normalize_text(path.stem).lower()
    stem = re.sub(r"(?:_cleaned|_notta_raw|_raw)$", "", stem)
    return re.sub(r"-\d+$", "", stem)


def expand_summary_contexts(
    question: str,
    results: list[DocumentChunk],
    chunks: list[DocumentChunk],
    limit: int,
    max_documents: int = 3,
    max_document_chars: int = 80_000,
) -> list[DocumentChunk]:
    """Build chronological evidence segments for a meeting-wide summary."""
    requested_sequence = requested_meeting_sequence(question)
    if requested_sequence is not None:
        suffix_pattern = re.compile(
            rf"-{re.escape(requested_sequence)}(?:_cleaned|_notta_raw|_raw)?$",
            re.IGNORECASE,
        )
        if results:
            leading_family = meeting_family_key(results[0].file_path)
            representatives: dict[Path, DocumentChunk] = {}
            for chunk in chunks:
                representatives.setdefault(chunk.file_path, chunk)
            sequence_matches = [
                representative
                for path, representative in representatives.items()
                if meeting_family_key(path) == leading_family
                and suffix_pattern.search(normalize_text(path.stem))
            ]
            sequence_matches.sort(
                key=lambda chunk: (
                    normalize_text(chunk.file_path.stem).lower().endswith("_cleaned"),
                    chunk.file_path.name,
                )
            )
            if sequence_matches:
                results = sequence_matches[:1]

        results = sorted(
            enumerate(results),
            key=lambda item: (
                0 if suffix_pattern.search(normalize_text(item[1].file_path.stem)) else 1,
                1 if normalize_text(item[1].file_path.stem).lower().endswith("_cleaned") else 0,
                item[0],
            ),
        )
        results = [result for _, result in results]

    chunks_by_file: dict[Path, list[DocumentChunk]] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    unique_results: list[DocumentChunk] = []
    seen_files: set[Path] = set()
    for result in results:
        if result.file_path in seen_files:
            continue
        seen_files.add(result.file_path)
        unique_results.append(result)
        if len(unique_results) >= limit:
            break

    if not unique_results:
        return []

    candidate_families = Counter(
        meeting_family_key(result.file_path) for result in unique_results[:3]
    )
    has_ambiguous_family = (
        requested_sequence is None
        and any(count >= 2 for count in candidate_families.values())
    )
    if has_ambiguous_family:
        return [
            full_document_context(
                result,
                chunks_by_file.get(result.file_path, [result]),
                max_document_chars,
            )
            for result in unique_results
        ]

    return build_summary_segments(
        unique_results[0],
        chunks_by_file.get(unique_results[0].file_path, [unique_results[0]]),
        max_segments=max(1, min(limit, 6)),
        max_document_chars=max_document_chars,
    )


def full_document_context(
    source: DocumentChunk,
    file_chunks: list[DocumentChunk],
    max_document_chars: int = 80_000,
) -> DocumentChunk:
    try:
        document_text = read_text_file(source.file_path)
    except OSError:
        document_text = "\n\n".join(
            chunk.text for chunk in sorted(file_chunks, key=lambda item: item.chunk_index)
        )
    return DocumentChunk(
        file_path=source.file_path,
        chunk_index=0,
        text=document_text[:max_document_chars],
        metadata=source.metadata,
        score=source.score,
        keyword_score=source.keyword_score,
        semantic_score=source.semantic_score,
    )


def build_summary_segments(
    source: DocumentChunk,
    file_chunks: list[DocumentChunk],
    max_segments: int = 6,
    max_document_chars: int = 80_000,
) -> list[DocumentChunk]:
    ordered = sorted(file_chunks, key=lambda item: item.chunk_index)
    if not ordered:
        return []

    total_chars = sum(len(chunk.text) for chunk in ordered)
    segment_count = min(max_segments, len(ordered))
    target_chars = max(1, min(max_document_chars, total_chars) // segment_count)
    groups: list[list[DocumentChunk]] = []
    current: list[DocumentChunk] = []
    current_chars = 0
    consumed_chars = 0

    for chunk in ordered:
        remaining_groups = segment_count - len(groups)
        remaining_chunks = len(ordered) - sum(len(group) for group in groups) - len(current)
        should_close = (
            current
            and len(groups) < segment_count - 1
            and current_chars >= target_chars
            and remaining_chunks >= remaining_groups - 1
        )
        if should_close:
            groups.append(current)
            current = []
            current_chars = 0
        if consumed_chars >= max_document_chars:
            break
        current.append(chunk)
        current_chars += len(chunk.text)
        consumed_chars += len(chunk.text)
    if current:
        groups.append(current)

    segments: list[DocumentChunk] = []
    for group in groups[:max_segments]:
        first_index = group[0].chunk_index
        last_index = group[-1].chunk_index
        text = "\n\n".join(chunk.text for chunk in group)
        segments.append(
            DocumentChunk(
                file_path=source.file_path,
                chunk_index=first_index,
                text=f"[元chunk {first_index}-{last_index}]\n{text}",
                metadata=source.metadata,
                score=source.score,
                keyword_score=source.keyword_score,
                semantic_score=source.semantic_score,
            )
        )
    return segments


def unsupported_amount_question(
    question: str,
    contexts: list[DocumentChunk],
) -> bool:
    normalized = unicodedata.normalize("NFKC", question)
    match = re.search(r"([一-龯々ァ-ヶーA-Za-z0-9]{2,})はいくら", normalized)
    if not match:
        return False
    answer_subject = match.group(1)
    context_text = normalize_text("\n".join(context.text for context in contexts))
    return answer_subject not in context_text


def answerability_failure_reason(
    question: str,
    contexts: list[DocumentChunk],
) -> str | None:
    if not contexts:
        return "質問に関連する会議ログ"

    normalized_question = unicodedata.normalize("NFKC", question)
    context_text = normalize_text(
        "\n".join(
            f"{context.file_path.stem} {context.metadata.meeting_name} {context.text}"
            for context in contexts
        )
    )
    topic_supported = has_question_topic_support(normalized_question, context_text)

    amount_match = re.search(
        r"([一-龯々ァ-ヶーA-Za-z0-9]{2,})は(?:いくら|何円)",
        normalized_question,
    )
    if amount_match:
        subject = amount_match.group(1)
        amount_pattern = re.compile(r"\d[\d,.]*\s*(?:円|万(?:円)?|億(?:円)?)")
        specific_terms = amount_question_terms(normalized_question)
        has_supported_amount = False
        for context in contexts:
            item_text = normalize_text(
                f"{context.file_path.stem} {context.metadata.meeting_name} {context.text}"
            )
            subject_supported = subject in item_text or (
                len(subject) >= 5
                and any(
                    subject[index : index + 5] in item_text
                    for index in range(len(subject) - 4)
                )
            )
            matched_terms = sum(term in item_text for term in specific_terms)
            if amount_pattern.search(item_text) and (
                subject_supported or matched_terms >= 2
            ):
                has_supported_amount = True
                break
        if not has_supported_amount:
            return f"{subject}の金額"

    asks_date = bool(
        re.search(r"(?:いつ|何日|何時|期限|締切)", normalized_question)
    )
    if asks_date:
        has_date = bool(
            re.search(
                r"(?:\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*時|"
                r"今日|明日|明後日|今週|来週|翌営業日|\d+\s*営業日|期限|締切)",
                context_text,
            )
        )
        if not has_date or not topic_supported:
            return "日時または期限"

    asks_count = bool(re.search(r"何\s*(?:人|名|件|回)", normalized_question))
    if asks_count:
        has_count = bool(re.search(r"\d[\d,]*\s*(?:人|名|件|回)", context_text))
        if not has_count or not topic_supported:
            return "人数または件数"

    person_match = re.search(
        r"([一-龯々ァ-ヶーA-Za-z0-9]{2,})は誰", normalized_question
    )
    if person_match:
        subject = person_match.group(1)
        generic_subjects = {"担当者", "責任者", "窓口", "その担当者"}
        asks_explicit_assignment = bool(
            re.search(r"誰に(?:任せ|お願い|依頼)|誰が(?:担当|対応|実施)", normalized_question)
        )
        has_subject = (
            subject in generic_subjects
            or subject in context_text
            or asks_explicit_assignment
        )
        has_assignment = bool(
            re.search(r"担当|任せ|対応|責任|お願い|依頼|やって|実施", context_text)
        )
        if not has_subject or not has_assignment or not topic_supported:
            return f"{subject}を特定できる担当情報"

    return None


def has_question_topic_support(question: str, context_text: str) -> bool:
    ignored_terms = {
        "期限",
        "締切",
        "日程",
        "日時",
        "担当者",
        "責任者",
        "窓口",
        "対象",
        "人数",
        "件数",
        "何日",
        "何時",
        "何人",
        "何名",
        "何件",
        "何回",
    }
    terms = [
        normalize_text(term)
        for term in re.findall(r"[一-龯々ァ-ヶーA-Za-z0-9]{2,}", question)
        if normalize_text(term) not in ignored_terms
        and not normalize_text(term).startswith("何")
    ]
    if not terms:
        return True

    for term in terms:
        if term in context_text:
            return True
        window_size = 5 if len(term) >= 5 else 3
        if len(term) >= window_size and any(
            term[index : index + window_size] in context_text
            for index in range(len(term) - window_size + 1)
        ):
            return True
    return False


def amount_question_terms(question: str) -> list[str]:
    ignored = {
        "今回",
        "今年",
        "年度",
        "会社全体",
        "金額",
        "予算",
        "年収",
        "月給",
        "売上",
        "目標",
        "何円",
    }
    terms = re.findall(r"[ァ-ヶー]{3,}|[一-龯々]{2,}|[A-Za-z][A-Za-z0-9]+", question)
    return [
        normalize_text(term)
        for term in terms
        if normalize_text(term) not in ignored
        and not normalize_text(term).isdigit()
    ]


def summary_ambiguity_message(
    question: str,
    contexts: list[DocumentChunk],
) -> str | None:
    if not is_summary_question(question) or requested_meeting_sequence(question):
        return None

    candidates: list[DocumentChunk] = []
    seen_paths: set[Path] = set()
    for context in contexts:
        if context.file_path in seen_paths:
            continue
        seen_paths.add(context.file_path)
        candidates.append(context)
        if len(candidates) >= 3:
            break
    families = Counter(meeting_family_key(context.file_path) for context in candidates)
    ambiguous_families = {family for family, count in families.items() if count >= 2}
    if not ambiguous_families:
        return None

    names = [
        context.file_path.name
        for context in candidates
        if meeting_family_key(context.file_path) in ambiguous_families
    ]
    options = "\n".join(f"- {name}" for name in names)
    return (
        "同名の会議ログが複数あり、対象回を一意に特定できません。"
        "次のいずれかをファイル名または回数で指定してください。\n"
        f"{options}"
    )


def is_advice_question(question: str) -> bool:
    return any(
        marker in normalize_text(question)
        for marker in (
            "どうしたら", "どうすれば", "どうしたい", "どう進め", "どう改善",
            "上手く", "うまく", "成功させ", "改善したい", "アドバイス", "提案して",
            "おすすめ", "相談", "べき", "には?", "には？",
        )
    )


def advice_search_scope(question: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    groups = (
        ("経理", "財務", "会計"),
        ("エンジニア", "開発職", "バックエンド", "フロントエンド", "技術職"),
    )
    normalized = normalize_text(question)
    terms = next((group for group in groups if any(t in normalized for t in group)), ())
    if "経理" in normalized:
        terms = ("経理",)
    if not terms:
        return chunks
    scoped = [
        chunk for chunk in chunks
        if any(term in normalize_text(f"{chunk.file_path.stem} {chunk.text}") for term in terms)
    ]
    if "フロー" in normalized:
        flow_terms = (
            "選考フロー", "先行フロー", "書類選考", "書類先行", "技術面接",
            "技術評価", "面接官", "スクリーニング",
        )
        scoped = [chunk for chunk in scoped if any(term in normalize_text(chunk.text) for term in flow_terms)]
    return scoped


def advice_follow_up(question: str) -> str:
    if "フロー" in question or "選考" in question:
        return "確認: 書類選考・技術面接・最終面接のうち、どの段階に課題がありますか？"
    return "確認: 応募数・選考通過・内定承諾のうち、いちばん困っているのはどこですか？"


def prepare_advice_search(
    question: str, chunks: list[DocumentChunk]
) -> tuple[str, list[DocumentChunk]]:
    unique_chunks = {}
    for chunk in advice_search_scope(question, chunks):
        unique_chunks.setdefault((normalize_text(chunk.text), chunk.chunk_index), chunk)
    if "フロー" in question or "選考" in question:
        question += " 書類選考 面接 評価基準 技術 選考スピード"
    question += " 判断基準 検討 理由"
    return question, list(unique_chunks.values())


def build_user_prompt(
    question: str,
    contexts: list[DocumentChunk],
    structured: bool = True,
    advice_mode: bool = False,
) -> str:
    if not contexts:
        response_instruction = (
            "通常の構造化形式で回答してください。"
            if structured
            else (
                "回答モード: 会話\n"
                "固定見出しを使わず、会話調で簡潔に回答してください。"
            )
        )
        return (
            "質問に関連する会議ログは見つかりませんでした。\n"
            "この前提で、回答ルールに従って答えてください。\n\n"
            f"質問:\n{question}\n\n"
            f"{response_instruction}"
        )

    context_text = "\n\n".join(
        (
            f"[S{index}] [source: {chunk.file_path.name} / chunk {chunk.chunk_index}]\n"
            f"会議名: {chunk.metadata.meeting_name}\n"
            f"日付: {chunk.metadata.meeting_date or '不明'}\n"
            f"発言者: {', '.join(chunk.metadata.speakers) or '不明'}\n"
            f"{chunk.text}"
        )
        for index, chunk in enumerate(contexts, start=1)
    )
    summary_instruction = (
        "\n会議全体の要約では、時系列の根拠区間ごとに内容を確認し、"
        "各箇条書きへ、その内容を直接示す最も近い根拠IDを付けてください。"
        "複数区間を統合した文には必要なIDをすべて付け、発言者であることだけを"
        "理由に担当者と推測しないでください。"
        if is_summary_question(question)
        else ""
    )
    response_instruction = (
        "次の見出しで整理してください。該当情報がなければ「確認できません」と書いてください。\n"
        "## 結論\n## 決定事項\n## 次の作業\n## 担当者\n## 期限\n## 未決事項\n## 根拠\n"
        "担当者には、原文で人物と作業の対応が明示されたものだけを書き、"
        "担当者不明の依頼は次の作業へ書いてください。"
        "事実を含む各行の末尾に、必ず [S1] の形式で根拠IDを付けてください。"
        f"{summary_instruction}"
        if structured
        else (
            "回答モード: 会話\n"
            "固定見出しは使わず、会話の流れを踏まえて質問に直接答えてください。"
            "短い質問には簡潔に、複数の事実がある場合は箇条書きで整理し、"
            "事実を含む各行の末尾に必ず [S1] の形式で根拠IDを付けてください。"
            f"{summary_instruction}"
        )
    )
    if advice_mode or is_advice_question(question):
        return (
            "回答モード: 相談\n"
            "利用者は過去の決定事項の照会ではなく改善方法を相談しています。"
            "今回の進め方を先に勧め、その理由として過去の会議で何を比較し、何を重視して"
            "判断しようとしていたかを説明してください。過去の課題や決定事項の一覧を主役にしないでください。"
            "過去の検討方法から今回に持ち込める考え方を取り出し、具体的な次の行動につなげます。"
            "例えば『まず必須条件と育成できる条件を分けてみてはどうでしょう。以前の会議でも"
            "条件の優先順位を整理していたので、今回も同じように比べると判断しやすそうです』"
            "のような構成です。この例の会議内容を事実として流用せず、実際の根拠に置き換えてください。"
            "原文が検討や相談の段階なら『決まった』『実施して成功した』と書かず、"
            "判断経緯が不明ならその旨を明示して、確認できる課題から控えめに提案してください。"
            "口調は同僚の相談に乗るような自然な『です・ます』調にしてください。"
            "『体制を構築する』『強化を図る』『提案します』を繰り返す報告書調は避け、"
            "『まず〜をそろえるとよさそうです』『例えば〜を書き出してみてください』のように、"
            "背景から次の行動へつながる説明にしてください。馴れ馴れしい口調や成果の断定は避けます。"
            "会議ログで確認できる課題・施策をヒントに、質問の職種とテーマに絞って回答してください。"
            "選考フローの相談は、選考の順序・判断基準・担当の分担・連絡導線に絞り、"
            "母集団形成や採用広報の話へ逸れないでください。"
            "採用全般の相談では、候補者像・求人条件・評価基準・候補者対応を優先してください。"
            "目安として内容の7割を会議ログに基づく説明と提案、3割をモデルの一般知識による補足にしてください。"
            "会議に基づく提案は最大3つ。重複させず、それぞれ『今回何をどう進めるとよさそうか』を先に、"
            "『その理由となる過去の検討・判断の考え方』を後に、合わせて180〜220字程度で説明してください。"
            "抽象的な『基準を明確にする』だけでなく、比較する項目、作る資料、確認する手順など"
            "すぐ実行できる行動まで具体化してください。未実施の行動案は提案として書いてください。"
            "背景の主語と対象は原文を厳密に維持し、社員への対応を候補者対応へ読み替えないでください。"
            "原文にある現状の数値だけを理由に、選考基準を厳格化すべきと結論しないでください。"
            "背景を直接裏付ける原文の短い一節をevidenceに格納してください。この一節は画面には表示しません。"
            "モデルの一般知識はsupplementに180〜250字程度で分離し、会議で言われたこととして扱わず、"
            "会議の引用IDは付けないでください。一般知識による方法や記入例を条件付きで示し、"
            "架空の担当者・会社の実績・成功率・決定済みの期限は作らないでください。"
            "会議で決まった事実と、あなたの提案を区別し、成果を保証しないでください。"
            "直接同じ相談への答えがなくても、根拠にある課題から条件付きの提案をして構いません。"
            "根拠にない人物・人数・日付・金額は補わないでください。"
            "確認質問はアプリ側で追加するので出力不要です。"
            "会議の事実と、それに基づく提案には[S1]形式の引用を必須にし、全体を700〜1000字程度にしてください。"
            "根拠が少ない場合は7対3に合わせて水増しせず、確認できる範囲だけ説明してください。\n\n"
            f"会議ログ抜粋:\n{context_text}\n\n質問:\n{question}"
        )
    return (
        "以下の会議ログ抜粋だけを根拠に回答してください。\n\n"
        "抜粋が質問への答えを直接示さない場合は、関連する話題があっても"
        "「会議ログ内では確認できません」と答えてください。\n"
        f"会議ログ抜粋:\n{context_text}\n\n"
        f"質問:\n{question}\n\n"
        f"{response_instruction}"
    )


def evidence_excerpt(text: str, max_chars: int = 600) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def cited_claim_has_unsupported_literals(
    line: str,
    cited_ids: set[str],
    contexts: list[DocumentChunk],
) -> bool:
    claim = re.sub(r"\[S\d+\]", "", normalize_text(line))
    evidence = normalize_text(
        "\n".join(contexts[int(source_id[1:]) - 1].text for source_id in cited_ids)
    )

    evidence_numbers = [
        float(value.replace(",", ""))
        for value in re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", evidence)
    ]
    for value in re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", claim):
        number = float(value.replace(",", ""))
        if any(math.isclose(number, candidate) for candidate in evidence_numbers):
            continue
        calculated = any(
            math.isclose(number, result)
            for left in evidence_numbers
            for right in evidence_numbers
            for result in (left + right, abs(left - right), left * right)
        )
        if not calculated:
            return True

    generic_names = {"担当者", "責任者", "候補者", "応募者", "皆"}
    claim_names = {
        name
        for name in re.findall(r"([一-龯々]{1,8})(?:さん|氏|様)", claim)
        if name not in generic_names
    }
    return any(name not in evidence for name in claim_names)


def validate_grounded_answer(
    answer: str,
    contexts: list[DocumentChunk],
    structured: bool = True,
    allow_advice: bool = False,
) -> str:
    valid_ids = {f"S{index}" for index in range(1, len(contexts) + 1)}
    citation_pattern = re.compile(r"\[(S\d+)\]")
    unknown_markers = ("確認できません", "会議ログ内では確認できません")
    lines: list[str] = []
    used_ids: set[str] = set()
    supplements: list[str] = []
    in_evidence_section = False

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_evidence_section = line == "## 根拠"
            if not in_evidence_section:
                lines.append(raw_line)
            continue
        if in_evidence_section:
            continue
        if not line:
            lines.append(raw_line)
            continue
        cited_ids = set(citation_pattern.findall(line))
        invalid_ids = cited_ids - valid_ids
        if allow_advice and line.startswith("一般的な補足（会議ログ未確認）:"):
            if (
                not cited_ids and not supplements and len(line) <= 550
                and not re.search(r"\d|[一-龯々]{1,8}(?:さん|氏|様)", line)
            ):
                supplements.append(line)
            continue
        if (
            allow_advice and not cited_ids
            and line.startswith("確認:") and line.endswith(("？", "?"))
        ):
            lines.append(raw_line)
            continue
        if invalid_ids or (not cited_ids and not any(marker in line for marker in unknown_markers)):
            continue
        if cited_ids and cited_claim_has_unsupported_literals(line, cited_ids, contexts):
            continue
        used_ids.update(cited_ids)
        lines.append(raw_line)

    if not used_ids and not any(marker in line for line in lines for marker in unknown_markers):
        return "会議ログ内では確認できません。検証可能な根拠がないため回答を省略しました。"

    if used_ids and supplements:
        lines.extend(["", supplements[0]])

    if structured:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.extend(["", "## 根拠"])
    elif used_ids:
        lines.extend(["", "根拠:"])

    for source_id in sorted(used_ids, key=lambda value: int(value[1:])):
        index = int(source_id[1:]) - 1
        chunk = contexts[index]
        lines.append(
            f"- [{source_id}] {chunk.file_path.name} / chunk {chunk.chunk_index}: "
            f"{evidence_excerpt(chunk.text)}"
        )
    return "\n".join(lines).strip()


def contextual_search_query(question: str, history: list[dict[str, str]]) -> str:
    contextual_markers = (
        "それ",
        "その",
        "これ",
        "あれ",
        "誰",
        "いつ",
        "どこ",
        "期限",
        "担当",
        "詳しく",
        "具体的",
        "続き",
    )
    needs_context = len(question.strip()) <= 14 or any(
        marker in question for marker in contextual_markers
    )
    if not needs_context:
        return question

    previous_questions = [
        message["content"]
        for message in reversed(history)
        if message["role"] == "user"
    ]
    if not previous_questions:
        return question
    previous_answers = [
        message["content"]
        for message in reversed(history)
        if message["role"] == "assistant"
    ]
    source_names: list[str] = []
    if previous_answers:
        source_names = re.findall(
            r"^- \[S\d+\]\s+(.+?)\s+/\s+chunk\s+\d+:",
            previous_answers[0],
            flags=re.MULTILINE,
        )
    source_context = (
        f"\n直前の回答で参照した会議: {' '.join(dict.fromkeys(source_names))}"
        if source_names
        else ""
    )
    return f"{previous_questions[0]}{source_context}\n続きの質問: {question}"


def trim_conversation_history(
    history: list[dict[str, str]],
    max_messages: int = 8,
    max_chars_per_message: int = 5000,
) -> list[dict[str, str]]:
    return [
        {
            "role": message["role"],
            "content": message["content"][-max_chars_per_message:],
        }
        for message in history[-max_messages:]
    ]


def answer_question(
    question: str,
    docs_dir: Path,
    model: str,
    provider: str,
    system_path: Path = DEFAULT_SYSTEM_PATH,
    top_k: int = 6,
    search_mode: str = "keyword",
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    conversation_history: list[dict[str, str]] | None = None,
    structured: bool = True,
    chunks: list[DocumentChunk] | None = None,
    system_prompt: str | None = None,
    deadline: float | None = None,
) -> str:
    history = trim_conversation_history(conversation_history or [])
    if chunks is None:
        chunks = load_documents(docs_dir)
    retrieval_question = contextual_search_query(question, history)
    advice = is_advice_question(retrieval_question)
    if advice:
        retrieval_question, chunks = prepare_advice_search(retrieval_question, chunks)
        if not chunks:
            return (
                "該当職種の会議ログは確認できません。まず採用要件と選考段階を整理したいです。"
                "今困っているのは、応募が集まらないこと、選考中の辞退、内定後の辞退のどれですか？"
            )
    contexts = retrieve(
        retrieval_question,
        chunks,
        limit=top_k,
        search_mode=search_mode,
        cache_path=cache_path,
        embedding_model=embedding_model,
        gbrain_timeout=min(2.5, max(0, deadline - time.monotonic() - 8)) if deadline else 2.5,
    )
    ambiguity_message = summary_ambiguity_message(question, contexts)
    if ambiguity_message is not None:
        return ambiguity_message
    failure_reason = answerability_failure_reason(question, contexts)
    if failure_reason is not None:
        return f"会議ログ内では確認できません（{failure_reason}を示す根拠がありません）。"
    if system_prompt is None:
        system_prompt = load_system_prompt(system_path)
    if advice:
        system_prompt += (
            "\n相談モードでは、会議ログに記載された事実を根拠にした条件付きの提案を許可する。"
            "『提案:』と『確認:』で事実と区別する。相談への直接回答が原文にないことだけを"
            "理由に拒否しない。会社の決定事項・実績を作らない。"
            "一般知識による補足を全体の約3割まで許可し、会議の事実と分離する。"
            "一般的な方法や例は提案として説明し、会議で実施されたと断定しない。"
        )
    user_prompt = build_user_prompt(question, contexts, structured=structured, advice_mode=advice)
    if not structured:
        user_prompt += (
            "\n回答は700〜1000字程度。会議の原文を長く引用せず、背景と具体策を説明してください。"
            if advice else "\n回答は質問に必要な背景と具体的な手順を含めて説明してください。原文の長い引用は不要です。"
        )

    if provider == "gemini":
        answer = answer_with_gemini(
            system_prompt,
            user_prompt,
            model,
            conversation_history=history,
            deadline=deadline,
        )
        validated = validate_grounded_answer(answer, contexts, structured=structured, allow_advice=advice)
        return f"{validated}\n\n{advice_follow_up(question)}" if advice else validated

    if provider != "openai":
        raise ValueError(f"未対応のproviderです: {provider}")

    answer = answer_with_openai(
        system_prompt,
        user_prompt,
        model,
        conversation_history=history,
        deadline=deadline,
    )
    validated = validate_grounded_answer(answer, contexts, structured=structured, allow_advice=advice)
    return f"{validated}\n\n{advice_follow_up(question)}" if advice else validated


def answer_with_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    conversation_history: list[dict[str, str]] | None = None,
    deadline: float | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai パッケージがインストールされていません。"
            "`pip install -r requirements.txt` を実行してください。"
        ) from exc

    client = OpenAI(timeout=max(0.1, deadline - time.monotonic()) if deadline else 60, max_retries=0)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": user_prompt})
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=0.2,
    )
    return response.output_text


def evidence_quote_matches(quote: str, source_text: str) -> bool:
    compact_source = re.sub(r"\s+", "", normalize_text(source_text))
    compact_quote = re.sub(r"\s+", "", normalize_text(quote))
    if len(compact_quote) >= 10 and compact_quote in compact_source:
        return True
    # Transcript speaker headings can interrupt an otherwise verbatim quote.
    return any(
        len(sentence) >= 20 and sentence in compact_source
        for sentence in re.split(r"[。！？\n]", compact_quote)
    )


def answer_with_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    conversation_history: list[dict[str, str]] | None = None,
    deadline: float | None = None,
) -> str:
    api_key = get_gemini_api_key()
    model_name = model.removeprefix("models/")
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in (conversation_history or [])
    ]
    contents.append(
        {
            "role": "user",
            "parts": [{"text": user_prompt}],
        }
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2800 if deadline else 4096,
        },
    }
    advice_output = "回答モード: 相談" in user_prompt
    if advice_output:
        source_ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\] \[source:", user_prompt)))
        source_texts = dict(re.findall(
            r"\[(S\d+)\] \[source:[^\n]+\]\n(.*?)(?=\n\n\[S\d+\] \[source:|\n\n質問:|\Z)",
            user_prompt, flags=re.DOTALL,
        ))
        payload["generationConfig"].update({
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "proposals": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "text": {
                                    "type": "STRING",
                                    "description": "今回の相談への行動提案を先に話す。『まず〜してはどうでしょう』『例えば〜を比べるとよさそうです』と具体的な手順を勧める。過去の事実の羅列から始めない。100〜140文字程度。引用IDは含めない。",
                                },
                                "background": {
                                    "type": "STRING",
                                    "description": "この提案を勧める理由。過去の会議で何を比べ何を重視して検討したかを原文の範囲で説明し、その考え方が今回どう役立つかにつなぐ。未決を決定済みや成功例としない。判断経緯が不明なら明示し、確認できた課題だけを根拠にする。60〜100文字程度。引用IDは含めない。",
                                },
                                "evidence": {
                                    "type": "STRING",
                                    "description": "背景を直接示す原文の一節を改変せず20〜100文字程度で転記する。候補者と社員など主語・対象を変えない。",
                                },
                                "sources": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING", "enum": source_ids},
                                },
                            },
                            "required": ["background", "evidence", "text", "sources"],
                        },
                    },
                    "supplement": {
                        "type": "STRING",
                        "description": "会議ログによらない一般知識の補足。180〜250文字程度で具体的な方法や記入例を説明。会議の事実や決定事項と混同しない。引用ID・数値・人物名・成果保証は禁止。",
                    },
                },
                "required": ["proposals", "supplement"],
            },
        })
    def generate(selected_model: str) -> dict:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{selected_model}:generateContent?key={api_key}"
        )
        if selected_model.startswith("gemini-2.5-flash"):
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        if deadline is None:
            return post_gemini_json(url, payload)
        remaining = deadline - time.monotonic()
        if remaining <= 0.2:
            raise TimeoutError("回答の待機時間上限に達しました。")
        return post_gemini_json(url, payload, timeout=min(12, remaining), retries=1)

    try:
        result = generate(model_name)
    except RuntimeError as exc:
        fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"
        ).removeprefix("models/")
        if "Gemini API error 503" not in str(exc) or fallback_model == model_name:
            raise
        result = generate(fallback_model)

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini APIの回答候補が空です: {result}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Gemini APIの回答テキストが空です: {result}")

    result_text = "\n".join(text_parts)
    if advice_output:
        result_data = json.loads(result_text)
        lines = []
        for proposal in result_data.get("proposals", [])[:3]:
            sources = [source for source in proposal.get("sources", []) if source in source_ids]
            quote = proposal.get("evidence", "")
            if not any(
                evidence_quote_matches(quote, source_texts.get(source, ""))
                for source in sources
            ):
                continue
            citations = " ".join(f"[{source}]" for source in sources)
            if citations:
                background = re.sub(r"\[S\d+\]", "", proposal.get("background", ""))
                background = re.sub(r"\s+", " ", background).strip()
                text = re.sub(r"\[S\d+\]", "", proposal["text"])
                text = re.sub(r"\s+", " ", text).strip()
                lines.append(f"提案: {text} {citations}")
                if background:
                    lines.append(f"参考にした考え方: {background} {citations}")
                lines.append("")
        supplement = re.sub(r"\[S\d+\]", "", result_data.get("supplement", ""))
        supplement = re.sub(r"\s+", " ", supplement).strip()
        if len(supplement) > 300:
            sentences = re.findall(r"[^。！？]+[。！？]", supplement)
            selected = []
            for sentence in sentences:
                if sum(map(len, selected)) + len(sentence) > 300:
                    break
                selected.append(sentence)
            supplement = "".join(selected)
        if supplement:
            lines.append(f"一般的な補足（会議ログ未確認）: {supplement}")
        return "\n".join(lines)
    return result_text


def get_env_or_windows_user_env(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    if os.name != "nt":
        return None

    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value if value else None
    except OSError:
        return None


def load_gemini_key_from_goal_file(goal_path: Path = DEFAULT_GOAL_PATH) -> str | None:
    if not goal_path.exists():
        return None

    text = goal_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"(?m)^\s*(AIza[0-9A-Za-z_\-]+|AQ\.[0-9A-Za-z_\-]+)\s*$", text
    )
    return matches[-1] if matches else None


def run_chat(
    docs_dir: Path,
    model: str,
    provider: str,
    system_path: Path,
    top_k: int,
    search_mode: str,
    cache_path: Path,
    embedding_model: str,
    initial_question: str | None = None,
) -> None:
    chunks = load_documents(docs_dir)
    system_prompt = load_system_prompt(system_path)
    history: list[dict[str, str]] = []
    pending_question = initial_question

    print("会議ログAIチャットを開始しました。")
    print("終了: /exit  履歴を消去: /clear  ヘルプ: /help")

    while True:
        try:
            if pending_question is not None:
                question = pending_question
                pending_question = None
                print(f"\nあなた> {question}")
            else:
                question = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nチャットを終了しました。")
            return

        if not question:
            continue
        command = question.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            print("チャットを終了しました。")
            return
        if command == "/clear":
            history.clear()
            print("会話履歴を消去しました。")
            continue
        if command == "/help":
            print("/exit: 終了  /clear: 会話履歴を消去  /help: この案内を表示")
            continue

        try:
            answer = answer_question(
                question=question,
                docs_dir=docs_dir,
                model=model,
                provider=provider,
                system_path=system_path,
                top_k=top_k,
                search_mode=search_mode,
                cache_path=cache_path,
                embedding_model=embedding_model,
                conversation_history=history,
                structured=False,
                chunks=chunks,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            print(f"\nエラー> {exc}")
            continue

        print(f"\nAI> {answer}")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        history = trim_conversation_history(history)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Markdown化した会議ログに質問するHermes Agent")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="会議ログに対する質問。--chatでは最初の質問として使用",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Markdown/txt会議ログの保存フォルダ",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Drive URLとローカル同期先を持つ設定ファイル",
    )
    parser.add_argument(
        "--system",
        type=Path,
        default=DEFAULT_SYSTEM_PATH,
        help="Agentのsystemプロンプト",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="利用するLLMプロバイダー",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="利用するモデル。未指定時はproviderごとの既定値",
    )
    parser.add_argument("--top-k", type=int, default=6, help="回答に使う検索結果数")
    parser.add_argument(
        "--search",
        choices=["local", "hybrid", "keyword", "gbrain", "enhanced", "parallel"],
        default="keyword",
        help=(
            "検索方式。keywordはBM25、gbrainはGBrain、enhancedは両者の順位融合、"
            "localはローカルEmbedding、hybridは従来のGemini文書Embedding"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="意味検索に使うGemini Embeddingモデル",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_PATH,
        help="埋め込みキャッシュの保存先",
    )
    parser.add_argument(
        "--show-search-results",
        action="store_true",
        help="回答生成を行わず検索結果だけ表示",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保持する対話型チャットを開始",
    )
    args = parser.parse_args()
    docs_dir = resolve_docs_dir(args.docs_dir, args.config)
    model = args.model
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if args.provider == "gemini" else os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if args.chat:
        run_chat(
            docs_dir=docs_dir,
            model=model,
            provider=args.provider,
            system_path=args.system,
            top_k=args.top_k,
            search_mode=args.search,
            cache_path=args.embedding_cache,
            embedding_model=args.embedding_model,
            initial_question=args.question,
        )
        return

    if args.question is None:
        parser.error("質問を指定するか、--chat を付けてください。")

    if args.show_search_results:
        chunks = load_documents(docs_dir)
        contexts = retrieve(
            args.question,
            chunks,
            limit=args.top_k,
            search_mode=args.search,
            cache_path=args.embedding_cache,
            embedding_model=args.embedding_model,
        )
        for rank, chunk in enumerate(contexts, start=1):
            print(
                f"{rank}. {chunk.file_path.name} / chunk {chunk.chunk_index} "
                f"(score={chunk.score:.4f}, semantic={chunk.semantic_score:.4f}, "
                f"keyword={chunk.keyword_score:.4f})"
            )
            print(
                f"   会議名={chunk.metadata.meeting_name} "
                f"日付={chunk.metadata.meeting_date or '不明'} "
                f"発言者={', '.join(chunk.metadata.speakers) or '不明'}"
            )
        return

    print(
        answer_question(
            question=args.question,
            docs_dir=docs_dir,
            model=model,
            provider=args.provider,
            system_path=args.system,
            top_k=args.top_k,
            search_mode=args.search,
            cache_path=args.embedding_cache,
            embedding_model=args.embedding_model,
        )
    )


if __name__ == "__main__":
    main()
