from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIMESTAMP_PATTERN = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$")
PUNCT_ONLY_PATTERN = re.compile(r"^\s*[、。，．・･.。！？!?ー\-…]+\s*$")


DEFAULT_GREETINGS = [
	"おはよう",
	"おはようございます",
	"こんにちは",
	"こんばんは",
	"失礼します",
	"お願いします",
	"お願いいたします",
	"よろしくお願いします",
	"よろしくお願いいたします",
	"ありがとうございます",
	"ありがとうございました",
	"すみません",
	"すいません",
]


@dataclass(frozen=True)
class TextProcessingOptions:
	greetings: tuple[str, ...] = tuple(DEFAULT_GREETINGS)
	min_chars: int = 2000
	max_chars: int = 2000
	overlap_chars: int = 180


@dataclass(frozen=True)
class TextProcessingResult:
	cleaned_text: str
	chunks: list[str]


def find_split_point(text: str, start: int, min_end: int, max_end: int) -> int:
	"""Find a readable split point between min_end and max_end."""
	window = text[min_end:max_end]
	if not window:
		return max_end

	newline_pos = window.rfind("\n")
	if newline_pos != -1:
		return min_end + newline_pos + 1

	period_pos = window.rfind("。")
	if period_pos != -1:
		return min_end + period_pos + 1

	return max_end


def adjust_split_end_to_avoid_timestamp(text: str, start: int, end: int) -> int:
	"""Move end backward so a chunk does not end with only a timestamp or punctuation line."""
	adjusted_end = end
	while adjusted_end > start:
		while adjusted_end > start and text[adjusted_end - 1] in " \t\r\n":
			adjusted_end -= 1

		if adjusted_end <= start:
			return start

		line_start = text.rfind("\n", start, adjusted_end)
		if line_start == -1:
			line_start = start
		else:
			line_start += 1

		last_line = text[line_start:adjusted_end].strip()
		if TIMESTAMP_PATTERN.match(last_line) or PUNCT_ONLY_PATTERN.match(last_line):
			adjusted_end = line_start
			continue

		break

	return adjusted_end


def find_timestamp_starts(text: str, start: int, end: int) -> list[int]:
	"""Return start indexes of timestamp-only lines in the specified range."""
	if start < 0:
		start = 0
	if end <= start or start >= len(text):
		return []

	positions: list[int] = []
	line_start = text.rfind("\n", 0, start)
	line_start = 0 if line_start == -1 else line_start + 1

	while line_start < len(text) and line_start < end:
		line_end = text.find("\n", line_start)
		if line_end == -1:
			line_end = len(text)

		if line_start >= start and line_start < end:
			line = text[line_start:line_end].strip()
			if TIMESTAMP_PATTERN.match(line):
				positions.append(line_start)

		line_start = line_end + 1

	return positions


def find_first_timestamp_start(text: str) -> int | None:
	positions = find_timestamp_starts(text, 0, len(text))
	if not positions:
		return None
	return positions[0]


def adjust_next_start_to_timestamp(
	text: str,
	current_start: int,
	proposed_start: int,
	current_end: int,
) -> int:
	"""Move the next chunk start to the nearest timestamp line when possible."""
	lower = current_start + 1
	upper = min(current_end, len(text))
	if lower >= upper:
		return proposed_start

	if proposed_start < lower:
		proposed_start = lower
	if proposed_start > upper:
		proposed_start = upper

	positions = find_timestamp_starts(text, lower, upper)
	if not positions:
		return proposed_start

	before_or_equal = [pos for pos in positions if pos <= proposed_start]
	if before_or_equal:
		return before_or_equal[-1]

	return positions[0]


def split_text_with_overlap(
	text: str,
	min_chars: int = 2000,
	max_chars: int = 2000,
	overlap_chars: int = 180,
) -> list[str]:
	"""Split text into overlapping chunks."""
	if min_chars <= 0 or max_chars <= 0:
		raise ValueError("min_chars/max_chars は 1 以上にしてください")
	if min_chars > max_chars:
		raise ValueError("min_chars は max_chars 以下にしてください")
	if overlap_chars < 0:
		raise ValueError("overlap_chars は 0 以上にしてください")

	text = text.strip()
	if not text:
		return []

	chunks: list[str] = []
	start = 0
	text_len = len(text)
	first_timestamp_start = find_first_timestamp_start(text)
	if first_timestamp_start is not None:
		start = first_timestamp_start

	while start < text_len:
		remaining = text_len - start
		if remaining <= max_chars:
			end = adjust_split_end_to_avoid_timestamp(text, start, text_len)
			if end <= start:
				end = text_len
			chunk = text[start:end].strip()
			if chunk:
				chunks.append(chunk)
			break

		min_end = min(start + min_chars, text_len)
		max_end = min(start + max_chars, text_len)
		end = find_split_point(text, start, min_end, max_end)
		end = adjust_split_end_to_avoid_timestamp(text, start, end)

		if end <= start:
			end = max_end
			end = adjust_split_end_to_avoid_timestamp(text, start, end)
			if end <= start:
				end = max_end

		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)

		next_start = max(0, end - overlap_chars)
		if next_start <= start:
			next_start = end

		next_start = adjust_next_start_to_timestamp(text, start, next_start, end)
		if next_start <= start:
			next_start = end
			next_start = adjust_next_start_to_timestamp(text, start, next_start, end)
			if next_start <= start:
				next_start = end

		start = next_start

	return chunks


def cleanup_orphan_timestamp_lines(text: str) -> str:
	lines = text.splitlines()
	result: list[str] = []
	i = 0

	while i < len(lines):
		line = lines[i]
		if not TIMESTAMP_PATTERN.match(line):
			if not PUNCT_ONLY_PATTERN.match(line):
				result.append(line)
			i += 1
			continue

		j = i + 1
		seen_content = False
		while j < len(lines):
			next_line = lines[j]
			if not next_line.strip() or PUNCT_ONLY_PATTERN.match(next_line):
				j += 1
				continue
			if TIMESTAMP_PATTERN.match(next_line):
				break
			seen_content = True
			break

		if seen_content:
			result.append(line)

		i += 1

	return "\n".join(result)


def remove_greetings(text: str, greetings: Iterable[str] = DEFAULT_GREETINGS) -> str:
	"""Remove configured greeting phrases from text."""
	cleaned = text

	escaped = [re.escape(word) for word in sorted(set(greetings), key=len, reverse=True) if word]
	if escaped:
		pattern = "|".join(escaped)
		cleaned = re.sub(pattern, "", cleaned)

	cleaned = cleanup_orphan_timestamp_lines(cleaned)
	cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

	return cleaned.strip()


def process_text(text: str, options: TextProcessingOptions | None = None) -> str:
	"""Normalize a text body. This is independent from files or databases."""
	options = options or TextProcessingOptions()
	return remove_greetings(text, options.greetings)


def process_text_to_chunks(text: str, options: TextProcessingOptions | None = None) -> TextProcessingResult:
	"""Normalize text and return both the full cleaned text and split chunks."""
	options = options or TextProcessingOptions()
	cleaned_text = process_text(text, options)
	chunks = split_text_with_overlap(
		cleaned_text,
		min_chars=options.min_chars,
		max_chars=options.max_chars,
		overlap_chars=options.overlap_chars,
	)
	return TextProcessingResult(cleaned_text=cleaned_text, chunks=chunks)


def load_text_from_file(input_file: Path) -> str:
	return input_file.read_text(encoding="utf-8")


def save_text_to_file(output_file: Path, text: str) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(text, encoding="utf-8")


def build_output_path(input_file: Path, output_dir: Path, suffix: str) -> Path:
	return output_dir / f"{input_file.stem}_{suffix}{input_file.suffix}"


def build_chunk_output_path(base_file: Path, chunk_index: int, split_suffix: str) -> Path:
	return base_file.parent / f"{base_file.stem}_{split_suffix}{chunk_index:03d}{base_file.suffix}"


def build_processing_options(args: argparse.Namespace) -> TextProcessingOptions:
	return TextProcessingOptions(
		greetings=tuple(DEFAULT_GREETINGS + (args.remove or [])),
		min_chars=args.min_chars,
		max_chars=args.max_chars,
		overlap_chars=args.overlap_chars,
	)


def process_one_file_to_source_subdir(
	input_file: Path,
	output_root_dir: Path,
	options: TextProcessingOptions,
	normalize_suffix: str,
	split_suffix: str,
) -> tuple[Path, int]:
	"""Save split files under output_root_dir/<source file name>/."""
	result = process_text_to_chunks(load_text_from_file(input_file), options)

	target_dir = output_root_dir / input_file.stem
	target_dir.mkdir(parents=True, exist_ok=True)

	for idx, chunk in enumerate(result.chunks, start=1):
		chunk_path = target_dir / f"{input_file.stem}_{normalize_suffix}_{split_suffix}{idx:03d}{input_file.suffix}"
		save_text_to_file(chunk_path, chunk)

	return target_dir, len(result.chunks)


def process_one_file(
	input_file: Path,
	output_file: Path,
	options: TextProcessingOptions,
	split_enabled: bool,
	split_suffix: str,
) -> None:
	result = process_text_to_chunks(load_text_from_file(input_file), options)
	save_text_to_file(output_file, result.cleaned_text)

	if not split_enabled:
		return

	for idx, chunk in enumerate(result.chunks, start=1):
		chunk_path = build_chunk_output_path(output_file, idx, split_suffix)
		save_text_to_file(chunk_path, chunk)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="文字起こしテキストから挨拶語を除去し、必要に応じて分割する"
	)
	parser.add_argument(
		"input_path",
		type=Path,
		nargs="?",
		default=Path("テキスト"),
		help="入力テキストファイル、または入力フォルダ",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="出力先。ファイル入力時は出力ファイル名、フォルダ入力時は出力フォルダ",
	)
	parser.add_argument(
		"--suffix",
		type=str,
		default="挨拶除去",
		help="出力ファイル名の末尾に付ける処理名",
	)
	parser.add_argument(
		"--remove",
		nargs="*",
		default=None,
		help="追加で削除したい語句",
	)
	parser.add_argument(
		"--split",
		action="store_true",
		help="処理後テキストを分割して保存する",
	)
	parser.add_argument(
		"--min-chars",
		type=int,
		default=2000,
		help="1チャンクの最小文字数",
	)
	parser.add_argument(
		"--max-chars",
		type=int,
		default=2000,
		help="1チャンクの最大文字数",
	)
	parser.add_argument(
		"--overlap-chars",
		type=int,
		default=180,
		help="チャンク間で重複させる文字数",
	)
	parser.add_argument(
		"--split-suffix",
		type=str,
		default="分割",
		help="分割ファイル名に付ける処理名",
	)
	parser.add_argument(
		"--split-subdir-by-source",
		action="store_true",
		help="フォルダ入力時に、出力先/元ファイル名/ の下へ分割ファイルを保存する",
	)

	args = parser.parse_args()
	options = build_processing_options(args)

	if not args.input_path.exists():
		raise FileNotFoundError(f"入力が見つかりません: {args.input_path}")

	if args.input_path.is_file():
		output_file = args.output or build_output_path(
			args.input_path,
			args.input_path.parent,
			args.suffix,
		)
		process_one_file(
			args.input_path,
			output_file,
			options,
			split_enabled=args.split,
			split_suffix=args.split_suffix,
		)
		print(f"保存: {output_file}")
		return

	if args.input_path.is_dir():
		output_dir = args.output or args.input_path
		output_dir.mkdir(parents=True, exist_ok=True)

		files = sorted(args.input_path.glob("*.txt"))
		if not files:
			print("処理対象の .txt ファイルが見つかりませんでした。")
			return

		processed_count = 0
		for input_file in files:
			if f"_{args.suffix}" in input_file.stem:
				continue

			if args.split and args.split_subdir_by_source:
				target_dir, chunk_count = process_one_file_to_source_subdir(
					input_file,
					output_dir,
					options,
					normalize_suffix=args.suffix,
					split_suffix=args.split_suffix,
				)
				processed_count += 1
				print(f"保存: {target_dir} ({chunk_count}分割)")
				continue

			output_file = build_output_path(input_file, output_dir, args.suffix)
			process_one_file(
				input_file,
				output_file,
				options,
				split_enabled=args.split,
				split_suffix=args.split_suffix,
			)
			processed_count += 1
			print(f"保存: {output_file}")

		print(f"完了: {processed_count}件")


if __name__ == "__main__":
	main()
