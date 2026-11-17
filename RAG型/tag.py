from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def normalize_for_match(text: str) -> str:
	return re.sub(r"\s+", "", text).lower()


def is_primary_split_file(file_path: Path, normalize_suffix: str, split_suffix: str) -> bool:
	"""一次生成の分割ファイルのみを判定する。"""
	stem = file_path.stem
	if stem.count(f"_{normalize_suffix}") != 1:
		return False
	if stem.count(f"_{split_suffix}") != 1:
		return False
	return re.search(rf"_{re.escape(split_suffix)}\d{{3}}$", stem) is not None


def parse_tag_rows_from_pdf(pdf_path: Path) -> list[tuple[str, str]]:
	"""PDFから (メインタグ, サブタグ) の行を抽出する。"""
	try:
		from pypdf import PdfReader
	except ImportError as exc:
		raise ImportError("pypdf がインストールされていません。`pip install pypdf` を実行してください") from exc

	reader = PdfReader(str(pdf_path))
	rows: list[tuple[str, str]] = []

	for page in reader.pages:
		text = page.extract_text() or ""
		for line in text.splitlines():
			clean = re.sub(r"\s+", " ", line).strip()
			if not clean or clean == "メインタグ サブタグ":
				continue
			if clean.startswith("＝"):
				continue

			parts = clean.split(" ", 1)
			if len(parts) != 2:
				continue
			main_tag, sub_tag = parts[0].strip(), parts[1].strip()
			if not main_tag or not sub_tag:
				continue

			sub_tag = sub_tag.replace("SaaS＞ATS＝採用管理システム", "").strip()
			if not sub_tag:
				continue

			rows.append((main_tag, sub_tag))

	seen: set[tuple[str, str]] = set()
	unique_rows: list[tuple[str, str]] = []
	for row in rows:
		if row in seen:
			continue
		seen.add(row)
		unique_rows.append(row)

	return unique_rows


def build_keyword_candidates(sub_tag: str) -> list[str]:
	"""サブタグからマッチ用キーワード候補を生成する。"""
	base = sub_tag
	tokens = re.split(r"[\s/（）()・＞>]+", base)

	candidates = [base]
	for token in tokens:
		token = token.strip()
		if len(token) >= 2:
			candidates.append(token)

	extra_aliases = {
		"Findy": ["findy", "ファインディ"],
		"Linkedin": ["linkedin", "linkedin", "リンクトイン"],
		"HRMOS": ["hrmos", "ハーモス"],
		"JACリクルートメント": ["jac", "ジェイエーシー"],
	}
	for key, aliases in extra_aliases.items():
		if key in sub_tag:
			candidates.extend(aliases)

	seen: set[str] = set()
	result: list[str] = []
	for candidate in candidates:
		norm = normalize_for_match(candidate)
		if not norm or norm in seen:
			continue
		seen.add(norm)
		result.append(norm)

	return result


def tag_text(text: str, tag_rows: list[tuple[str, str]]) -> tuple[str, list[str]]:
	"""テキストに対して (メインタグ, サブタグ群) を推定する。"""
	norm_text = normalize_for_match(text)
	scored: list[tuple[int, str, str]] = []

	for main_tag, sub_tag in tag_rows:
		keywords = build_keyword_candidates(sub_tag)
		score = 0
		for keyword in keywords:
			if keyword and keyword in norm_text:
				score += max(1, min(len(keyword), 8))
		if score > 0:
			scored.append((score, main_tag, sub_tag))

	if not scored:
		return "未分類", []

	scored.sort(key=lambda x: x[0], reverse=True)
	best_main = scored[0][1]
	sub_tags = [sub for _, main, sub in scored if main == best_main]

	seen_sub: set[str] = set()
	unique_sub_tags: list[str] = []
	for sub in sub_tags:
		if sub in seen_sub:
			continue
		seen_sub.add(sub)
		unique_sub_tags.append(sub)
		if len(unique_sub_tags) >= 5:
			break

	return best_main, unique_sub_tags


def parse_llm_json_output(raw_output: str) -> dict:
	"""LLM出力からJSONオブジェクトを抽出する。"""
	text = raw_output.strip()
	if text.startswith("```"):
		text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
		text = re.sub(r"\s*```$", "", text)

	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			return parsed
	except json.JSONDecodeError:
		pass

	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("LLM出力からJSONを抽出できませんでした")

	parsed = json.loads(text[start:end + 1])
	if not isinstance(parsed, dict):
		raise ValueError("LLMのJSON出力がオブジェクト形式ではありません")
	return parsed


def build_llm_tag_prompt(text: str, tag_rows: list[tuple[str, str]]) -> str:
	"""タグ付け用プロンプトを構築する。"""
	main_to_subs: dict[str, list[str]] = {}
	for main_tag, sub_tag in tag_rows:
		main_to_subs.setdefault(main_tag, []).append(sub_tag)

	taxonomy_lines = []
	for main_tag, sub_tags in main_to_subs.items():
		taxonomy_lines.append(f"- {main_tag}: {' / '.join(sub_tags)}")

	taxonomy_text = "\n".join(taxonomy_lines)

	return (
		"あなたは会議ログのタグ付けアシスタントです。\n"
		"以下のタグ候補だけを使って分類してください。\n"
		"\n"
		"要件:\n"
		"1) main_tag は必ず1つ。候補外は不可。\n"
		"2) sub_tags は main_tag に属する候補から 0〜5 個。\n"
		"3) どれにも当てはまらなければ main_tag=未分類, sub_tags=[]。\n"
		"4) 出力は JSON のみ。説明文は出力しない。\n"
		"\n"
		"JSON形式:\n"
		"{\"main_tag\": \"...\", \"sub_tags\": [\"...\", \"...\"]}\n"
		"\n"
		"タグ候補:\n"
		f"{taxonomy_text}\n"
		"\n"
		"対象テキスト:\n"
		"<<<\n"
		f"{text}\n"
		">>>\n"
	)


def validate_llm_tag_result(parsed: dict, tag_rows: list[tuple[str, str]]) -> tuple[str, list[str]]:
	"""LLM出力を候補タグに照らして検証・整形する。"""
	allowed_main_to_subs: dict[str, set[str]] = {}
	for main_tag, sub_tag in tag_rows:
		allowed_main_to_subs.setdefault(main_tag, set()).add(sub_tag)

	main_tag = str(parsed.get("main_tag", "")).strip()
	raw_sub_tags = parsed.get("sub_tags", [])

	if isinstance(raw_sub_tags, str):
		raw_sub_tags = [raw_sub_tags]
	if not isinstance(raw_sub_tags, list):
		raw_sub_tags = []

	sub_tags = [str(item).strip() for item in raw_sub_tags if str(item).strip()]

	if main_tag == "未分類":
		return "未分類", []

	if main_tag not in allowed_main_to_subs:
		raise ValueError(f"候補外の main_tag が返されました: {main_tag}")

	valid_subs = allowed_main_to_subs[main_tag]
	filtered: list[str] = []
	for sub in sub_tags:
		if sub in valid_subs and sub not in filtered:
			filtered.append(sub)
		if len(filtered) >= 5:
			break

	return main_tag, filtered


def tag_text_with_llm_cli(
	text: str,
	tag_rows: list[tuple[str, str]],
	llm_command: str,
	llm_command_args: list[str],
	llm_timeout: int,
	llm_model: str | None,
) -> tuple[str, list[str]]:
	"""外部LLM CLI（例: Gemini CLI）でタグ付けする。"""
	prompt = build_llm_tag_prompt(text, tag_rows)
	args = list(llm_command_args)

	has_prompt_flag = any(arg in ("-p", "--prompt") for arg in args)
	if not has_prompt_flag:
		args.extend(["-p", "入力内容を要件どおりJSONのみで返してください。"])
	else:
		for idx, arg in enumerate(args):
			if arg in ("-p", "--prompt") and idx == len(args) - 1:
				args.append("入力内容を要件どおりJSONのみで返してください。")
				break

	if llm_model:
		args = ["-m", llm_model, *args]

	lower_cmd = llm_command.lower()
	if lower_cmd.endswith(".cmd") or lower_cmd.endswith(".bat"):
		command = ["cmd", "/c", llm_command, *args]
	else:
		command = [llm_command, *args]

	completed = subprocess.run(
		command,
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
		input=prompt,
		timeout=llm_timeout,
	)

	if completed.returncode != 0:
		stderr = (completed.stderr or "").strip()
		raise RuntimeError(f"LLMコマンド実行に失敗しました: {stderr}")

	stdout = (completed.stdout or "").strip()
	if not stdout:
		raise ValueError("LLMの出力が空でした")

	parsed = parse_llm_json_output(stdout)
	return validate_llm_tag_result(parsed, tag_rows)


def tag_text_with_transformers(
	text: str,
	tag_rows: list[tuple[str, str]],
	model,
	tokenizer,
	max_new_tokens: int,
	temperature: float,
) -> tuple[str, list[str]]:
	"""Transformersモデルでタグ付けする。"""
	prompt = build_llm_tag_prompt(text, tag_rows)
	inputs = tokenizer(prompt, return_tensors="pt")
	inputs = {key: value.to(model.device) for key, value in inputs.items()}

	output_ids = model.generate(
		**inputs,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		do_sample=temperature > 0,
		pad_token_id=tokenizer.eos_token_id,
	)
	decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	if decoded.startswith(prompt):
		decoded = decoded[len(prompt):].strip()

	parsed = parse_llm_json_output(decoded)
	return validate_llm_tag_result(parsed, tag_rows)


def tag_text_with_openai(
	text: str,
	tag_rows: list[tuple[str, str]],
	llm_model: str,
	max_output_tokens: int,
	temperature: float,
) -> tuple[str, list[str]]:
	"""OpenAI APIのGPTモデルでタグ付けする。"""
	if not os.environ.get("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY が設定されていません")

	try:
		from openai import OpenAI
	except ImportError as exc:
		raise ImportError("OpenAI Python SDK がインストールされていません。`pip install openai` を実行してください") from exc

	client = OpenAI()
	prompt = build_llm_tag_prompt(text, tag_rows)
	response = client.responses.create(
		model=llm_model,
		input=prompt,
		temperature=temperature,
		max_output_tokens=max_output_tokens,
	)

	raw_output = getattr(response, "output_text", "") or ""
	if not raw_output:
		raise ValueError("OpenAI APIの出力が空でした")

	parsed = parse_llm_json_output(raw_output)
	return validate_llm_tag_result(parsed, tag_rows)


def export_tags_csv(
	input_files: list[Path],
	tag_rows: list[tuple[str, str]],
	output_csv: Path,
	tagger: str,
	llm_backend: str,
	llm_command: str,
	llm_command_args: list[str],
	llm_timeout: int,
	llm_fallback_keyword: bool,
	llm_model: str | None,
	llm_max_new_tokens: int,
	llm_temperature: float,
) -> None:
	"""分割済みテキスト群にタグ付けしてCSV出力する。"""
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	llm_active = tagger == "llm"
	llm_client = None
	llm_tokenizer = None
	use_transformers = llm_backend == "transformers"
	use_openai = llm_backend == "openai"
	if llm_active and use_transformers:
		from transformers import AutoModelForCausalLM, AutoTokenizer

		if not llm_model:
			raise ValueError("Transformersバックエンドでは --llm-model が必須です")
		llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
		llm_client = AutoModelForCausalLM.from_pretrained(
			llm_model,
			torch_dtype="auto",
			device_map="auto",
		)

	with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["file", "main_tag", "sub_tags"])
		for file_path in input_files:
			text = file_path.read_text(encoding="utf-8")

			if llm_active:
				try:
					if use_openai:
						if not llm_model:
							raise ValueError("OpenAIバックエンドでは --llm-model が必須です")
						main_tag, sub_tags = tag_text_with_openai(
							text,
							tag_rows,
							llm_model=llm_model,
							max_output_tokens=llm_max_new_tokens,
							temperature=llm_temperature,
						)
					elif use_transformers:
						if llm_client is None or llm_tokenizer is None:
							raise RuntimeError("Transformersモデルの初期化に失敗しました")
						main_tag, sub_tags = tag_text_with_transformers(
							text,
							tag_rows,
							llm_client,
							llm_tokenizer,
							max_new_tokens=llm_max_new_tokens,
							temperature=llm_temperature,
						)
					else:
						main_tag, sub_tags = tag_text_with_llm_cli(
							text,
							tag_rows,
							llm_command=llm_command,
							llm_command_args=llm_command_args,
							llm_timeout=llm_timeout,
							llm_model=llm_model,
						)
				except Exception as exc:
					if not llm_fallback_keyword:
						raise
					print(f"[WARN] LLMタグ付けに失敗したため以降はキーワード判定へフォールバックします: {exc}")
					llm_active = False
					main_tag, sub_tags = tag_text(text, tag_rows)
			else:
				main_tag, sub_tags = tag_text(text, tag_rows)

			writer.writerow([
				file_path.as_posix(),
				main_tag,
				" | ".join(sub_tags),
			])


def main() -> None:
	parser = argparse.ArgumentParser(description="分割済みテキストにタグ付けしてCSV出力する")
	parser.add_argument(
		"--tag-pdf",
		type=Path,
		required=True,
		help="タグ定義PDFのパス",
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("テキスト"),
		help="タグ付け対象ファイルを探索するディレクトリ",
	)
	parser.add_argument(
		"--tag-target-glob",
		type=str,
		default="*_挨拶除去_分割[0-9][0-9][0-9].txt",
		help="タグ付け対象ファイルのglob",
	)
	parser.add_argument(
		"--normalize-suffix",
		type=str,
		default="挨拶除去",
		help="正規化ファイル名のサフィックス",
	)
	parser.add_argument(
		"--split-suffix",
		type=str,
		default="分割",
		help="分割ファイル名のサフィックス",
	)
	parser.add_argument(
		"--tag-output-csv",
		type=Path,
		default=Path("テキスト/タグ付け結果.csv"),
		help="タグ付け結果のCSV出力先",
	)
	parser.add_argument(
		"--tagger",
		type=str,
		choices=["keyword", "llm"],
		default="llm",
		help="タグ付け方式（keyword または llm）",
	)
	parser.add_argument(
		"--llm-backend",
		type=str,
		choices=["openai", "cli", "transformers"],
		default="openai",
		help="LLMバックエンド（openai、cli、transformers）",
	)
	parser.add_argument(
		"--llm-command",
		type=str,
		default="gemini",
		help="LLM実行コマンド（例: gemini）",
	)
	parser.add_argument(
		"--llm-command-args",
		nargs="*",
		default=["-p", "入力内容を要件どおりJSONのみで返してください。"],
		help="LLMコマンド引数（本文は標準入力で送信）",
	)
	parser.add_argument(
		"--llm-timeout",
		type=int,
		default=120,
		help="LLMコマンドのタイムアウト秒（既定: 120）",
	)
	parser.add_argument(
		"--llm-model",
		type=str,
		default="gpt-4.1-mini",
		help="LLMモデル名（例: gpt-4.1-mini）",
	)
	parser.add_argument(
		"--llm-max-new-tokens",
		type=int,
		default=256,
		help="LLMの最大生成トークン数（既定: 256）",
	)
	parser.add_argument(
		"--llm-temperature",
		type=float,
		default=0.2,
		help="LLMの温度パラメータ（既定: 0.2）",
	)
	parser.add_argument(
		"--no-llm-fallback",
		action="store_true",
		help="LLM失敗時にキーワード判定へフォールバックしない",
	)

	args = parser.parse_args()
	effective_tagger = args.tagger
	effective_llm_command = args.llm_command
	effective_llm_backend = args.llm_backend

	if args.tagger == "llm" and args.llm_backend == "cli" and shutil.which(args.llm_command) is None:
		message = f"LLMコマンドが見つかりません: {args.llm_command}"
		if args.no_llm_fallback:
			raise FileNotFoundError(message)
		print(f"[WARN] {message}。keyword方式でタグ付けします。")
		effective_tagger = "keyword"
	elif args.tagger == "llm" and args.llm_backend == "cli":
		resolved = shutil.which(args.llm_command)
		if resolved:
			effective_llm_command = resolved

	if not args.input_dir.exists():
		raise FileNotFoundError(f"入力ディレクトリが見つかりません: {args.input_dir}")

	tag_rows = parse_tag_rows_from_pdf(args.tag_pdf)
	target_files = [
		f for f in sorted(args.input_dir.rglob(args.tag_target_glob))
		if is_primary_split_file(f, args.normalize_suffix, args.split_suffix)
	]

	export_tags_csv(
		target_files,
		tag_rows,
		args.tag_output_csv,
		tagger=effective_tagger,
		llm_backend=effective_llm_backend,
		llm_command=effective_llm_command,
		llm_command_args=args.llm_command_args,
		llm_timeout=args.llm_timeout,
		llm_fallback_keyword=not args.no_llm_fallback,
		llm_model=args.llm_model,
		llm_max_new_tokens=args.llm_max_new_tokens,
		llm_temperature=args.llm_temperature,
	)
	print(f"タグCSV: {args.tag_output_csv} ({len(target_files)}件)")


if __name__ == "__main__":
	main()
