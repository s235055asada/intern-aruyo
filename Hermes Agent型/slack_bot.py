from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hermes_agent import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDING_CACHE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SYSTEM_PATH,
    answer_question,
    cached_search_tokens,
    get_env_or_windows_user_env,
    load_documents,
    load_system_prompt,
    resolve_docs_dir,
    trim_conversation_history,
)


LOGGER = logging.getLogger("hermes.slack")
SLACK_MESSAGE_LIMIT = 2800


def clean_slack_question(text: str, bot_user_id: str | None = None) -> str:
    if bot_user_id:
        text = re.sub(rf"<@{re.escape(bot_user_id)}>", "", text, count=1)
    else:
        text = re.sub(r"<@[A-Z0-9]+>", "", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def format_slack_answer(answer: str) -> str:
    lines = []
    for line in answer.splitlines():
        source = re.fullmatch(
            r"- \[(S\d+)\]\s+(.+?)\s+/\s+chunk\s+\d+:.*",
            line.strip(),
        )
        if source:
            lines.append(f"- [{source.group(1)}] {source.group(2)}")
            continue
        if line.strip() == "根拠:":
            lines.append("*参照ファイル*")
            continue
        if line.startswith("会議での背景: "):
            line = line.removeprefix("会議での背景: ")
        elif line.startswith("提案: "):
            line = line.removeprefix("提案: ")
        elif line.startswith("確認: "):
            line = line.removeprefix("確認: ")
        elif line.startswith("一般的な補足（会議ログ未確認）: "):
            line = "ここからは会議で決まったことではなく、一般的な進め方の補足です。" + line.removeprefix("一般的な補足（会議ログ未確認）: ")
        heading = re.fullmatch(r"##\s+(.+)", line.strip())
        if heading:
            title = "参照ファイル" if heading.group(1) == "根拠" else heading.group(1)
            lines.append(f"*{title}*")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def split_slack_message(text: str, limit: int = SLACK_MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return parts


def post_slack_answer(
    client: Any,
    channel_id: str,
    thread_ts: str,
    progress_ts: str,
    answer: str,
) -> None:
    parts = split_slack_message(answer)
    try:
        client.chat_update(channel=channel_id, ts=progress_ts, text=parts[0])
    except Exception as exc:
        response = getattr(exc, "response", {})
        error = response.get("error") if hasattr(response, "get") else None
        if error != "msg_too_long":
            raise
        parts = split_slack_message(answer, limit=900)
        client.chat_update(channel=channel_id, ts=progress_ts, text=parts[0])

    for part in parts[1:]:
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=part,
        )


@dataclass
class ThreadConversationStore:
    histories: dict[tuple[str, str], list[dict[str, str]]] = field(
        default_factory=dict
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def exists(self, channel: str, thread_ts: str) -> bool:
        with self.lock:
            return (channel, thread_ts) in self.histories

    def get(self, channel: str, thread_ts: str) -> list[dict[str, str]]:
        with self.lock:
            return [dict(item) for item in self.histories.get((channel, thread_ts), [])]

    def append(self, channel: str, thread_ts: str, question: str, answer: str) -> None:
        with self.lock:
            history = self.histories.setdefault((channel, thread_ts), [])
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            self.histories[(channel, thread_ts)] = trim_conversation_history(history)

    def clear(self, channel: str, thread_ts: str) -> None:
        with self.lock:
            self.histories.pop((channel, thread_ts), None)


@dataclass(frozen=True)
class SlackBotSettings:
    docs_dir: Path
    provider: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    search_mode: str = "keyword"
    top_k: int = 6
    allowed_channel_id: str | None = None
    system_path: Path = DEFAULT_SYSTEM_PATH
    embedding_cache: Path = DEFAULT_EMBEDDING_CACHE_PATH
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    response_timeout: float = 16.0
    refresh_seconds: float = 86400.0


class SlackHermesBot:
    def __init__(
        self,
        settings: SlackBotSettings,
        answerer: Callable[..., str] = answer_question,
    ) -> None:
        self.settings = settings
        self.answerer = answerer
        self.conversations = ThreadConversationStore()
        self.processed_events: set[tuple[str, str]] = set()
        self.event_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="hermes-answer")
        self.capacity = threading.BoundedSemaphore(3)
        self.active_threads: set[tuple[str, str]] = set()
        self.active_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.chunks = None
        self.system_prompt = None
        self.last_refresh = 0.0

    def refresh_documents(self) -> None:
        if not self.refresh_lock.acquire(blocking=False):
            return
        try:
            chunks = load_documents(self.settings.docs_dir)
            if not chunks:
                raise ValueError("参照先に読み取り可能な会議ログがありません。")
            for chunk in chunks:
                cached_search_tokens(
                    f"{chunk.file_path.stem} {chunk.metadata.meeting_name} {chunk.text}"
                )
            self.system_prompt = load_system_prompt(self.settings.system_path)
            self.chunks = chunks
            self.last_refresh = time.monotonic()
            LOGGER.info("文書キャッシュ更新: %dチャンク", len(chunks))
        finally:
            self.refresh_lock.release()

    def refresh_in_background(self) -> None:
        # A failed refresh keeps the previous snapshot and retries after a minute.
        self.last_refresh = time.monotonic() - self.settings.refresh_seconds + 60
        try:
            self.refresh_documents()
        except Exception:
            LOGGER.error("文書キャッシュ更新失敗。前回の資料を継続使用します。")

    def channel_is_allowed(self, channel_id: str) -> bool:
        allowed = self.settings.allowed_channel_id
        return allowed is None or channel_id == allowed

    def mark_event_seen(self, channel_id: str, event_ts: str) -> bool:
        key = (channel_id, event_ts)
        with self.event_lock:
            if key in self.processed_events:
                return False
            self.processed_events.add(key)
            if len(self.processed_events) > 2000:
                self.processed_events = set(list(self.processed_events)[-1000:])
        return True

    def answer(self, question: str, channel_id: str, thread_ts: str) -> str:
        if question.lower() in {"/clear", "clear", "履歴削除"}:
            self.conversations.clear(channel_id, thread_ts)
            return "このスレッドの会話履歴を削除しました。"
        if question.lower() in {"/help", "help", "使い方"}:
            return (
                "会議ログについて質問してください。最初はチャンネルでBotをメンションし、"
                "続きは同じスレッドへ投稿してください。`/clear`で履歴を削除できます。"
            )

        started = time.monotonic()
        key = (channel_id, thread_ts)
        with self.active_lock:
            if key in self.active_threads:
                return "このスレッドの前の質問を処理中です。回答後に続けて質問してください。"
            if not self.capacity.acquire(blocking=False):
                return "現在ほかの質問に回答中です。少し待ってから再送してください。"
            self.active_threads.add(key)

        if self.chunks is not None and started - self.last_refresh >= self.settings.refresh_seconds:
            threading.Thread(target=self.refresh_in_background, daemon=True).start()

        def generate() -> str:
            history = self.conversations.get(channel_id, thread_ts)
            return self.answerer(
                question=question,
                docs_dir=self.settings.docs_dir,
                model=self.settings.model,
                provider=self.settings.provider,
                system_path=self.settings.system_path,
                top_k=self.settings.top_k,
                search_mode=self.settings.search_mode,
                cache_path=self.settings.embedding_cache,
                embedding_model=self.settings.embedding_model,
                conversation_history=history,
                structured=False,
                chunks=self.chunks,
                system_prompt=self.system_prompt,
                deadline=started + self.settings.response_timeout - 0.5,
            )

        def release_capacity(future: Any) -> None:
            self.capacity.release()

        future = self.executor.submit(generate)
        future.add_done_callback(release_capacity)
        try:
            answer = future.result(timeout=self.settings.response_timeout)
            self.conversations.append(channel_id, thread_ts, question, answer)
            return answer
        except (FutureTimeout, TimeoutError):
            return "回答の生成が時間内に完了しませんでした。少し待ってから再送してください。"
        finally:
            def clear_active(finished: Any) -> None:
                with self.active_lock:
                    self.active_threads.discard(key)
            future.add_done_callback(clear_active)
            LOGGER.info("回答待機終了: %.2f秒", time.monotonic() - started)


def create_slack_app(
    bot: SlackHermesBot,
    bot_token: str,
    verify_token: bool = True,
) -> Any:
    try:
        from slack_bolt import App
        from slack_sdk import WebClient
    except ImportError as exc:
        raise ImportError(
            "slack-boltがインストールされていません。"
            "`python -m pip install -r requirements.txt`を実行してください。"
        ) from exc

    app = App(
        client=WebClient(token=bot_token, timeout=2, retry_handlers=[]),
        token_verification_enabled=verify_token,
    )

    def respond(event: dict[str, Any], client: Any, context: Any) -> None:
        channel_id = event.get("channel", "")
        event_ts = event.get("ts", "")
        if not channel_id or not event_ts or not bot.channel_is_allowed(channel_id):
            return
        if event.get("bot_id") or event.get("subtype"):
            return
        if not bot.mark_event_seen(channel_id, event_ts):
            return

        thread_ts = event.get("thread_ts") or event_ts
        bot_user_id = context.get("bot_user_id") if context else None
        question = clean_slack_question(event.get("text", ""), bot_user_id)
        if not question:
            question = "使い方"

        progress = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="会議ログを検索しています...",
        )
        try:
            answer = format_slack_answer(bot.answer(question, channel_id, thread_ts))
            LOGGER.info("Slack回答を投稿します: %d文字", len(answer))
            post_slack_answer(
                client,
                channel_id,
                thread_ts,
                progress["ts"],
                answer,
            )
        except Exception as exc:
            LOGGER.error("Slackの質問処理に失敗しました: %s", type(exc).__name__)
            error_message = (
                "Gemini APIが一時的に混雑しています。少し待ってから、同じ質問を"
                "もう一度送信してください。"
                if "Gemini API error 503" in str(exc)
                else "回答処理に失敗しました。Botを実行しているPCのログを確認してください。"
            )
            client.chat_update(
                channel=channel_id,
                ts=progress["ts"],
                text=error_message,
            )

    @app.event("app_mention")
    def handle_app_mention(event: dict[str, Any], client: Any, context: Any) -> None:
        respond(event, client, context)

    @app.event("message")
    def handle_thread_message(event: dict[str, Any], client: Any, context: Any) -> None:
        thread_ts = event.get("thread_ts")
        channel_id = event.get("channel", "")
        if not thread_ts or not bot.conversations.exists(channel_id, thread_ts):
            return
        bot_user_id = context.get("bot_user_id") if context else None
        if bot_user_id and f"<@{bot_user_id}>" in event.get("text", ""):
            return
        respond(event, client, context)

    return app


def required_setting(name: str) -> str:
    value = get_env_or_windows_user_env(name)
    if not value:
        raise RuntimeError(f"{name}が設定されていません。")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Agent Slack Bot")
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--search",
        choices=["local", "hybrid", "keyword", "gbrain", "enhanced"],
        default="keyword",
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    model = args.model
    if model is None:
        model = (
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
            if args.provider == "gemini"
            else os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        )
    settings = SlackBotSettings(
        docs_dir=resolve_docs_dir(args.docs_dir, args.config),
        provider=args.provider,
        model=model,
        search_mode=args.search,
        top_k=args.top_k,
        allowed_channel_id=get_env_or_windows_user_env("SLACK_CHANNEL_ID"),
    )
    bot_token = required_setting("SLACK_BOT_TOKEN")
    app_token = required_setting("SLACK_APP_TOKEN")
    bot = SlackHermesBot(settings)
    print("会議ログを読み込んでいます。完了後にSlackへ接続します。", flush=True)
    bot.refresh_documents()
    app = create_slack_app(bot, bot_token)

    try:
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError as exc:
        raise ImportError("slack-boltのSocket Modeを読み込めません。") from exc

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    channel_description = settings.allowed_channel_id or "Botが参加している全チャンネル"
    print(f"Hermes Slack Botを起動します。対象: {channel_description}")
    print(f"モデル: {settings.model} / 文書: {len(bot.chunks)}チャンク / 回答待機上限: {settings.response_timeout}秒", flush=True)
    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":
    main()
