from __future__ import annotations

import unittest
import threading
import time
from unittest.mock import patch
from pathlib import Path
from hermes_agent import DocumentChunk, DocumentMetadata

from slack_bot import (
    SlackBotSettings,
    SlackHermesBot,
    clean_slack_question,
    create_slack_app,
    format_slack_answer,
    post_slack_answer,
    split_slack_message,
)


class SlackTextTests(unittest.TestCase):
    def test_general_supplement_paragraph_is_not_displayed(self) -> None:
        formatted = format_slack_answer("提案: 条件をそろえてはどうでしょう。[S1]\n一般的な補足（会議ログ未確認）: 不要な一般論の段落。")
        self.assertIn("条件をそろえてはどうでしょう。", formatted)
        self.assertNotIn("一般的な補足", formatted)
        self.assertNotIn("不要な一般論", formatted)

    def test_bot_mention_is_removed(self) -> None:
        self.assertEqual(
            "採用会議の決定事項は？",
            clean_slack_question(
                "<@U012ABC> 採用会議の決定事項は？", "U012ABC"
            ),
        )

    def test_markdown_headings_are_converted_for_slack(self) -> None:
        self.assertEqual(
            "*結論*\n回答です。",
            format_slack_answer("## 結論\n回答です。[S1]"),
        )

    def test_evidence_is_reduced_to_source_filename(self) -> None:
        answer = (
            "回答です。[S1]\n\n根拠:\n"
            "- [S1] オンボ-2.md / chunk 17: 実際の長い会話ログ原文"
        )

        formatted = format_slack_answer(answer, include_sources=True)

        self.assertIn("回答です。[S1]", formatted)
        self.assertIn("*参照ファイル*", formatted)
        self.assertIn("- [S1] オンボ-2.md", formatted)
        self.assertNotIn("chunk 17", formatted)
        self.assertNotIn("会話ログ原文", formatted)

    def test_sources_are_hidden_by_default_but_final_question_remains(self) -> None:
        answer = "提案: 条件を整理してはどうでしょう。[S1]\n\n根拠:\n- [S1] meeting.txt / chunk 1: 非表示の原文\n\n確認: どの段階で困っていますか？"
        formatted = format_slack_answer(answer)
        self.assertIn("条件を整理してはどうでしょう。", formatted)
        self.assertIn("どの段階で困っていますか？", formatted)
        self.assertNotIn("meeting.txt", formatted)
        self.assertNotIn("非表示の原文", formatted)
        self.assertNotIn("[S1]", formatted)
        self.assertNotIn("根拠:", formatted)

    def test_long_messages_are_split_without_data_loss(self) -> None:
        source = "段落1\n" + "あ" * 30 + "\n段落3"
        parts = split_slack_message(source, limit=20)

        self.assertGreater(len(parts), 1)
        self.assertEqual(source.replace("\n", ""), "".join(parts).replace("\n", ""))
        self.assertTrue(all(len(part) <= 20 for part in parts))

    def test_slack_rejection_retries_with_smaller_parts(self) -> None:
        class MessageTooLongError(Exception):
            response = {"error": "msg_too_long"}

        class FakeClient:
            def __init__(self) -> None:
                self.updates: list[str] = []
                self.posts: list[str] = []

            def chat_update(self, **kwargs) -> None:
                self.updates.append(kwargs["text"])
                if len(self.updates) == 1:
                    raise MessageTooLongError()

            def chat_postMessage(self, **kwargs) -> None:
                self.posts.append(kwargs["text"])

        client = FakeClient()
        answer = "あ" * 3000

        post_slack_answer(client, "C1", "100.1", "100.2", answer)

        delivered = client.updates[-1] + "".join(client.posts)
        self.assertEqual(answer, delivered)
        self.assertTrue(all(len(part) <= 900 for part in client.updates[1:] + client.posts))


class SlackBotTests(unittest.TestCase):
    def test_source_question_returns_only_previous_answer_files_without_api(self) -> None:
        raw_answer = "提案: 条件を整理してください。[S1]\n\n根拠:\n- [S1] meeting.txt / chunk 1: 原文\n"
        self.bot.conversations.append("C1", "1", "相談", raw_answer)
        result = self.bot.answer("どのファイルを参考にした？", "C1", "1")
        self.assertIn("meeting.txt", result)
        self.assertNotIn("chunk", result)
        self.assertNotIn("原文", result)
        self.assertEqual([], self.calls)
        self.assertNotIn("meeting.txt", self.bot.answer("参照ファイルは？", "C1", "2"))
        self.bot.answer("/clear", "C1", "1")
        self.assertNotIn("meeting.txt", self.bot.answer("参照ファイルは？", "C1", "1"))

    def test_slow_generation_returns_within_wait_budget_without_storing_history(self) -> None:
        release = threading.Event()
        def slow_answerer(**kwargs) -> str:
            release.wait(2)
            return "遅い回答"
        bot = SlackHermesBot(SlackBotSettings(Path("docs"), response_timeout=0.05), answerer=slow_answerer)
        try:
            started = time.monotonic()
            answer = bot.answer("質問", "C1", "1")
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertIn("時間内", answer)
            self.assertEqual([], bot.conversations.get("C1", "1"))
        finally:
            release.set()
            bot.executor.shutdown(wait=True)

    def test_preloaded_documents_are_reused_between_questions(self) -> None:
        snapshot = [DocumentChunk(Path("meeting.txt"), 1, "会議ログ", DocumentMetadata("定例", None, (), None))]
        with (
            patch("slack_bot.load_documents", return_value=snapshot) as loader,
            patch("slack_bot.load_system_prompt", return_value="system"),
        ):
            self.bot.refresh_documents()
            self.bot.answer("質問1", "C1", "1")
            self.bot.answer("質問2", "C1", "1")
            self.assertEqual(1, loader.call_count)
            loader.return_value = []
            with self.assertRaises(ValueError):
                self.bot.refresh_documents()
        self.assertIs(snapshot, self.bot.chunks)
        self.assertIs(self.calls[0]["chunks"], self.calls[1]["chunks"])

    def setUp(self) -> None:
        self.calls: list[dict] = []

        def fake_answerer(**kwargs) -> str:
            self.calls.append(kwargs)
            return f"回答: {kwargs['question']}"

        self.bot = SlackHermesBot(
            SlackBotSettings(
                docs_dir=Path("docs"),
                allowed_channel_id="C_ALLOWED",
            ),
            answerer=fake_answerer,
        )

    def test_channel_is_restricted_when_configured(self) -> None:
        self.assertTrue(self.bot.channel_is_allowed("C_ALLOWED"))
        self.assertFalse(self.bot.channel_is_allowed("C_OTHER"))

    def tearDown(self) -> None:
        self.bot.executor.shutdown(wait=True)

    def test_thread_history_is_passed_to_follow_up(self) -> None:
        first = self.bot.answer("最初の質問", "C_ALLOWED", "100.1")
        second = self.bot.answer("その担当者は？", "C_ALLOWED", "100.1")

        self.assertEqual("回答: 最初の質問", first)
        self.assertEqual("回答: その担当者は？", second)
        self.assertEqual([], self.calls[0]["conversation_history"])
        self.assertEqual("最初の質問", self.calls[1]["conversation_history"][0]["content"])
        self.assertFalse(self.calls[1]["structured"])

    def test_clear_removes_thread_history_without_calling_api(self) -> None:
        self.bot.answer("質問", "C_ALLOWED", "100.1")
        result = self.bot.answer("/clear", "C_ALLOWED", "100.1")

        self.assertIn("削除", result)
        self.assertEqual([], self.bot.conversations.get("C_ALLOWED", "100.1"))
        self.assertEqual(1, len(self.calls))

    def test_duplicate_event_is_rejected(self) -> None:
        self.assertTrue(self.bot.mark_event_seen("C_ALLOWED", "100.1"))
        self.assertFalse(self.bot.mark_event_seen("C_ALLOWED", "100.1"))

    def test_slack_listeners_can_be_registered_without_network(self) -> None:
        try:
            app = create_slack_app(self.bot, "xoxb-test-token", verify_token=False)
        except ImportError:
            self.skipTest("slack-bolt is not installed")

        self.assertEqual(2, len(app._listeners))


if __name__ == "__main__":
    unittest.main()
