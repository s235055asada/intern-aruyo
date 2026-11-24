from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hermes_agent import (
    answerability_failure_reason,
    answer_with_gemini,
    advice_search_scope,
    build_user_prompt,
    is_advice_question,
    build_summary_segments,
    canonical_document_id,
    contextual_search_query,
    expand_search_text,
    expand_summary_contexts,
    gbrain_query_cache_key,
    has_question_topic_support,
    is_summary_question,
    load_gbrain_query_cache,
    load_gemini_key_from_goal_file,
    requested_meeting_sequence,
    retrieve_keyword,
    save_gbrain_query_cache,
    summary_ambiguity_message,
    unsupported_amount_question,
    parse_gbrain_search_output,
    tokenize_for_bm25,
    trim_conversation_history,
    validate_grounded_answer,
    DocumentChunk,
    DocumentMetadata,
)
from pathlib import Path


class ConversationTests(unittest.TestCase):
    def test_follow_up_question_uses_previous_question(self) -> None:
        history = [
            {"role": "user", "content": "オンボーディングについて教えて"},
            {"role": "assistant", "content": "担当者と手順が決まりました。"},
        ]

        query = contextual_search_query("その担当者は誰？", history)

        self.assertIn("オンボーディングについて教えて", query)
        self.assertIn("その担当者は誰？", query)

    def test_new_topic_does_not_use_previous_question(self) -> None:
        history = [
            {"role": "user", "content": "オンボーディングについて教えて"},
            {"role": "assistant", "content": "回答"},
        ]
        question = "HRオペレーションのAI化について説明して"

        self.assertEqual(question, contextual_search_query(question, history))

    def test_follow_up_question_uses_previous_answer_source(self) -> None:
        history = [
            {"role": "user", "content": "採用会議の決定事項は？"},
            {
                "role": "assistant",
                "content": (
                    "決定事項です。[S1]\n\n根拠:\n"
                    "- [S1] 採用定例-2.md / chunk 4: 対応方針を決めた。"
                ),
            },
        ]

        query = contextual_search_query("その会議の次の作業は？", history)

        self.assertIn("採用定例-2.md", query)

    def test_history_is_limited(self) -> None:
        history = [
            {"role": "user", "content": f"質問{i}"}
            for i in range(12)
        ]

        trimmed = trim_conversation_history(history, max_messages=4)

        self.assertEqual(4, len(trimmed))
        self.assertEqual("質問8", trimmed[0]["content"])


class SearchTests(unittest.TestCase):
    def test_gemini_key_can_be_loaded_from_goal_file_during_testing(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "goal.txt"
            dummy_key = "AI" + "za_test_key"
            path.write_text(f"目標\n{dummy_key}\n", encoding="utf-8")

            self.assertEqual(dummy_key, load_gemini_key_from_goal_file(path))

    def test_japanese_bm25_uses_character_ngrams(self) -> None:
        tokens = tokenize_for_bm25("重複応募")

        self.assertIn("重複", tokens)
        self.assertIn("複応", tokens)
        self.assertIn("重複応", tokens)

    def test_search_text_expands_known_synonyms(self) -> None:
        expanded = expand_search_text("連休前に承認された分はいつ止める？")

        self.assertIn("ゴールデンウィーク", expanded)
        self.assertIn("OK", expanded)
        self.assertIn("ストップ", expanded)

    def test_onboarding_query_expands_abbreviated_title(self) -> None:
        expanded = expand_search_text("オンボーディングについて教えて")

        self.assertIn("オンボ", expanded)
        self.assertIn("受け入れ", expanded)

    def test_abbreviated_meeting_title_beats_body_only_match(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        chunks = [
            DocumentChunk(
                Path("採用定例.md"),
                0,
                "オンボーディングという言葉を何度も話した。" * 10,
                metadata,
            ),
            DocumentChunk(
                Path("オンボ-2.md"),
                0,
                "候補者対応の手順を説明した。",
                DocumentMetadata("オンボ-2", None, (), None),
            ),
        ]

        results = retrieve_keyword("オンボーディングについて教えて", chunks, limit=2)

        self.assertEqual("オンボ-2.md", results[0].file_path.name)
        self.assertEqual(["オンボ-2.md"], [item.file_path.name for item in results])

    def test_search_text_expands_transcription_errors(self) -> None:
        expanded = expand_search_text(
            "書類選考後の選考フローと面接官、母集団形成、辞退率を確認したい"
        )

        self.assertIn("書類先行", expanded)
        self.assertIn("先行フロー", expanded)
        self.assertIn("先行缶", expanded)
        self.assertIn("母集団女性", expanded)
        self.assertIn("事態の率", expanded)

    def test_request_wording_and_person_name_prioritize_matching_task(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        chunks = [
            DocumentChunk(
                Path("general.md"), 0, "森さんが採用状況を説明した。", metadata
            ),
            DocumentChunk(
                Path("task.md"),
                0,
                "森さんに訴求ポイントの優先順位の調査をお願いした。",
                metadata,
            ),
        ]

        results = retrieve_keyword("森氏への依頼内容は？", chunks, limit=2)

        self.assertEqual("task.md", results[0].file_path.name)

    def test_keyword_results_are_diverse_by_file(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        chunks = [
            DocumentChunk(Path("first.md"), 0, "採用候補者への対応", metadata),
            DocumentChunk(Path("first.md"), 1, "採用候補者への連絡", metadata),
            DocumentChunk(Path("second.md"), 0, "採用候補者の確認", metadata),
        ]

        results = retrieve_keyword("採用候補者", chunks, limit=3)

        self.assertEqual(len(results), len({item.file_path for item in results}))

    def test_gbrain_output_is_parsed(self) -> None:
        output = "[0.9135] 人力_md/オンボ-2 -- 最初の行\n続きの行\n"

        self.assertEqual(
            [(0.9135, "人力_md/オンボ-2", "最初の行\n続きの行")],
            parse_gbrain_search_output(output),
        )

    def test_gbrain_query_cache_depends_on_document_content(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        first = [DocumentChunk(Path("meeting.md"), 1, "変更前", metadata)]
        second = [DocumentChunk(Path("meeting.md"), 1, "変更後", metadata)]

        self.assertNotEqual(
            gbrain_query_cache_key("質問", first, 5),
            gbrain_query_cache_key("質問", second, 5),
        )

    def test_gbrain_query_cache_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            save_gbrain_query_cache(path, {"key": "result"})

            self.assertEqual({"key": "result"}, load_gbrain_query_cache(path))

    def test_document_id_ignores_unicode_marks_and_punctuation(self) -> None:
        self.assertEqual(
            canonical_document_id("人力_md/オンボ-2"),
            canonical_document_id("人力_md/オンボ-2.md"),
        )

    def test_summary_question_is_detected(self) -> None:
        self.assertTrue(is_summary_question("この会議の会話を要約してまとめて"))
        self.assertFalse(is_summary_question("承認した担当者は誰？"))

    def test_summary_context_contains_the_whole_document(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        path = Path("meeting.md")
        chunks = [
            DocumentChunk(path, 0, "冒頭の議題", metadata, score=2.0),
            DocumentChunk(path, 1, "中盤の決定", metadata, score=1.0),
            DocumentChunk(path, 2, "終盤の次回対応", metadata, score=0.5),
        ]

        expanded = expand_summary_contexts("会議全体を要約して", [chunks[1]], chunks, limit=3)

        combined = "\n".join(segment.text for segment in expanded)
        self.assertIn("冒頭の議題", combined)
        self.assertIn("中盤の決定", combined)
        self.assertIn("終盤の次回対応", combined)
        self.assertGreater(len(expanded), 1)

    def test_summary_segments_cover_document_in_chronological_order(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        chunks = [
            DocumentChunk(Path("meeting.md"), index, f"内容{index}" * 20, metadata)
            for index in range(1, 9)
        ]

        segments = build_summary_segments(chunks[0], chunks, max_segments=3)

        self.assertEqual(3, len(segments))
        combined = "\n".join(segment.text for segment in segments)
        for index in range(1, 9):
            self.assertIn(f"内容{index}", combined)
        self.assertLess(combined.index("内容1"), combined.index("内容8"))

    def test_requested_meeting_sequence_is_prioritized(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        first = DocumentChunk(Path("定例.md"), 0, "初回", metadata)
        second = DocumentChunk(Path("定例-2.md"), 0, "2回目", metadata)

        expanded = expand_summary_contexts(
            "定例の2回目をまとめて", [first, second], [first, second], limit=2
        )

        self.assertEqual("定例-2.md", expanded[0].file_path.name)
        self.assertEqual("2", requested_meeting_sequence("定例の2回目をまとめて"))

    def test_hyphenated_meeting_sequence_is_loaded_from_all_chunks(self) -> None:
        metadata = DocumentMetadata("オンボ", None, (), None)
        first = DocumentChunk(Path("オンボ-2.md"), 0, "2回目", metadata, score=2.0)
        third = DocumentChunk(Path("オンボ-3.md"), 0, "3回目", metadata)

        expanded = expand_summary_contexts(
            "オンボ-3の会話を要約して", [first], [first, third], limit=1
        )

        self.assertEqual("オンボ-3.md", expanded[0].file_path.name)
        self.assertEqual(1, len(expanded))

    def test_ambiguous_summary_requests_clarification(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        contexts = [
            DocumentChunk(Path("定例.md"), 0, "初回", metadata),
            DocumentChunk(Path("定例-2.md"), 0, "2回目", metadata),
        ]

        message = summary_ambiguity_message("定例をまとめて", contexts)

        self.assertIsNotNone(message)
        self.assertIn("定例-2.md", message or "")
        self.assertIsNone(summary_ambiguity_message("定例の2回目をまとめて", contexts))

    def test_unsupported_amount_question_is_rejected(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        contexts = [
            DocumentChunk(Path("meeting.md"), 0, "2026年の採用目標は29名。", metadata)
        ]

        self.assertTrue(
            unsupported_amount_question(
                "2026年度の会社全体の売上目標はいくらですか？", contexts
            )
        )
        self.assertFalse(unsupported_amount_question("採用目標はいくらですか？", contexts))

    def test_answerability_rejects_missing_amount(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        contexts = [
            DocumentChunk(Path("meeting.md"), 0, "2026年の採用目標は29名。", metadata)
        ]

        reason = answerability_failure_reason(
            "2026年度の会社全体の売上目標はいくらですか？", contexts
        )

        self.assertIn("金額", reason or "")

    def test_answerability_accepts_supported_date_count_and_person(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        contexts = [
            DocumentChunk(
                Path("meeting.md"),
                0,
                "担当は山田さんにお願いする。5月7日までに候補者3名へ対応する。",
                metadata,
            )
        ]

        self.assertIsNone(answerability_failure_reason("期限はいつ？", contexts))
        self.assertIsNone(answerability_failure_reason("候補者は何名？", contexts))
        self.assertIsNone(answerability_failure_reason("担当者は誰？", contexts))

    def test_answerability_rejects_missing_date_count_and_person(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        contexts = [DocumentChunk(Path("meeting.md"), 0, "方針を検討した。", metadata)]

        self.assertIn(
            "日時", answerability_failure_reason("期限はいつ？", contexts) or ""
        )
        self.assertIn(
            "人数", answerability_failure_reason("対象は何名？", contexts) or ""
        )
        self.assertIn(
            "担当", answerability_failure_reason("社長は誰？", contexts) or ""
        )

    def test_answerability_rejects_unrelated_topic_with_dates_and_counts(self) -> None:
        metadata = DocumentMetadata("採用定例", None, (), None)
        contexts = [
            DocumentChunk(
                Path("meeting.md"),
                0,
                "採用面談は5月7日。候補者3名を山田さんが担当する。",
                metadata,
            )
        ]

        self.assertIn(
            "日時",
            answerability_failure_reason("社員旅行の日程はいつ？", contexts) or "",
        )
        self.assertIn(
            "人数",
            answerability_failure_reason("社員旅行の参加者は何名？", contexts) or "",
        )
        self.assertFalse(has_question_topic_support("社員旅行の日程はいつ？", contexts[0].text))


class GroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contexts = [
            DocumentChunk(
                file_path=Path("meeting.md"),
                chunk_index=2,
                text="候補者への返信は古いものから順番に対応する。",
                metadata=DocumentMetadata("会議", None, (), None),
            )
        ]

    def test_uncited_claim_is_removed(self) -> None:
        answer = "## 結論\n新しいものから返信します。\n\n## 根拠\n- meeting.md"

        validated = validate_grounded_answer(answer, self.contexts)

        self.assertNotIn("新しいものから", validated)
        self.assertIn("検証可能な根拠がない", validated)

    def test_valid_citation_gets_real_excerpt(self) -> None:
        answer = "## 結論\n古いものから返信します。[S1]\n\n## 根拠\n- 自由記述"

        validated = validate_grounded_answer(answer, self.contexts)

        self.assertIn("古いものから返信します。[S1]", validated)
        self.assertIn("候補者への返信は古いものから", validated)
        self.assertNotIn("自由記述", validated)

    def test_uncited_intro_is_silently_removed_when_cited_answer_exists(self) -> None:
        answer = "会議ログによると次の通りです。\n古いものから返信します。[S1]"

        validated = validate_grounded_answer(answer, self.contexts, structured=False)

        self.assertNotIn("確認できません", validated)
        self.assertNotIn("会議ログによると", validated)
        self.assertIn("古いものから返信します。[S1]", validated)

    def test_unsupported_number_and_person_are_removed(self) -> None:
        contexts = [
            DocumentChunk(
                Path("meeting.md"),
                2,
                "候補者3名を山田さんが担当する。",
                DocumentMetadata("会議", None, (), None),
            )
        ]
        answer = (
            "候補者は8名です。[S1]\n"
            "担当は佐藤さんです。[S1]\n"
            "候補者は3名で、山田さんが担当します。[S1]"
        )

        validated = validate_grounded_answer(answer, contexts, structured=False)

        self.assertNotIn("8名", validated)
        self.assertNotIn("佐藤さん", validated)
        self.assertIn("3名", validated)
        self.assertIn("山田さん", validated)

    def test_calculated_number_from_evidence_is_allowed(self) -> None:
        contexts = [
            DocumentChunk(
                Path("meeting.md"),
                2,
                "月額49万円を15か月分支給する。",
                DocumentMetadata("会議", None, (), None),
            )
        ]

        validated = validate_grounded_answer(
            "初年度は735万円です。[S1]", contexts, structured=False
        )

        self.assertIn("735万円", validated)


class GeminiFallbackTests(unittest.TestCase):
    def test_deadline_disables_retries_and_thinking(self) -> None:
        response = {"candidates": [{"content": {"parts": [{"text": "回答"}]}}]}
        with (
            patch("hermes_agent.get_gemini_api_key", return_value="test-key"),
            patch("hermes_agent.time.monotonic", return_value=100),
            patch("hermes_agent.post_gemini_json", return_value=response) as post_json,
        ):
            answer_with_gemini("system", "question", "gemini-2.5-flash-lite", deadline=117)
        self.assertEqual(1, post_json.call_args.kwargs["retries"])
        self.assertLessEqual(post_json.call_args.kwargs["timeout"], 17)
        config = post_json.call_args.args[1]["generationConfig"]
        self.assertEqual(0, config["thinkingConfig"]["thinkingBudget"])

    def test_service_unavailable_uses_flash_lite_fallback(self) -> None:
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "フォールバック回答"}]}}
            ]
        }
        with (
            patch("hermes_agent.get_gemini_api_key", return_value="test-key"),
            patch(
                "hermes_agent.post_gemini_json",
                side_effect=[RuntimeError("Gemini API error 503: unavailable"), response],
            ) as post_json,
        ):
            answer = answer_with_gemini("system", "question", "gemini-2.5-flash")

        self.assertEqual("フォールバック回答", answer)
        self.assertIn("gemini-2.5-flash-lite", post_json.call_args_list[1].args[0])


class AdviceTests(unittest.TestCase):
    def test_default_is_concise_and_detail_is_enabled_only_when_requested(self) -> None:
        from hermes_agent import wants_detailed_answer
        contexts = [DocumentChunk(Path("meeting.txt"), 1, "選考基準を共有する。", DocumentMetadata("定例", None, (), None))]
        brief = build_user_prompt("エンジニアの選考フローどうしたら", contexts, structured=False)
        detailed = build_user_prompt("具体的な手順を詳しく教えて", contexts, structured=False, advice_mode=True)
        self.assertIn("回答の詳しさ: 簡潔", brief)
        self.assertIn("150〜300字", brief)
        self.assertIn("回答の詳しさ: 詳細", detailed)
        self.assertTrue(wants_detailed_answer("具体例を教えて"))
        self.assertFalse(wants_detailed_answer("具体的だけど短く教えて"))

    def test_advice_leads_with_current_action_then_past_decision_reasoning(self) -> None:
        import json
        evidence = "必須条件と入社後に育成できる条件を分けて検討している。"
        contexts = [DocumentChunk(Path("meeting.txt"), 1, evidence, DocumentMetadata("定例", None, (), None))]
        prompt = build_user_prompt("エンジニアの選考フローどうしたら", contexts, structured=False)
        result = {
            "proposals": [{
                "text": "まず必須条件と育成できる条件を分けて比べてはどうでしょう。",
                "background": "以前の会議では条件を分けて検討していました。今回も同じ軸で比べるとよさそうです。",
                "evidence": "E1_1", "sources": ["S1"],
            }], "supplement": "",
        }
        response = {"candidates": [{"content": {"parts": [{"text": json.dumps(result, ensure_ascii=False)}]}}]}
        with (
            patch("hermes_agent.get_gemini_api_key", return_value="test-key"),
            patch("hermes_agent.post_gemini_json", return_value=response),
        ):
            answer = answer_with_gemini("system", prompt, "gemini-2.5-flash-lite")
        self.assertTrue(answer.startswith("提案: まず必須条件"))
        self.assertLess(answer.index("提案:"), answer.index("参考にした考え方:"))
        self.assertIn("原文が検討や相談の段階", prompt)
        self.assertNotIn("会議での背景:", answer)
        self.assertNotIn("E1_1", answer)
        result["proposals"][0]["evidence"] = "E99_1"
        response["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(result, ensure_ascii=False)
        with (
            patch("hermes_agent.get_gemini_api_key", return_value="test-key"),
            patch("hermes_agent.post_gemini_json", return_value=response),
        ):
            self.assertEqual("", answer_with_gemini("system", prompt, "gemini-2.5-flash-lite"))

    def test_evidence_must_match_transcript_despite_speaker_boundaries(self) -> None:
        from hermes_agent import evidence_quote_matches
        source = "選考担当者が単独で不合格を決めるのは難しいと話しています。\n担当者 06:00\n判断基準を共有したいという相談がありました。"
        self.assertTrue(evidence_quote_matches("選考担当者が単独で不合格を決めるのは難しいと話しています。判断基準を共有したいという相談がありました。", source))
        self.assertFalse(evidence_quote_matches("全く関係のない海外出張の日程が会議で正式に決定された。", source))

    def test_general_knowledge_is_separate_and_requires_grounded_core(self) -> None:
        contexts = [DocumentChunk(Path("engineering.txt"), 1, "エンジニアの書類選考は判断基準の共有が課題。", DocumentMetadata("定例", None, (), None))]
        supplement = "一般的な補足（会議ログ未確認）: 評価表には必須スキルと入社後に学べるスキルを分けて記入する方法があります。"
        validated = validate_grounded_answer(
            f"会議での背景: 判断基準の共有が課題です。[S1]\n提案: 評価項目を共有して判断の違いを確認してください。[S1]\n{supplement}",
            contexts, structured=False, allow_advice=True,
        )
        self.assertIn(supplement, validated)
        self.assertNotIn(supplement, validate_grounded_answer(supplement, contexts, structured=False, allow_advice=True))
        self.assertNotIn(supplement, validate_grounded_answer(supplement, contexts, structured=False))

    def test_general_knowledge_cannot_claim_named_assignee_or_use_citations(self) -> None:
        contexts = [DocumentChunk(Path("engineering.txt"), 1, "評価基準を共有する。", DocumentMetadata("定例", None, (), None))]
        answer = (
            "提案: 評価基準を共有してください。[S1]\n"
            "一般的な補足（会議ログ未確認）: 山田さんが担当します。\n"
            "一般的な補足（会議ログ未確認）: 採用率が上がると決まっています。[S1]"
        )
        validated = validate_grounded_answer(answer, contexts, structured=False, allow_advice=True)
        self.assertNotIn("山田さん", validated)
        self.assertNotIn("一般的な補足", validated)

    def test_follow_up_continues_consultation_with_history(self) -> None:
        from hermes_agent import answer_question
        contexts = [DocumentChunk(Path("engineering.txt"), 1, "エンジニアの書類選考に待ち時間がある。", DocumentMetadata("定例", None, (), None))]
        with patch("hermes_agent.answer_with_gemini", return_value="提案: 書類選考の待ち時間短縮を検討してください。[S1]") as generate:
            answer = answer_question(
                "書類選考です", Path("docs"), "gemini-2.5-flash-lite", "gemini",
                chunks=contexts, system_prompt="system", structured=False,
                conversation_history=[{"role": "user", "content": "エンジニアの選考フローどうしたら"}],
            )
        self.assertIn("回答モード: 相談", generate.call_args.args[1])
        self.assertIn("提案:", answer)
        self.assertEqual(1, answer.count("確認:"))

    def test_user_examples_are_consultations_but_fact_question_is_not(self) -> None:
        self.assertTrue(is_advice_question("経理採用を上手く行かせるには"))
        self.assertTrue(is_advice_question("エンジニアの選考フローどうしたら"))
        self.assertFalse(is_advice_question("経理採用で決まった担当者は誰？"))

    def test_role_filter_excludes_unrelated_evidence(self) -> None:
        metadata = DocumentMetadata("定例", None, (), None)
        accounting = DocumentChunk(Path("accounting.txt"), 1, "経理の採用要件を見直す。", metadata)
        engineering = DocumentChunk(Path("engineering.txt"), 1, "エンジニアの技術面接を検討する。", metadata)
        self.assertEqual([accounting], advice_search_scope("経理採用をうまく進めたい", [accounting, engineering]))
        self.assertEqual([], advice_search_scope("経理採用をうまく進めたい", [engineering]))
        unrelated_finance = DocumentChunk(Path("sales.txt"), 1, "財務知識を使う営業職の要件。", metadata)
        self.assertEqual([accounting], advice_search_scope("経理採用をうまく進めたい", [accounting, unrelated_finance]))

    def test_proposals_require_citations_and_cannot_invent_people(self) -> None:
        contexts = [DocumentChunk(Path("meeting.txt"), 1, "エンジニアの選考は待ち時間が課題。", DocumentMetadata("定例", None, (), None))]
        prompt = build_user_prompt("エンジニアの選考フローどうしたら", contexts, structured=False)
        self.assertIn("回答モード: 相談", prompt)
        answer = (
            "提案: 選考の待ち時間を減らす運用を検討してください。[S1]\n"
            "提案: 山田さんが担当します。[S1]\n"
            "提案: 必ず採用に成功します。\n"
            "確認: どの段階の待ち時間が長いですか？"
        )
        validated = validate_grounded_answer(answer, contexts, structured=False, allow_advice=True)
        self.assertIn("待ち時間を減らす", validated)
        self.assertIn("確認: どの段階", validated)
        self.assertNotIn("山田さん", validated)
        self.assertNotIn("必ず採用", validated)


if __name__ == "__main__":
    unittest.main()
