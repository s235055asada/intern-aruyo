from __future__ import annotations

import threading
import time
import unittest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from hermes_agent import DocumentChunk, DocumentMetadata, retrieve_parallel, retrieve_gbrain


class ParallelSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        metadata = DocumentMetadata("meeting", None, (), None)
        self.keyword = DocumentChunk(Path("keyword.md"), 1, "keyword evidence", metadata, score=10)
        self.semantic = DocumentChunk(Path("semantic.md"), 1, "semantic evidence", metadata, score=0.9)
        self.chunks = [self.keyword, self.semantic]
        self.slot = threading.BoundedSemaphore(1)
        self.slot_patch = patch("hermes_agent._PARALLEL_GBRAIN_SLOT", self.slot)
        self.slot_patch.start()

    def tearDown(self) -> None:
        self.slot_patch.stop()

    def test_semantic_search_starts_before_keyword_search_finishes(self) -> None:
        semantic_started = threading.Event()
        keyword_started = threading.Event()

        def semantic(*args, **kwargs):
            semantic_started.set()
            self.assertTrue(keyword_started.wait(1))
            return [self.semantic]

        def keyword(*args, **kwargs):
            keyword_started.set()
            self.assertTrue(semantic_started.wait(1))
            return [self.keyword]

        with (
            patch("hermes_agent.retrieve_keyword", side_effect=keyword),
            patch("hermes_agent.retrieve_gbrain", side_effect=semantic),
            self.assertLogs("hermes.search", level="INFO") as logs,
        ):
            result = retrieve_parallel("question", self.chunks, limit=2, timeout_seconds=1)
        self.assertEqual({Path("keyword.md"), Path("semantic.md")}, {item.file_path for item in result})
        self.assertIn("gbrain=used", " ".join(logs.output))

    def test_slow_semantic_search_does_not_hold_up_keyword_answer(self) -> None:
        release = threading.Event()

        def slow(*args, **kwargs):
            release.wait(1)
            return [self.semantic]

        try:
            with (
                patch("hermes_agent.retrieve_keyword", return_value=[self.keyword]),
                patch("hermes_agent.retrieve_gbrain", side_effect=slow),
            ):
                started = time.monotonic()
                result = retrieve_parallel("question", self.chunks, timeout_seconds=0.04)
                self.assertLess(time.monotonic() - started, 0.3)
                self.assertEqual([self.keyword], result)
        finally:
            release.set()
            self.assertTrue(self.slot.acquire(timeout=1))
            self.slot.release()

    def test_missing_or_failed_gbrain_returns_original_keyword_results(self) -> None:
        for error in (FileNotFoundError("missing"), RuntimeError("failed")):
            with (
                patch("hermes_agent.retrieve_keyword", return_value=[self.keyword]),
                patch("hermes_agent.retrieve_gbrain", side_effect=error),
            ):
                self.assertEqual([self.keyword], retrieve_parallel("question", self.chunks))

    def test_busy_gbrain_does_not_spawn_another_search(self) -> None:
        self.slot.acquire()
        try:
            with (
                patch("hermes_agent.retrieve_keyword", return_value=[self.keyword]),
                patch("hermes_agent.retrieve_gbrain") as semantic,
            ):
                self.assertEqual([self.keyword], retrieve_parallel("question", self.chunks))
                semantic.assert_not_called()
        finally:
            self.slot.release()

    def test_zero_budget_disables_semantic_api_call(self) -> None:
        with (
            patch("hermes_agent.retrieve_keyword", return_value=[self.keyword]),
            patch("hermes_agent.retrieve_gbrain") as semantic,
        ):
            self.assertEqual([self.keyword], retrieve_parallel("question", self.chunks, timeout_seconds=0))
            semantic.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows process-tree cleanup")
    def test_gbrain_timeout_terminates_only_its_own_process_tree(self) -> None:
        with (
            patch("hermes_agent.gbrain_query_cache_key", return_value="key"),
            patch("hermes_agent.load_gbrain_query_cache", return_value={}),
            patch("hermes_agent.get_gemini_api_key", return_value="test-key"),
            patch("hermes_agent.find_gbrain_executable", return_value="gbrain.exe"),
            patch("hermes_agent.subprocess.Popen") as popen,
            patch("hermes_agent.subprocess.run") as cleanup,
        ):
            process = popen.return_value
            process.pid = 12345
            process.poll.return_value = 1
            process.communicate.side_effect = [subprocess.TimeoutExpired("gbrain", 0.1), ("", "")]
            with self.assertRaises(RuntimeError):
                retrieve_gbrain("question", self.chunks, timeout_seconds=0.1)
            self.assertEqual(["taskkill", "/PID", "12345", "/T", "/F"], cleanup.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
