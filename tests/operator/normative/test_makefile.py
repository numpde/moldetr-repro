"""Supported operator commands, literal inputs, and terminal cancellation."""

import unittest

from operator_fixtures import OperatorFixture, ROOT


class MakefileTests(OperatorFixture, unittest.TestCase):
    def test_help_and_dry_run_need_no_selection_or_engine(self):
        for args in [(), ("help",), ("-n", "weights/sanitize"), ("-qp",)]:
            with self.subTest(args=args):
                command_result = self.make(*args)

                self.assertIn(command_result.returncode, (0, 1))
                self.assertFalse(self.calls())

    def test_each_explicit_precision_reaches_container_from_another_directory(self):
        choices = ("keep", "fp16", "bf16", "fp32", "fp64")

        for precision in choices:
            command_result = self.make(
                "weights/sanitize",
                "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
                "FP=" + precision,
                cwd=self.directory,
            )

            self.assertEqual(first=0, second=command_result.returncode, msg=command_result.stderr)
            call = self.calls()[-1]

            self.assertEqual(first=["--name", "moldetr-weights-writer", "sanitize", "--fp", precision], second=call["args"][-5:])
            self.assertEqual(first=["--host", "unix:///tmp/science.sock"], second=call["args"][:2])
            self.assertIn(str(ROOT / "compose/weights.yaml"), call["args"])
            self.assertNotIn("DOCKER_HOST", call["env"])
            self.assertNotIn("COMPOSE_FILE", call["env"])
            self.assertNotIn("REPRO_FP", call["env"])

    def test_download_sequences_services_and_passes_literal_proxy(self):
        proxy = "http://proxy.example/path with spaces"

        command_result = self.make(
            "weights/download",
            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
            "PROXY_URL=" + proxy,
        )

        self.assertEqual(first=0, second=command_result.returncode, msg=command_result.stderr)
        runs = self.calls()[1:]

        self.assertEqual(first=["volume-init", "download"], second=[call["args"][-1] for call in runs])
        self.assertTrue(all((call["env"]["PROXY_URL"] == proxy) for call in runs))
        self.assertEqual(first=["--name", "moldetr-weights-writer", "download"], second=runs[1]["args"][-3:])

    def test_build_stages_locked_wheels_and_cleans_stage(self):
        self.check_build_cleanup("0")

    def test_terminal_selection_and_cancellation(self):
        self.check_terminal_answers((b"2\n", b"\n"))
