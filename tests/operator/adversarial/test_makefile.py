"""Invalid selections and engine failures stop or clean up the command."""

import unittest
import shutil

from operator_fixtures import OperatorFixture, ROOT


class MakefileTests(OperatorFixture, unittest.TestCase):
    def test_ambient_fp_does_not_select_precision(self):
        command_result = self.make(
            "weights/sanitize",
            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
        )

        self.assertNotEqual(command_result.returncode, 0)
        self.assertIn("Supply FP=", command_result.stderr)
        self.assertFalse(self.calls())

    def test_invalid_empty_and_make_syntax_inputs_fail_before_engine(self):
        marker = self.directory / "injected"

        for precision_input in (
            "",
            "float17",
            "keep fp16",
            f"$(shell touch {marker})",  # machine-string: Malicious Make expression must reach the literal-input guard unchanged.
        ):
            with self.subTest(precision_input=precision_input):
                command_result = self.make(
                    "weights/sanitize",
                    "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
                    "FP=" + precision_input,
                )

                self.assertNotEqual(command_result.returncode, 0)
                self.assertFalse(self.calls())
                self.assertFalse(marker.exists())

    def test_remote_context_is_rejected_before_running_container(self):
        self.environment["TEST_ENDPOINT"] = "ssh://remote"

        command_result = self.make(
            "weights/verify",
            "DOCKER_CONTEXT=remote",  # machine-string: Literal Make variable assignment.
        )

        self.assertNotEqual(command_result.returncode, 0)
        self.assertIn("local Unix socket", command_result.stderr)
        self.assertEqual(first=1, second=len(self.calls()))

    def test_verification_needs_no_fp_and_propagates_engine_failure(self):
        self.environment["TEST_DOCKER_STATUS"] = "7"

        command_result = self.make(
            "weights/verify",
            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
        )

        self.assertNotEqual(command_result.returncode, 0)
        self.assertIn("Error 7", command_result.stderr)
        self.assertEqual(first="verify", second=self.calls()[-1]["args"][-1])

    def test_failed_build_cleans_stage(self):
        self.check_build_cleanup("7")

    def test_out_of_range_terminal_selection(self):
        self.check_terminal_answers((b"99\n",))

    def test_changed_dependency_input_refuses_before_package_staging(self):
        repository = self.directory / "checkout"

        repository.mkdir()

        for name in ("Makefile", "docker-bake.hcl"):
            shutil.copyfile(ROOT / name, repository / name)

        for name in ("make", "builds"):
            shutil.copytree(ROOT / name, repository / name)

        inputs = repository / "builds/weights-sanitize/requirements.in"

        inputs.write_text(inputs.read_text() + "\n" "# changed dependency intent\n")
        pool = self.directory / "pool"

        pool.mkdir()

        command_result = self.make(
            "build",
            "PROFILE=weights-sanitize",  # machine-string: Literal Make variable assignment.
            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
            f"WHEELHOUSE={pool}",  # machine-string: Literal Make variable assignment.
            makefile=repository / "Makefile"
        )

        self.assertNotEqual(command_result.returncode, 0)
        self.assertIn("regenerate with make lock", command_result.stderr)
        self.assertFalse(any((call["args"][2:3] == ["run"]) for call in self.calls()))
