"""Invalid selections and engine failures stop or clean up the command."""

import unittest
import shutil

from operator_fixtures import OperatorFixture, ROOT


class MakefileTests(OperatorFixture, unittest.TestCase):
    def test_ambient_fp_does_not_select_precision(self):
        result = self.make("weights/sanitize", "DOCKER_CONTEXT=science")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Supply FP=", result.stderr)
        self.assertFalse(self.calls())

    def test_invalid_empty_and_make_syntax_inputs_fail_before_engine(self):
        marker = self.directory / "injected"
        for value in ("", "float17", "keep fp16", f"$(shell touch {marker})"):
            with self.subTest(value=value):
                result = self.make("weights/sanitize", "DOCKER_CONTEXT=science", "FP=" + value)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls())
                self.assertFalse(marker.exists())

    def test_remote_context_is_rejected_before_running_container(self):
        self.environment["TEST_ENDPOINT"] = "ssh://remote"
        result = self.make("weights/verify", "DOCKER_CONTEXT=remote")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local Unix socket", result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_verification_needs_no_fp_and_propagates_engine_failure(self):
        self.environment["TEST_DOCKER_STATUS"] = "7"
        result = self.make("weights/verify", "DOCKER_CONTEXT=science")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error 7", result.stderr)
        self.assertEqual(self.calls()[-1]["args"][-1], "verify")

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
        inputs.write_text(inputs.read_text() + "\n# changed dependency intent\n")
        pool = self.directory / "pool"
        pool.mkdir()
        result = self.make("build", "PROFILE=weights-sanitize", "DOCKER_CONTEXT=science",
                           f"WHEELHOUSE={pool}", makefile=repository / "Makefile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("regenerate with make lock", result.stderr)
        self.assertFalse(any(call["args"][2:3] == ["run"] for call in self.calls()))
