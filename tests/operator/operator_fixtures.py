"""Temporary fake-engine and terminal fixtures for public Make command tests."""

import hashlib
import json
import os
import pathlib
import pty
import re
import select
import signal
import shutil
import subprocess
import tempfile
import time

# Source location anchors repository paths independently of operator settings.
# repo-check: allow-module-constant
ROOT = pathlib.Path(__file__).resolve().parents[2]


class OperatorFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

        self.addCleanup(self.temporary.cleanup)
        self.directory = pathlib.Path(self.temporary.name)
        self.log = self.directory / "docker.jsonl"

        docker = self.directory / "docker"

        docker.write_text(
            """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
recorded_env = {
    key: os.environ[key]
    for key in ("DOCKER_HOST", "COMPOSE_FILE", "REPRO_FP", "PROXY_URL")
    if key in os.environ
}
with open(os.environ["TEST_DOCKER_LOG"], "a") as f:
    f.write(json.dumps({"args": args, "env": recorded_env}) + "\\n")
if args[:2] == ["context", "inspect"]:
    print(os.environ.get("TEST_ENDPOINT", "unix:///tmp/science.sock"))
elif args[2:3] == ["info"]:
    print(json.dumps(["name=rootless"]))
elif args[2:4] == ["buildx", "bake"] and "--print" in args:
    target = json.loads(os.environ["TEST_BAKE_TARGET"])
    print(json.dumps({"target": {args[-1]: target}}))
elif args[2:3] == ["image"] or args[2:3] == ["run"]:
    pass
else:
    sys.exit(int(os.environ.get("TEST_DOCKER_STATUS", "0")))
"""
        )

        docker.chmod(0o755)
        definition = (ROOT / "docker-bake.hcl").read_text()

        target = {
            "args": {
                name: re.search(name + r' = "([^"]+)"', definition)[1]
                for name in ("PYTHON_BASE", "PY_VERSION")
            },
            "platforms": ["linux/amd64"],
            "tags": ["moldetr/weights-sanitizer:local"],
        }

        self.environment = {
            **os.environ,
            "PATH": str(self.directory) + os.pathsep + os.environ["PATH"],
            "TEST_BAKE_TARGET": json.dumps(target),
            "TEST_DOCKER_LOG": str(self.log),
            "FP": "fp16",
            "REPRO_FP": "fp16",
            "TEST_ENDPOINT": "unix:///tmp/science.sock",
            "TEST_DOCKER_STATUS": "0",
            "DOCKER_HOST": "tcp://unwanted",
            "COMPOSE_FILE": "unwanted.yaml",
            "REPRO_EXPLICIT_INPUTS": "FP",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def make(self, *arguments, cwd=ROOT, makefile=None):
        return subprocess.run(
            ["make", "--no-print-directory", "-f", str(makefile or (ROOT / "Makefile")), *arguments],
            cwd=cwd,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            start_new_session=True,
            timeout=15,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def sdk_release(self):
        repository = self.directory / "checkout"
        pool = self.directory / "wheel pool"

        (repository / "builds").mkdir(parents=True)
        (repository / "make").mkdir()
        pool.mkdir()
        shutil.copyfile(ROOT / "Makefile", repository / "Makefile")
        shutil.copyfile(ROOT / "make/build.py", repository / "make/build.py")

        artifacts = {
            "wheelhouse-tool-9.8.7.pyz": b"tool fixture",
            "provider_core-9.8.7-native-install.lock": b"native lock fixture",
            "provider_core-9.8.7-install.lock": b"full lock fixture",
            "provider_core-9.8.7-py3-none-any.whl": b"SDK wheel fixture",
        }

        hashes = {artifact_name: hashlib.sha256(payload).hexdigest() for (artifact_name, payload) in artifacts.items()}
        manifest = json.dumps(hashes).encode()

        (repository / "builds/sdk.constraints.txt").write_text(
            f"provider-core==9.8.7\n"
            f"# release-manifest-sha256 = {hashlib.sha256(manifest).hexdigest()}\n"  # machine-string: SDK release pin fixture.
        )

        artifacts["provider_core-9.8.7-sha256.json"] = manifest

        return (repository, pool, artifacts)

    def check_build_cleanup(self, status):
        pool = self.directory / "wheel pool"

        pool.mkdir()
        self.environment["TEST_DOCKER_STATUS"] = status

        command_result = self.make(
            "build",
            "PROFILE=weights-sanitize",  # machine-string: Literal Make variable assignment.
            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
            f"WHEELHOUSE={pool}",  # machine-string: Literal Make variable assignment.
        )

        self.assertEqual(first=status == "0", second=command_result.returncode == 0, msg=command_result.stderr)
        args = self.calls()[-1]["args"]

        self.assertEqual(first=["buildx", "bake"], second=args[2:4])
        self.assertIn("weights-sanitize", args)
        stage = pathlib.Path(args[args.index("--set") + 1].split("=", 1)[1])

        self.assertFalse(stage.exists())
        preparation = next(call["args"] for call in self.calls() if call["args"][2:3] == ["run"])

        self.assertEqual(first="none", second=preparation[preparation.index("--network") + 1])
        self.assertIn("--require-hashes", preparation)
        self.assertIn("--no-index", preparation)

    def check_terminal_answers(self, answers):
        for answer in answers:
            with self.subTest(answer=answer):
                self.log.unlink(missing_ok=True)
                (pid, fd) = pty.fork()

                if pid == 0:
                    os.chdir(ROOT)

                    os.execvpe(
                        "make",
                        [
                            "make",
                            "--no-print-directory",
                            "weights/sanitize",
                            "DOCKER_CONTEXT=science",  # machine-string: Literal Make variable assignment.
                        ],
                        self.environment,
                    )

                output = b""
                sent = False

                try:
                    deadline = time.monotonic() + 10

                    while time.monotonic() < deadline:
                        (ready, _, _) = select.select([fd], [], [], 0.1)

                        if ready:
                            try:
                                chunk = os.read(fd, 65536)
                            except OSError:
                                break

                            if not chunk:
                                break

                            output += chunk

                            if (b"Select FP" in output) and (not sent):
                                self.assertIn(b"1) keep", output)
                                self.assertIn(b"5) fp64", output)
                                os.write(fd, answer)
                                sent = True
                    else:
                        self.fail("Interactive selection timed out")

                    (_, status) = os.waitpid(pid, 0)
                    pid = None

                    self.assertTrue(sent, msg=output.decode())

                    if answer == b"2\n":
                        self.assertEqual(first=0, second=os.waitstatus_to_exitcode(status), msg=output.decode())
                        self.assertEqual(first=["--fp", "fp16"], second=self.calls()[-1]["args"][-2:])
                    else:
                        self.assertNotEqual(os.waitstatus_to_exitcode(status), 0)
                        self.assertFalse(self.calls())
                finally:
                    os.close(fd)

                    if pid is not None:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
