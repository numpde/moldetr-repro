"""Generate locks and build images from cached packages without network access."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


class Build:
    def __init__(self, host, pool):
        self.pool = Path(pool).resolve(strict=True)
        if not self.pool.is_dir():
            raise ValueError(f"WHEELHOUSE must name an existing directory: {self.pool}")
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(("DOCKER_", "COMPOSE_", "BUILDX_", "BUILDKIT_"))}
        self.docker = ["docker", "--host", host]
        options = json.loads(self.run(self.docker + ["info", "--format", "{{json .SecurityOptions}}"],
                                      capture_output=True, text=True).stdout)
        if "name=rootless" not in options:
            raise ValueError("Offline package staging requires a rootless engine so generated files retain checkout ownership")

    def run(self, arguments, **kwargs):
        return subprocess.run(arguments, cwd=ROOT, env=self.environment, check=True, **kwargs)

    def target(self, profile):
        result = self.run(self.docker + ["buildx", "bake", "--builder", "default", "-f", "docker-bake.hcl", "--print", profile],
                          capture_output=True, text=True)
        return json.loads(result.stdout)["target"][profile]

    def input_hash(self, profile, target):
        inputs = {"requirements.in": (ROOT / "builds" / profile / "requirements.in").read_text(),
                  "args": target["args"], "platforms": target["platforms"],
                  "constraints": {path.name: path.read_text() for path in sorted((ROOT / "builds" / profile).glob("*.constraints.txt"))}}
        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    def container(self, image, profile, stage):
        return self.docker + [
            "run", "--rm", "--pull=never", "--network", "none", "--read-only", "--user", "0:0",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--cpus", "1", "--memory", "1g",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--mount", f"type=bind,src={self.pool},dst=/pool,readonly",
            "--mount", f"type=bind,src={ROOT / 'builds' / profile},dst=/input,readonly",
            "--mount", f"type=bind,src={stage},dst=/stage",
            "-e", "UV_CACHE_DIR=/stage/cache",
            "--entrypoint", "python", image,
        ]

    def lock(self, profile):
        target = self.target(profile)
        tooling = self.target("lock")
        for image in (target["args"]["PYTHON_BASE"], tooling["contexts"]["uv"].removeprefix("docker-image://")):
            self.run(self.docker + ["image", "inspect", image], stdout=subprocess.DEVNULL)
        self.run(self.docker + ["buildx", "bake", "--builder", "default", "-f", "docker-bake.hcl", "lock", "--load"])
        with tempfile.TemporaryDirectory(prefix="moldetr-lock-") as temporary:
            stage = Path(temporary)
            command = self.container(tooling["tags"][0], profile, stage)
            command[command.index("--entrypoint") + 1] = "uv"
            constraints = [value for path in sorted((ROOT / "builds" / profile).glob("*.constraints.txt"))
                           for value in ("--constraint", "/input/" + path.name)]
            self.run(command + ["pip", "compile", "/input/requirements.in", *constraints,
                                "--python-version", target["args"]["PY_VERSION"],
                                "--offline", "--no-index", "--find-links", "/pool",
                                "--generate-hashes", "--no-header", "--no-annotate",
                                "--output-file", "/stage/requirements.lock"])
            destination = ROOT / "builds" / profile / "requirements.lock"
            # Publish the complete generated lock in one rename.
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".requirements-", delete=False) as stream:
                pending = Path(stream.name)
                try:
                    stream.write(f"# inputs-sha256: {self.input_hash(profile, target)}\n".encode())
                    stream.write((stage / "requirements.lock").read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                    pending.chmod(0o644)
                    pending.replace(destination)
                finally:
                    pending.unlink(missing_ok=True)

    def build(self, profile):
        target = self.target(profile)
        lock = ROOT / "builds" / profile / "requirements.lock"
        if lock.read_text().splitlines()[0] != f"# inputs-sha256: {self.input_hash(profile, target)}":
            raise ValueError(f"Cannot build {profile}: dependency inputs changed; regenerate with make lock")
        base = target["args"]["PYTHON_BASE"]
        self.run(self.docker + ["image", "inspect", base], stdout=subprocess.DEVNULL)
        with tempfile.TemporaryDirectory(prefix="moldetr-wheels-") as temporary:
            stage = Path(temporary)
            self.run(self.container(base, profile, stage) + [
                "-m", "pip", "download", "--disable-pip-version-check", "--no-cache-dir",
                "--no-index", "--find-links", "/pool", "--only-binary=:all:",
                "--no-deps", "--require-hashes", "--dest", "/stage", "-r", "/input/requirements.lock",
            ])
            (stage / "requirements.lock").write_bytes((ROOT / "builds" / profile / "requirements.lock").read_bytes())
            self.run(self.docker + ["buildx", "bake", "--builder", "default", "-f", "docker-bake.hcl", profile,
                                   "--set", f"{profile}.contexts.wheels={stage}",
                                   "--allow", f"fs.read={stage}", "--load"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("lock", "build"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--profile", required=True,
                        choices=sorted(path.name for path in (ROOT / "builds").iterdir() if path.is_dir()))
    args = parser.parse_args()
    getattr(Build(args.host, args.pool), args.operation)(args.profile)


if __name__ == "__main__":
    main()
