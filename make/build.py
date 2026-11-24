"""Acquire pinned SDK releases and build images from cached packages offline."""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import urllib.request


# Source location anchors repository paths independently of operator settings.
# repo-check: allow-module-constant
ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(artifact_path):
    with artifact_path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Build:
    def __init__(self, host, pool):
        self.pool = pathlib.Path(pool).resolve(strict=True)

        if not self.pool.is_dir():
            raise ValueError(f"WHEELHOUSE must name an existing directory: {self.pool=}")

        self.environment = {
            key: environment_value
            for (key, environment_value) in os.environ.items()
            if not key.startswith(("DOCKER_", "COMPOSE_", "BUILDX_", "BUILDKIT_"))
        }

        self.docker = ["docker", "--host", host]

    def sdk_acquire(self):
        constraints = (ROOT / "builds/sdk.constraints.txt").read_text()
        sdk_version = re.search(r"^provider-core==([^\s]+)$", constraints, re.M)[1]
        manifest_sha256 = re.search(r"^# release-manifest-sha256 = ([0-9a-f]{64})$", constraints, re.M)[1]
        release_url = f"https://github.com/numpde/nmr-provider-tools/releases/download/v{sdk_version}/"  # machine-string: Published SDK release URL.
        manifest_name = f"provider_core-{sdk_version}-sha256.json"  # machine-string: Published SDK checksum manifest filename.

        self.fetch(release_url, manifest_name, manifest_sha256)
        hashes = json.loads((self.pool / manifest_name).read_text())

        artifact_names = (
            f"wheelhouse-tool-{sdk_version}.pyz",  # machine-string: Published wheelhouse tool filename.
            f"provider_core-{sdk_version}-native-install.lock",  # machine-string: Published scientific worker install lock filename.
            f"provider_core-{sdk_version}-install.lock",  # machine-string: Published full SDK install lock filename.
            f"provider_core-{sdk_version}-py3-none-any.whl",  # machine-string: Published SDK wheel filename.
        )

        for artifact_name in artifact_names:
            self.fetch(release_url, artifact_name, hashes[artifact_name])

    def fetch(self, release_url, artifact_name, expected_sha256):
        destination = self.pool / artifact_name

        if destination.exists():
            actual_sha256 = digest(destination)

            if not (actual_sha256 == expected_sha256):
                raise ValueError(f"Cannot use cached {destination=}: {actual_sha256=} differs from the pinned {expected_sha256=}")

            return

        with tempfile.NamedTemporaryFile(dir=self.pool, prefix=".sdk-", delete=False) as output:
            pending = pathlib.Path(output.name)

            try:
                with urllib.request.urlopen(release_url + artifact_name, timeout=120) as response:
                    total_bytes = 0

                    while block := response.read(65536):
                        total_bytes += len(block)

                        if not (total_bytes <= (8 * 1024 * 1024)):
                            raise ValueError(f"Cannot acquire {artifact_name=}: {total_bytes=} exceeds the 8 MiB SDK support artifact limit")

                        output.write(block)

                output.flush()
                os.fsync(output.fileno())
                actual_sha256 = digest(pending)

                if not (actual_sha256 == expected_sha256):
                    raise ValueError(f"Cannot acquire {artifact_name=}: downloaded {actual_sha256=} differs from the pinned {expected_sha256=}")

                pending.chmod(0o644)
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)

    def invoke(self, arguments, **kwargs):
        return subprocess.run(arguments, cwd=ROOT, env=self.environment, check=True, **kwargs)

    def target(self, profile):
        bake_result = self.invoke(
            self.docker + ["buildx", "bake", "--builder", "default", "-f", "docker-bake.hcl", "--print", profile],
            capture_output=True,
            text=True
        )

        return json.loads(bake_result.stdout)["target"][profile]

    def input_hash(self, profile, target):
        inputs = {
            "requirements.in": (ROOT / "builds" / profile / "requirements.in").read_text(),
            "args": target["args"],
            "platforms": target["platforms"],
            "constraints": {build_path.name: build_path.read_text() for build_path in sorted((ROOT / "builds" / profile).glob("*.constraints.txt"))}
        }

        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    def container(self, image, profile, stage):
        options = json.loads(
            self.invoke(
                self.docker + ["info", "--format", "{{json .SecurityOptions}}"],
                capture_output=True,
                text=True,
            ).stdout
        )

        rootless_option = "name=rootless"  # machine-string: Docker security option token.

        if not (rootless_option in options):
            raise ValueError(f"Offline package staging requires a rootless engine; observed {options=}")

        return self.docker + [
            "run",
            "--rm",
            "--pull=never",
            "--network",
            "none",
            "--read-only",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--mount",
            f"type=bind,src={self.pool},dst=/pool,readonly",  # machine-string: Docker bind mount specification.
            "--mount",
            f"type=bind,src={ROOT / 'builds' / profile},dst=/input,readonly",  # machine-string: Docker bind mount specification.
            "--mount",
            f"type=bind,src={stage},dst=/stage",  # machine-string: Docker bind mount specification.
            "-e",
            "UV_CACHE_DIR=/stage/cache",
            "--entrypoint",
            "python",
            image,
        ]

    def lock(self, profile):
        target = self.target(profile)
        tooling = self.target("lock")

        for image in (target["args"]["PYTHON_BASE"], tooling["contexts"]["uv"].removeprefix("docker-image://")):
            self.invoke(self.docker + ["image", "inspect", image], stdout=subprocess.DEVNULL)

        self.invoke(self.docker + ["buildx", "bake", "--builder", "default", "-f", "docker-bake.hcl", "lock", "--load"])

        with tempfile.TemporaryDirectory(prefix="moldetr-lock-") as temporary:
            stage = pathlib.Path(temporary)
            command = self.container(tooling["tags"][0], profile, stage)
            command[command.index("--entrypoint") + 1] = "uv"

            constraints = [
                argument_value
                for build_path in sorted((ROOT / "builds" / profile).glob("*.constraints.txt"))
                for argument_value in ("--constraint", "/input/" + build_path.name)
            ]

            self.invoke(
                command + [
                    "pip",
                    "compile",
                    "/input/requirements.in",
                    *constraints,
                    "--python-version",
                    target["args"]["PY_VERSION"],
                    "--offline",
                    "--no-index",
                    "--find-links",
                    "/pool",
                    "--generate-hashes",
                    "--no-header",
                    "--no-annotate",
                    "--output-file",
                    "/stage/requirements.lock"
                ]
            )

            destination = ROOT / "builds" / profile / "requirements.lock"

            input_header = f"# inputs-sha256: {self.input_hash(profile, target)}\n"  # machine-string: Generated lock identity header.

            # Publish the complete generated lock in one rename.
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".requirements-", delete=False) as stream:
                pending = pathlib.Path(stream.name)

                try:
                    stream.write(input_header.encode())
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

        input_header = f"# inputs-sha256: {self.input_hash(profile, target)}"  # machine-string: Generated lock identity header.

        if not (lock.read_text().splitlines()[0] == input_header):
            raise ValueError(f"Cannot build {profile=}: dependency inputs changed; regenerate with make lock")

        base = target["args"]["PYTHON_BASE"]

        self.invoke(self.docker + ["image", "inspect", base], stdout=subprocess.DEVNULL)

        with tempfile.TemporaryDirectory(prefix="moldetr-wheels-") as temporary:
            stage = pathlib.Path(temporary)

            self.invoke(
                self.container(base, profile, stage) + [
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--no-index",
                    "--find-links",
                    "/pool",
                    "--only-binary=:all:",
                    "--no-deps",
                    "--require-hashes",
                    "--dest",
                    "/stage",
                    "-r",
                    "/input/requirements.lock",
                ]
            )

            (stage / "requirements.lock").write_bytes((ROOT / "builds" / profile / "requirements.lock").read_bytes())

            self.invoke(
                self.docker + [
                    "buildx",
                    "bake",
                    "--builder",
                    "default",
                    "-f",
                    "docker-bake.hcl",
                    profile,
                    "--set",
                    f"{profile}.contexts.wheels={stage}",  # machine-string: Buildx Bake context assignment.
                    "--allow",
                    f"fs.read={stage}",  # machine-string: Buildx filesystem entitlement.
                    "--load"
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("operation", choices=("sdk/acquire", "lock", "build"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--pool", required=True)

    parser.add_argument(
        "--profile",
        choices=sorted(build_path.name for build_path in (ROOT / "builds").iterdir() if build_path.is_dir())
    )

    args = parser.parse_args()

    if (args.operation != "sdk/acquire") and (args.profile is None):
        parser.error("--profile is required for lock and build")

    build = Build(args.host, args.pool)

    if args.operation == "sdk/acquire":
        build.sdk_acquire()
    else:
        getattr(build, args.operation)(args.profile)


if __name__ == "__main__":
    main()
