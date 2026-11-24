variable "WHEELS_DIRECTORY" { default = "/missing-moldetr-selected-wheels" }

target "_python312" {
  platforms = ["linux/amd64"]
  network = "none"
  args = {
    PYTHON_BASE = "docker.io/library/python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487"
    PY_VERSION = "3.12.16"
  }
}

target "weights-sanitize" {
  inherits = ["_python312"]
  context = "containers/weights-sanitize"
  contexts = { wheels = WHEELS_DIRECTORY }
  tags = ["moldetr/weights-sanitizer:local"]
}

target "lock" {
  inherits = ["_python312"]
  context = "containers/lock"
  contexts = { uv = "docker-image://ghcr.io/astral-sh/uv@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0" }
  tags = ["moldetr/lock:local"]
}
