FROM docker.io/library/python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

RUN --mount=type=bind,from=wheels,target=/wheels \
    python -m pip install --no-index --no-cache-dir --require-hashes -r /wheels/install.txt && python -m pip check
COPY checkpoints.py /opt/sanitize/checkpoints.py
# Match the owner of the sealed volume created by curlimages/curl.
USER 100:101
ENTRYPOINT ["python", "-IB", "/opt/sanitize/checkpoints.py"]
