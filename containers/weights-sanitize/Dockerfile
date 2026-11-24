ARG PYTHON_BASE
FROM ${PYTHON_BASE}

RUN --mount=type=bind,from=wheels,target=/wheels \
    python -m pip install --no-index --no-cache-dir --no-deps --find-links /wheels --require-hashes -r /wheels/requirements.lock && python -m pip check
COPY sanitize.py /opt/sanitize/sanitize.py
# Match the owner of the sealed volume created by curlimages/curl.
USER 100:101
ENTRYPOINT ["python", "-IB", "/opt/sanitize/sanitize.py"]
