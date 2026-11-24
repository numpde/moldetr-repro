# MolDeTr reproduction

The upstream model is pinned as a submodule in `model/upstream`. Source weights
are pinned in `model/checkpoints.sha256` and stored in the Docker volume
`moldetr-weights_checkpoints`.

## Prepare weights

The downloader uses a pinned curl image without Torch. The sanitizer uses Torch
without network or GPU access. Its code and Dockerfile live together in
`containers/weights-sanitize`.

Supply a local Docker context and a wheelhouse containing the locked packages:

```sh
make build PROFILE=weights-sanitize DOCKER_CONTEXT=science WHEELHOUSE=/path/to/wheelhouse
make weights/download DOCKER_CONTEXT=science PROXY_URL=none
make weights/sanitize DOCKER_CONTEXT=science FP=fp16
make weights/verify DOCKER_CONTEXT=science
```

For a proxy, replace `PROXY_URL=none` with its URL as reachable from the
downloader. Omitted operator inputs prompt on a terminal; unattended commands
must provide them explicitly. `FP` supports `keep`, `fp16`, `bf16`, `fp32`, and
`fp64`.

Sanitization verifies the source pin, extracts the model state, optionally casts
floating tensors, and verifies restricted loading. It replaces the original
`.pth` file and publishes `<filename>.sanitized.json` with the resulting hashes.
Successful preparation removes its partial files. A failure after replacement
but before marker publication leaves unmarked weights: restore them with
`weights/download`, then sanitize again. Changing precision also requires
restoring the source first.

The source manifest is maintained input; output hashes are generated beside the
weights. Consumers must hash the selected files against that marker before
loading, using the same open file descriptors. There are no inference or
provider images in this checkout yet; their agreed target layout follows
ChefNMR2's CPU/CUDA worker and provider roles.

## Dependencies and checks

`docker-bake.hcl` owns build targets and pinned interpreter/tool images. The
substantive host workflow is `make/build.py`; Compose owns container execution.
Both locking and building operate without network access and require cached
base images and packages. Package staging uses a rootless Docker engine.

Regenerate a lock after changing a profile's dependency inputs:

```sh
make lock PROFILE=weights-sanitize DOCKER_CONTEXT=science WHEELHOUSE=/path/to/wheelhouse
make test/operator
make test/unit DOCKER_CONTEXT=science
```

`requirements.in` declares the preparation environment. The profile's cached
constraints preserve the previously verified wheel closure; the generated lock
records package hashes and its input identity. Builds reject stale locks and
stage only the selected hash-verified wheels into disposable scratch.
