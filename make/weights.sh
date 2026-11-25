#!/usr/bin/env bash
# Checkpoint recipes share the same Compose project, volume and selected engine.

weights_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$weights_root/make/operator.sh"

weights_compose() {
    "${docker[@]}" compose --project-directory "$weights_root" --env-file /dev/null \
        -f "$weights_root/compose/weights.yaml" "$@"
}

weights_run() {
    weights_compose run --rm --no-deps -T "$@"
}
