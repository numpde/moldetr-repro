#!/usr/bin/env bash
# Translate ChefNMR2's artifact command into the shared acquisition lifecycle.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
endpoint=$1 image=$2 route=$3
shift 3
mode=argv
if [[ $1 == --compose ]]; then
    mode=compose
    shift
    composition=("$@")
else
    grants=()
    while [[ $1 != -- ]]; do grants+=("$1"); shift; done
    shift
    application=("$@")
fi
[[ $route != none ]] || route=system
bundle=$root/make/bootstrap/acquisition
printf '%s\n' 'Checking acquisition support files' >&2
python3 -I -S -B "$bundle/acquisition_tool/source_update.py" verify --source-dir "$bundle"
source "$bundle/acquisition_tool/owner.sh"
run_acquisition() {
    local status
    acquisition_set_route "$route"
    if [[ $mode == compose ]]; then
        acquisition_compose_worker online status "${composition[@]}"
    else
        acquisition_worker online status "$image" grants application
    fi
    printf 'Acquisition worker exited with status %s; retiring network helpers\n' "$status" >&2
    return "$status"
}
printf '%s\n' 'Starting acquisition session: preparing identity, network route and worker' >&2
acquisition_scope --endpoint "$endpoint" --bootstrap-image "$image" -- run_acquisition
printf '%s\n' 'Acquisition session retired' >&2
