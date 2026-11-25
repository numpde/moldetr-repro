#!/usr/bin/env bash
# Sourced by Make recipes: select explicit inputs or ask through the terminal.

operator_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

remember_choice() (
    local file=$1 value=$2 option
    shift 2
    local -a recent=("$value")
    for option in "$@"; do
        if [[ $option != "$value" ]]; then recent+=("$option"); fi
        if [[ ${#recent[@]} == 5 ]]; then break; fi
    done
    umask 077
    mkdir -p -- "${file%/*}"
    printf '%s\n' "${recent[@]}" > "$file"
)

choose() {
    local name=$1 choices=${2:-} input="REPRO_$1" value option number valid history=
    local -a options=()
    # Discovery commands may return one choice per line.
    read -ra options <<< "${choices//$'\n'/ }"
    if [[ " $REPRO_EXPLICIT_INPUTS " == *" $name "* ]]; then
        value=${!input}
    else
        if ! { exec 3<>/dev/tty; } 2>/dev/null; then
            printf 'Supply %s=<value> on the make command line; a terminal is required to ask for it.\n' "$name" >&2
            return 2
        fi
        case $name in
            WHEELHOUSE|PROXY_URL)
                history="$operator_root/tmp/inputs/$name"
                if [[ -f $history ]]; then mapfile -t options < "$history"; fi
                ;;
        esac
        for number in "${!options[@]}"; do
            printf '  %s) %s\n' "$((number+1))" "${options[number]}" >&3
        done
        if [[ -n $history ]]; then
            printf 'Select %s or enter a new value (Enter cancels) > ' "$name" >&3
        else
            printf 'Select %s (Enter cancels) > ' "$name" >&3
        fi
        IFS= read -r value <&3 || value=
        exec 3>&-
        for number in "${!options[@]}"; do
            if [[ $value == "$((number+1))" ]]; then
                value=${options[number]}
                break
            fi
        done
    fi
    if [[ -z $value ]]; then
        printf '%s selection cancelled.\n' "$name" >&2
        return 2
    fi
    if [[ -n $choices ]]; then
        valid=
        for option in "${options[@]}"; do
            if [[ $value == "$option" ]]; then valid=yes; break; fi
        done
        if [[ -z $valid ]]; then
            printf 'Invalid %s=%s. Choose from %s.\n' "$name" "$value" "${options[*]}" >&2
            return 2
        fi
    fi
    # History is advisory. Never retain proxy userinfo, queries or fragments.
    if [[ -n $history && ! ( $name == PROXY_URL && ( $value == *@* || $value == *'?'* || $value == *'#'* ) ) ]]; then
        if ! remember_choice "$history" "$value" "${options[@]}"; then
            printf 'Could not remember %s in %s; continuing with the selected value.\n' "$name" "$history" >&2
        fi
    fi
    printf '%s\n' "$value"
}

select_docker() {
    local variable context endpoint contexts=
    for variable in ${!DOCKER_@} ${!COMPOSE_@} ${!BUILDX_@} ${!BUILDKIT_@}; do
        unset "$variable"
    done
    if [[ " $REPRO_EXPLICIT_INPUTS " != *' DOCKER_CONTEXT '* ]]; then
        contexts=$(docker context ls --format '{{.Name}}')
    fi
    context=$(choose DOCKER_CONTEXT "$contexts")
    endpoint=$(docker context inspect "$context" --format '{{.Endpoints.docker.Host}}')
    case $endpoint in
        unix:///*) ;;
        *) printf 'Docker context %s must use a local Unix socket.\n' "$context" >&2; return 2 ;;
    esac
    printf 'Using Docker context %s (%s)\n' "$context" "$endpoint" >&2
    for variable in ${!REPRO_@}; do unset "$variable"; done
    unset MAKEFLAGS MFLAGS GNUMAKEFLAGS MAKEOVERRIDES
    docker=(docker --host "$endpoint")
    docker_host=$endpoint
}
