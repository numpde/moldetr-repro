override SHELL := /bin/bash
override .SHELLFLAGS := -eu -o pipefail -c
override .DEFAULT_GOAL := help
override REPOSITORY_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Capture literal command-line inputs before Make or the shell can interpret them.
# Omitted inputs are selected at the terminal by the function below.
override export REPRO_EXPLICIT_INPUTS := $(foreach name,DOCKER_CONTEXT PROXY_URL WHEELHOUSE FP,$(if $(filter command line,$(origin $(name))),$(name)))
override export REPRO_DOCKER_CONTEXT := $(value DOCKER_CONTEXT)
override export REPRO_PROXY_URL := $(value PROXY_URL)
override export REPRO_WHEELHOUSE := $(value WHEELHOUSE)
override export REPRO_FP := $(value FP)
unexport DOCKER_CONTEXT PROXY_URL WHEELHOUSE FP
override HOST_PYTHON := python3 -B

.PHONY: help
help: ## List workflows; inputs may prompt on a terminal and environment values never select them.
	@awk -F ':.*## ' '/^[a-z][a-z0-9_./-]*:.*## / { printf "  make %-28s %s\n", $$1, $$2 }' "$(REPOSITORY_ROOT)/Makefile"

# One selector owns explicit input, terminal prompting, cancellation and choices.
define operator_inputs
choose() { \
	local name=$$1 choices=$${2:-} input="REPRO_$$1" value option number valid; \
	local -a options=(); \
	read -ra options <<< "$$choices"; \
	if [[ " $$REPRO_EXPLICIT_INPUTS " == *" $$name "* ]]; then \
		value=$${!input}; \
	else \
		if ! { exec 3<>/dev/tty; } 2>/dev/null; then \
			printf 'Supply %s=<value> on the make command line; a terminal is required to ask for it.\n' "$$name" >&2; return 2; \
		fi; \
		for number in "$${!options[@]}"; do printf '  %s) %s\n' "$$((number+1))" "$${options[number]}" >&3; done; \
		printf 'Select %s (Enter cancels) > ' "$$name" >&3; \
		IFS= read -r value <&3 || value=; exec 3>&-; \
		for number in "$${!options[@]}"; do \
			if [[ $$value == "$$((number+1))" ]]; then value=$${options[number]}; break; fi; \
		done; \
	fi; \
	if [[ -z $$value ]]; then printf '%s selection cancelled.\n' "$$name" >&2; return 2; fi; \
	if [[ -n $$choices ]]; then \
		valid=; \
		for option in "$${options[@]}"; do if [[ $$value == "$$option" ]]; then valid=yes; fi; done; \
		if [[ -z $$valid ]]; then printf 'Invalid %s=%s. Choose from %s.\n' "$$name" "$$value" "$$choices" >&2; return 2; fi; \
	fi; \
	printf '%s\n' "$$value"; \
}; \
select_docker() { \
	local variable context endpoint; \
	for variable in $${!DOCKER_@} $${!COMPOSE_@} $${!BUILDX_@} $${!BUILDKIT_@}; do unset "$$variable"; done; \
	context=$$(choose DOCKER_CONTEXT); \
	endpoint=$$(docker context inspect "$$context" --format '{{.Endpoints.docker.Host}}'); \
	case $$endpoint in unix:///*) ;; *) printf 'Docker context %s must use a local Unix socket.\n' "$$context" >&2; return 2;; esac; \
	for variable in $${!REPRO_@}; do unset "$$variable"; done; \
	unset MAKEFLAGS MFLAGS GNUMAKEFLAGS MAKEOVERRIDES; \
	docker=(docker --host "$$endpoint"); \
	docker_host=$$endpoint; \
}; \
weights_compose() { \
	"$${docker[@]}" compose --project-directory "$(REPOSITORY_ROOT)" --env-file /dev/null -f "$(REPOSITORY_ROOT)/compose/weights.yaml" "$$@"; \
}; \
weights_run() { weights_compose run --rm --no-deps --pull never -T "$$@"; }
endef


.PHONY: weights/download weights/sanitize weights/verify weights/build-sanitizer
weights/download: ## Download or restore pinned source weights. Inputs DOCKER_CONTEXT, PROXY_URL=<url-or-none>.
	@$(operator_inputs); \
	proxy=$$(choose PROXY_URL); select_docker; export PROXY_URL="$$proxy"; \
	weights_run volume-init; \
	weights_run --name moldetr-weights-writer download; \
	weights_run verify

weights/verify: ## Verify stored weights offline. Input DOCKER_CONTEXT.
	@$(operator_inputs); select_docker; weights_run verify

weights/sanitize: ## Replace weights and publish a receipt. Inputs DOCKER_CONTEXT, FP=<precision>.
	@$(operator_inputs); \
	choices=$$(awk '!/^\#/ && NF { print $$2 }' "$(REPOSITORY_ROOT)/model/exported.sha256" | paste -sd ' '); \
	test -n "$$choices" || { printf '%s\n' 'No FP choices are available in model/exported.sha256.' >&2; exit 2; }; \
	precision=$$(choose FP "$$choices"); select_docker; \
	weights_run --name moldetr-weights-writer sanitize --fp "$$precision"

weights/build-sanitizer: ## Build from cached locked wheels. Inputs DOCKER_CONTEXT, WHEELHOUSE=<directory>.
	@$(operator_inputs); pool=$$(choose WHEELHOUSE); \
	test -d "$$pool" || { printf 'WHEELHOUSE must name an existing directory: %s\n' "$$pool" >&2; exit 2; }; \
	select_docker; \
	stage=$$(mktemp -d -t moldetr-sanitize-wheels-XXXXXXXX); \
	trap 'rm -rf -- "$$stage"' EXIT; \
	cp "$(REPOSITORY_ROOT)/builds/weights-sanitize/requirements.lock" "$$stage/install.txt"; \
	while read -r wheel checksum; do cp -- "$$pool/$${wheel##*/}" "$$stage/"; done < "$$stage/install.txt"; \
	"$${docker[@]}" buildx build --builder default --platform linux/amd64 --network none \
		--build-context "wheels=$$stage" \
		--tag moldetr/weights-sanitizer:local --load "$(REPOSITORY_ROOT)/containers/weights-sanitize"

.PHONY: test/operator test/operator/normative test/operator/adversarial
test/operator: test/operator/normative test/operator/adversarial ## Run normative and adversarial host operator tests.
test/operator/normative: ## Run normative host operator tests.
test/operator/adversarial: ## Run adversarial host operator tests.
test/operator/normative test/operator/adversarial:
	@PYTHONPATH="$(REPOSITORY_ROOT)/tests/operator" $(HOST_PYTHON) -m unittest discover -v -s "$(REPOSITORY_ROOT)/tests/operator/$(@F)" -p 'test_*.py'


.PHONY: test/unit test/unit/normative test/unit/adversarial
test/unit: test/unit/normative test/unit/adversarial ## Run all normative and adversarial checkpoint unit tests in the sanitizer image.
test/unit/normative: ## Run normative checkpoint unit tests in the sanitizer image; requires DOCKER_CONTEXT.
test/unit/adversarial: ## Run adversarial checkpoint unit tests in the sanitizer image; requires DOCKER_CONTEXT.
test/unit/normative test/unit/adversarial:
	@$(operator_inputs); select_docker; \
	"$${docker[@]}" run --rm --pull=never --network none --read-only \
		--cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 1 --memory 1g \
		--tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
		-e PYTHONPATH=/tests/unit:/opt/sanitize \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/tests/unit,dst=/tests/unit,readonly" \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/containers/weights-sanitize/checkpoints.py,dst=/opt/sanitize/checkpoints.py,readonly" \
		--entrypoint python moldetr/weights-sanitizer:local \
		-B -m unittest discover -v -s "/tests/unit/$(@F)" -p 'test_*.py'
