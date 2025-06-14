# --- Configuration ---
SHELL := /usr/bin/bash
PYTHON := python3

# --- Directories ---
RAW_DATA_DIR := data/raw
CONVERTED_DATA_DIR := data/converted

# --- Source files ---
CONVERTER_SCRIPT := scripts/convert_txt_to_npz.py
UTILS_SRC := src/radarsig/utils.py

# --- Data ---
RAW_FILES := $(shell find $(RAW_DATA_DIR) -name "*.txt")
CONVERTED_FILES := $(patsubst $(RAW_DATA_DIR)/%.txt, $(CONVERTED_DATA_DIR)/%.npz, $(RAW_FILES))

# --- Main target ---
.PHONY: all
all: $(CONVERTED_FILES) ## Run conversion from txt to npz

$(CONVERTED_DATA_DIR)/%.npz: $(RAW_DATA_DIR)/%.txt $(CONVERTER_SCRIPT) $(UTILS_SRC)
	@echo "Converting $< -> $@"
	@mkdir -p $(dir $@)
	$(PYTHON) $(CONVERTER_SCRIPT) $< $@

# --- Environment ---
ENV_DIR := envs
ENV_PATH := $(shell pwd)/$(ENV_DIR)

.PHONY: env
env: ## Create/update the conda environment from environment.yaml.
	@if [ -d "$(ENV_PATH)" ]; then \
		echo "Conda environment '$(ENV_PATH)' already exists."; \
		echo "Updating environment '$(ENV_PATH)'..."; \
		conda env update -f environment.yaml --prune --prefix $(ENV_PATH); \
	else \
		echo "Creating conda environment '$(ENV_PATH)'..."; \
		conda env create -f environment.yaml --prefix $(ENV_PATH); \
	fi

# --- Tags ---
.PHONY: tags
tags: ## Create tags using Universal Ctags
	@ctags -R --exclude=envs --exclude=docs --exclude=.*/* --exclude=Makefile .

# --- Vars ---
.PHONY: vars
vars: ## Print variables for debug
	$(info RAW_FILES is $(RAW_FILES))
	$(info CONVERTED_FILES is $(CONVERTED_FILES))

# --- Help ---
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# --- Clean ---
.PHONY: clean
clean: ## Remove all created files
	@rm -rf $(CONVERTED_DATA_DIR)
