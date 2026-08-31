# --- Configuration ---
SHELL := /usr/bin/bash
PYTHON := python3

# --- Directories ---
RAW_DATA_DIR := data/raw
CONVERTED_DATA_DIR := data/converted

# --- Source files ---
CONVERTER_SCRIPT := scripts/convert_txt_to_npz.py
PARSER_SRC := src/radarsig/parsers.py

# --- Data ---
RAW_FILES := $(shell find $(RAW_DATA_DIR) -name "*.txt")
CONVERTED_FILES := $(patsubst $(RAW_DATA_DIR)/%.txt, $(CONVERTED_DATA_DIR)/%.npz, $(RAW_FILES))

# --- Main target ---
.PHONY: all
all: $(CONVERTED_FILES) ## Run conversion from txt to npz

$(CONVERTED_DATA_DIR)/%.npz: $(RAW_DATA_DIR)/%.txt $(CONVERTER_SCRIPT) $(PARSER_SRC)
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

# --- FPGA Analysis ---
IQ_DIR := data/raw/fpga/iq
IQ_I_FILES := $(shell find $(IQ_DIR) -name "*_i.data")
PAIR_IDS := $(notdir $(basename $(patsubst %_i.data,%,$(IQ_I_FILES))))

MEDIAN_PLOTS := $(patsubst %, results/fpga_spectrum/median/filter_spectrum_median_%.png, $(PAIR_IDS))
CSD_PLOTS := $(patsubst %, results/fpga_spectrum/csd/filter_spectrum_csd_%.png, $(PAIR_IDS))
COHERENCE_PLOTS := $(patsubst %, results/fpga_spectrum/coherence/filter_spectrum_coherence_%.png, $(PAIR_IDS))

WINDOW ?= hamming
FPGA_FLAGS := --window $(WINDOW)

results/fpga_spectrum/median/filter_spectrum_median_%.png: $(IQ_DIR)/%_i.data $(IQ_DIR)/%_q.data scripts/fpga_analyze_spectrum.py
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/fpga_analyze_spectrum.py --method median --pair $* --out-dir $(dir $@) $(FPGA_FLAGS)

results/fpga_spectrum/csd/filter_spectrum_csd_%.png: $(IQ_DIR)/%_i.data $(IQ_DIR)/%_q.data scripts/fpga_analyze_spectrum.py
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/fpga_analyze_spectrum.py --method csd --pair $* --out-dir $(dir $@) $(FPGA_FLAGS)

results/fpga_spectrum/coherence/filter_spectrum_coherence_%.png: $(IQ_DIR)/%_i.data $(IQ_DIR)/%_q.data scripts/fpga_analyze_spectrum.py
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/fpga_analyze_spectrum.py --method coherence --pair $* --out-dir $(dir $@) $(FPGA_FLAGS)

.PHONY: fpga-analyze-median
fpga-analyze-median: $(MEDIAN_PLOTS) ## Run FPGA median IIR filter spectrum analysis

.PHONY: fpga-analyze-csd
fpga-analyze-csd: $(CSD_PLOTS) ## Run FPGA CSD IIR filter spectrum analysis

.PHONY: fpga-analyze-coherence
fpga-analyze-coherence: $(COHERENCE_PLOTS) ## Run FPGA coherence analysis

.PHONY: fpga-analyze-all
fpga-analyze-all: fpga-analyze-median fpga-analyze-csd fpga-analyze-coherence ## Run all FPGA spectral analyses

# --- FPGA Noise Analysis ---
NOISE_OUT_DIR := results/noise
NOISE_CSVS := $(patsubst %, $(NOISE_OUT_DIR)/noise_stats_%.csv, $(PAIR_IDS))

$(NOISE_CSVS): scripts/fpga_analyze_noise.py $(IQ_I_FILES)
	@mkdir -p $(NOISE_OUT_DIR)
	$(PYTHON) scripts/fpga_analyze_noise.py --out-dir $(NOISE_OUT_DIR)

.PHONY: fpga-analyze-noise
fpga-analyze-noise: $(NOISE_CSVS) ## Run FPGA noise noise analysis (ranges plots, summary plots, and CSVs)

# --- Tags ---
.PHONY: tags
tags: ## Create tags using Universal Ctags
	@ctags -R --exclude=envs --exclude=docs --exclude=.*/* --exclude=Makefile .

# --- Vars ---
.PHONY: vars
vars: ## Print variables for debug
	$(info RAW_FILES is $(RAW_FILES))
	$(info CONVERTED_FILES is $(CONVERTED_FILES))
	$(info IQ_I_FILES is $(IQ_I_FILES))
	$(info NOISE_CSVS is $(NOISE_CSVS))

# --- Help ---
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# --- Clean ---
.PHONY: clean
clean: ## Remove all created files
	@rm -rf $(CONVERTED_DATA_DIR) results
	@rm -rf $(MEDIAN_PLOTS) results
	@rm -rf $(CSD_PLOTS) results
	@rm -rf $(COHERENCE_PLOTS) results

