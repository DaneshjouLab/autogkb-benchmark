# AutoGKB Benchmark System

A comprehensive system for generating and evaluating pharmacogenomic knowledge base annotations from biomedical literature.

## Overview

This repository contains:

- **`generation/`** — Multi-stage pipeline for extracting pharmacogenomic annotations from articles (variants, sentences, citations, summaries).
- **`benchmark/`** — Evaluation suite for scoring generated annotations against ground truth (V1 comprehensive + V2 per-stage modular).
- **`shared/`** — Shared utilities, data setup, and term normalization lookups.
- **[`pubmed-markdown`](https://pypi.org/project/pubmed-markdown/)** — Article downloading tools (PMID → PMCID → HTML → Markdown), installed as a PyPI dependency.

## Setup

### Install dependencies

```bash
pixi install
```

### Set up data

```bash
pixi run setup-repo
```

This runs `pixi install` followed by the data setup script (`python -m shared.data_setup.main`).

### Environment variables

Create a `.env` file in the project root with the following keys:

```
NCBI_EMAIL=your_email@example.com
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

## Generation Pipeline

The generation pipeline extracts pharmacogenomic annotations from PubMed Central articles through five stages:

1. **Variant Finding** — Extracts genetic variants from full-text articles.
2. **Term Normalization** — Normalizes variant names against PharmGKB.
3. **Sentence Generation** — Generates sentences describing clinical significance of each variant.
4. **Citation Finding** — Identifies source sentences from the original article supporting each generated sentence.
5. **Summary Generation** — Creates a concise summary of key pharmacogenomic findings.

Each stage has multiple method implementations under `generation/modules/<stage>/methods/`.

### Running the pipeline

```bash
# Run on specific PMIDs (auto-converts to PMCIDs)
pixi run generate --pmid 12345678

# Run on specific PMCIDs
pixi run generate --pmcids PMC5508045

# Run specific stages only
pixi run generate --pmcids PMC5508045 --stages variants,sentences

# Run on N articles from the benchmark set
pixi run generate --num-pmcids 5

# Use a pre-computed variants file
pixi run generate --pmcids PMC5508045 --variants-file path/to/variants.json

# Use a custom config
pixi run generate --config path/to/config.yaml
```

Output is written to `data/generations.jsonl` as JSONL records.

## Benchmarks

### Benchmark V1

The original comprehensive benchmark evaluates four annotation types:

1. **Drug Annotations** (`var_drug_ann`) — Drug-gene-variant associations
2. **Phenotype Annotations** (`var_pheno_ann`) — Phenotype-gene-variant associations
3. **Functional Analysis** (`var_fa_ann`) — Functional effects of variants
4. **Study Parameters** (`study_parameters`) — Study design and statistical parameters

```bash
# Run benchmark on all files
pixi run benchmark

# Run on a single file
python -m benchmark.v1.run_benchmark --single_file PMC5508045

# Show detailed mismatches
python -m benchmark.v1.run_benchmark --single_file PMC5508045 --show_mismatches
```

### Benchmark V2

Per-stage modular evaluations for the generation pipeline. Includes benchmarks for variant matching, sentence validation, citation evaluation, summary evaluation, and field extraction.

#### Variant benchmark

```bash
# Score variants from an annotation file
python -m benchmark.v2.variant_bench score_annotation <path_to_annotation_file>

# Score all annotations in a directory
python -m benchmark.v2.variant_bench score_all_annotations --annotations_dir <path>

# Score variants from a generated JSON file
python -m benchmark.v2.variant_bench score_generated_variants <path_to_file>
```

#### Other V2 benchmarks

Evaluation runners for each stage are in `benchmark/v2/eval/`:

- `variant_eval.py` / `variant_run.py`
- `sentence_eval.py` / `sentence_run.py`
- `citation_eval.py` / `citation_run.py`
- `summary_eval.py` / `summary_run.py`

## Project Structure

```
autogkb-benchmark/
├── generation/           # Annotation generation pipeline
│   ├── pipeline.py       # Main pipeline orchestrator
│   ├── models.py         # Pydantic models (GenerationRecord, etc.)
│   ├── configs/          # YAML pipeline configs
│   └── modules/          # Pipeline stage implementations
│       ├── variant_finding/
│       ├── term_normalization/
│       ├── sentence_generation/
│       ├── citations/
│       └── summary/
├── benchmark/            # Evaluation suite
│   ├── eval/             # Eval pipeline
│   ├── v1/               # Comprehensive benchmark
│   └── v2/               # Per-stage modular benchmarks
│       └── eval/         # Stage-specific evaluators
├── shared/               # Shared utilities
│   ├── utils.py          # Common helpers (LLM calls, paths)
│   ├── data_setup/       # Data download and setup scripts
│   └── term_normalization/  # Term lookup and normalization
├── data/                 # Articles, ground truth, generated outputs
│   ├── articles/         # Downloaded article markdown
│   ├── cache/            # BioC supplement cache
│   └── generations.jsonl # Pipeline output
└── pixi.toml             # Dependency and task definitions
```

## Dependencies

Managed with [pixi](https://pixi.sh). Key dependencies include:

- `litellm` — Unified LLM API
- `sentence-transformers` — Semantic similarity
- `scikit-learn`, `numpy`, `pandas`
- `pydantic` — Data models
- `loguru` — Logging
- `biopython`, `requests` — PubMed/NCBI access

Install with:

```bash
pixi install
```

The project is also installable as a Python package (`autogkb-pipeline`) via `pip install -e .`, which exposes the `generate` CLI command.