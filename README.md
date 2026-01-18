# AutoGKB Benchmark System

A comprehensive benchmarking system for evaluating the quality of genomic knowledge base annotations.

## Overview

This repository contains two versions of the AutoGKB benchmark system:

*   **Benchmark V1**: The original, comprehensive benchmark that evaluates four types of annotations.
*   **Benchmark V2**: A newer, more modular benchmark that is currently focused on variant matching.

## Benchmark V1

The original benchmark system evaluates four types of annotations:

1.  **Drug Annotations** (`var_drug_ann`) - Drug-gene-variant associations
2.  **Phenotype Annotations** (`var_pheno_ann`) - Phenotype-gene-variant associations
3.  **Functional Analysis** (`var_fa_ann`) - Functional effects of variants
4.  **Study Parameters** (`study_parameters`) - Study design and statistical parameters

### Quick Start (V1)

#### Run benchmark on all files:

```bash
PYTHONPATH=src pixi run python src/benchmark_v1/run_benchmark.py
```

#### Run benchmark on a single file:

```bash
PYTHONPATH=src pixi run python src/benchmark_v1/run_benchmark.py --single_file PMC5508045
```

#### Show detailed mismatches:

```bash
PYTHONPATH=src pixi run python src/benchmark_v1/run_benchmark.py \
    --single_file PMC5508045 \
    --show_mismatches
```

### V1 Scoring

The overall score is a weighted average of individual benchmark scores. Each field is scored using appropriate comparison metrics, including exact match, semantic similarity, and numeric tolerance. The system also performs dependency validation to penalize logical inconsistencies.

For more details on the V1 benchmark, please refer to the original `README_BENCHMARK.md`.

## Benchmark V2

The V2 benchmark is a newer, more modular system designed for focused evaluations. It currently includes a benchmark for variant matching.

### Variant Benchmark (V2)

The variant benchmark (`variant_bench.py`) compares a list of proposed variants against a ground truth set, calculating match rates, misses, and extras.

### Quick Start (V2)

#### Score variants from a proposed annotation file:

```bash
PYTHONPATH=src pixi run python src/benchmark_v2/variant_bench.py score_annotation <path_to_annotation_file>
```

#### Score all annotations in a directory:

```bash
PYTHONPATH=src pixi run python src/benchmark_v2/variant_bench.py score_all_annotations --annotations_dir <path_to_annotations_dir>
```

#### Score variants from a generated JSON file:

```bash
PYTHONPATH=src pixi run python src/benchmark_v2/variant_bench.py score_generated_variants <path_to_generated_variants_file>
```

### V2 Output

The V2 variant benchmark provides a JSON output with the following structure:

```json
{
  "timestamp": "",
  "run_name": "",
  "total_match_rate": 0.0,
  "per_annotation_scores": [
    {
      "pmcid": "PMC5508045",
      "title": "",
      "match_rate": 0.0,
      "matches": [],
      "misses": [],
      "extras": []
    }
  ]
}
```

## Experiments

The `src/experiments` directory contains scripts and notebooks for developing and testing new features, such as improved methods for variant extraction.

## Dependencies

Required packages are managed with `pixi` and are listed in the `pixi.toml` file. Key dependencies include:

-   `sentence-transformers`
-   `scikit-learn`
-   `numpy`
-   `pydantic`

To install all dependencies, run:

```bash
pixi install
```

## Contributing

When adding new features:

1.  Add new benchmark modules to the `src/benchmark_v2` directory.
2.  Ensure that new features are tested.
3.  Document new metrics and functionalities in this README.