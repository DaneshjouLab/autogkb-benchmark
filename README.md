# AutoGKB Benchmark System

A comprehensive benchmarking system for evaluating the quality of genomic knowledge base annotations.

## Overview

This benchmark system evaluates four types of annotations:

1. **Drug Annotations** (`var_drug_ann`) - Drug-gene-variant associations
2. **Phenotype Annotations** (`var_pheno_ann`) - Phenotype-gene-variant associations
3. **Functional Analysis** (`var_fa_ann`) - Functional effects of variants
4. **Study Parameters** (`study_parameters`) - Study design and statistical parameters

Each annotation type is scored independently using specialized comparison metrics, and a weighted overall score is computed.

## Features

### Core Functionality

- **Automatic alignment** of annotations by variant, rsID, gene, and drug
- **Field-level scoring** with semantic similarity and exact matching
- **Dependency validation** to catch logical inconsistencies
- **Penalty system** for invalid or inconsistent annotations
- **Unmatched tracking** to identify missing and hallucinated annotations
- **Weighted scoring** to prioritize important fields

### Display Functions

- **Mismatch analysis** showing what's wrong and where
- **Low-score filtering** to focus on problematic areas
- **Field-by-field comparison** of ground truth vs predictions
- **Aggregate statistics** across all files and benchmarks

## Quick Start

### Run benchmark on all files:

```bash
PYTHONPATH=src pixi run python src/benchmarks/run_benchmark.py
```

### Run benchmark on a single file:

```bash
PYTHONPATH=src pixi run python src/benchmarks/run_benchmark.py --single_file PMC5508045
```

### Show detailed mismatches:

```bash
PYTHONPATH=src pixi run python src/benchmarks/run_benchmark.py \
    --single_file PMC5508045 \
    --show_mismatches
```

## Files Created

### Main Implementation

- **`src/benchmarks/run_benchmark.py`** - Main benchmark runner with command-line interface
  - `run_single_benchmark()` - Run benchmarks on one annotation file
  - `run_all_benchmarks()` - Run benchmarks on all files in directories
  - `display_mismatches()` - Show detailed mismatch information for one file
  - `display_all_mismatches()` - Show mismatches for multiple files

### Documentation

- **`BENCHMARK_USAGE.md`** - Detailed usage guide with examples
- **`example_usage.py`** - Python examples showing how to use the API
- **`README_BENCHMARK.md`** - This file

## Benchmark Scoring

### Overall Score Calculation

The overall score is a weighted average of individual benchmark scores:

- Drug Annotations: **weight 1.5** (high priority)
- Phenotype Annotations: **weight 1.5** (high priority)
- Functional Analysis: **weight 1.0**
- Study Parameters: **weight 1.0**

### Field Scoring

Each field is scored using appropriate comparison metrics:

- **Exact Match** (0 or 1): PMID, IDs
- **Category Equal** (0 or 1): Categorical fields like "Significance", "Study Type"
- **Semantic Similarity** (0-1): Text fields using PubMedBERT embeddings
- **Variant Substring Match** (0-1): Variant/Haplotype strings
- **Numeric Tolerance** (0-1): Numeric fields with tolerance bands
- **P-value Match** (0-1): Special handling for p-values with operators

### Score Interpretation

- **1.0** - Perfect match
- **0.9-0.99** - Excellent match (small differences)
- **0.7-0.89** - Good match (minor issues)
- **0.5-0.69** - Fair match (significant differences)
- **0.0-0.49** - Poor match (major issues)

## Understanding Output

### Single File Results

```python
{
  "pmid": "28550460",
  "pmcid": "PMC5508045",
  "title": "The impact of non-genetic...",
  "overall_score": 0.197,
  "num_benchmarks": 4,
  "benchmarks": {
    "drug_annotations": {
      "overall_score": 0.357,
      "total_samples": 4,
      "field_scores": {...},
      "detailed_results": [...],
      "unmatched_ground_truth": [...],  # Missing annotations
      "unmatched_predictions": [...]     # Hallucinated annotations
    },
    ...
  }
}
```

### Key Metrics

- **`overall_score`**: Weighted average across all benchmarks
- **`field_scores`**: Mean score for each field across all samples
- **`detailed_results`**: Per-sample field-by-field scores and values
- **`unmatched_ground_truth`**: Annotations that should exist but are missing
- **`unmatched_predictions`**: Annotations that exist but shouldn't (hallucinations)

## Common Use Cases

### 1. Evaluate a new annotation system

```bash
PYTHONPATH=src pixi run python src/benchmarks/run_benchmark.py \
    --proposed_dir data/my_new_annotations \
    --output_file my_results.json
```

### 2. Find the worst-performing files

```bash
PYTHONPATH=src pixi run python src/benchmarks/run_benchmark.py \
    --show_mismatches \
    --show_only_low_scores \
    --score_threshold 0.5
```

### 3. Compare two annotation systems

```python
from benchmarks.run_benchmark import run_all_benchmarks

results_a = run_all_benchmarks(
    ground_truth_dir='data/benchmark_annotations',
    proposed_dir='data/system_a_annotations',
    verbose=False
)

results_b = run_all_benchmarks(
    ground_truth_dir='data/benchmark_annotations',
    proposed_dir='data/system_b_annotations',
    verbose=False
)

print(f"System A: {results_a['overall_mean_score']:.3f}")
print(f"System B: {results_b['overall_mean_score']:.3f}")
```

### 4. Analyze specific fields

```python
from benchmarks.run_benchmark import run_single_benchmark

result = run_single_benchmark(
    ground_truth_file='data/benchmark_annotations/PMC5508045.json',
    proposed_file='data/proposed_annotations/PMC5508045.json'
)

# Check drug annotation field scores
drug_results = result['benchmarks']['drug_annotations']
for field, stats in drug_results['field_scores'].items():
    if stats['mean_score'] < 0.7:
        print(f"Low score for {field}: {stats['mean_score']:.3f}")
```

## Alignment Strategy

The benchmark uses a sophisticated alignment strategy to match annotations:

1. **Variant ID matching** (highest priority) - Uses normalized variant IDs
2. **rsID matching** - Matches by rsID intersection
3. **Substring matching** - Normalized variant string containment
4. **Gene + Drug matching** (fallback) - For phenotype annotations

This ensures that annotations are correctly paired even when variant representations differ.

## Dependency Validation

The system checks for logical inconsistencies:

### Drug Annotations
- Direction of effect requires "Associated with" status
- Comparison alleles require variants to be specified
- Multiple drugs operator should match drugs field

### Functional Analysis
- Gene product requires gene to be specified
- Comparison alleles require variants
- Functional terms should have gene product

### Study Parameters
- Confidence intervals must be valid ranges
- Ratio stat should be within confidence interval
- Frequencies should be between 0 and 1

Violations result in penalties (up to 30% reduction in field scores).

## Current Results (Example Run)

Based on the current proposed annotations:

```
Total files processed: 32
Overall Mean Score: 0.244

Per-Benchmark Statistics:
  drug_annotations:     Mean: 0.104 (needs improvement)
  phenotype_annotations: Mean: 0.346 (fair)
  functional_analysis:   Mean: 0.001 (critical issues)
  study_parameters:      Mean: 0.547 (good)
```

## Next Steps

1. **Improve functional analysis** - Currently scoring very low (0.001)
2. **Enhance drug annotations** - Many unmatched variants and missing fields
3. **Fix PMID fields** - Currently scoring 0.0 in many cases
4. **Validate variant formats** - Many variants are incorrectly formatted

## Dependencies

Required packages (installed via pixi):
- `sentence-transformers` - Semantic similarity with PubMedBERT
- `scikit-learn` - Utility functions
- `numpy` - Numerical operations
- Standard library: `json`, `pathlib`, `typing`, `re`, `difflib`

## Contributing

When adding new benchmark features:

1. Add field evaluators in the appropriate `*_benchmark.py` file
2. Update `shared_utils.py` for common utilities
3. Test with `run_benchmark.py --single_file` before full runs
4. Document new metrics in this README

## Troubleshooting

### Import errors
Make sure `PYTHONPATH=src` is set when running scripts.

### Missing dependencies
Run `pixi install` to install all required packages.

### Low scores
Use `--show_mismatches` to see exactly what's wrong.

### Slow execution
Use `--quiet` flag and consider running on a subset of files first.
