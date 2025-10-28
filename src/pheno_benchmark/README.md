# AutoGKB Benchmark

Benchmarking tools for evaluating automated phenotype annotation extraction.

## Phenotype Annotation Benchmark

Evaluates how well predicted phenotype annotations match ground truth annotations by comparing key fields with weighted importance.

### Usage

```python
from src.var_pheno_benchmark import benchmark_phenotype_annotations

# Your model's predictions
predictions = [
    {
        "Variant/Haplotypes": "rs6539870",
        "Gene": None,
        "Drug(s)": "etoposide",
        "Phenotype": "Other:sensitivity to etoposide",
        "Alleles": "GG",
        "Direction of effect": "increased",
        "Is/Is Not associated": "Associated with",
        # ... other fields
    }
]

# Ground truth annotations
ground_truths = [
    {
        "Variant/Haplotypes": "rs6539870",
        "Drug(s)": "etoposide",
        "Phenotype": "Other:sensitivity to etoposide",
        # ... other fields
    }
]

# Get similarity score (0-100)
score = benchmark_phenotype_annotations(predictions, ground_truths)
print(f"Model Score: {score:.1f}/100")
```

### How It Works

1. **Field Comparison**: Compares 10 core annotation fields with weighted importance:
   - High priority (1.5-2.0×): Phenotype, Drug(s), Direction of effect, Alleles
   - Standard (1.0×): Gene, Variant/Haplotypes, Is/Is Not associated
   - Lower priority (0.5×): Phenotype Category, When treated with

2. **Similarity Matching**: Each field comparison uses:
   - Exact match → 1.0
   - Substring match → 0.8
   - Token-based Jaccard similarity → 0.0-1.0

3. **Annotation Matching**: Handles many-to-one scenarios where multiple predicted annotations can match a single ground truth

4. **Score Calculation**: Returns average weighted similarity across all ground truth annotations (0-100 scale)

### Parameters

```python
benchmark_phenotype_annotations(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    matching_threshold: float = 0.7  # Minimum score to consider a match
)
```

### Examples

#### Perfect Match
```python
predictions = [{"Phenotype": "sensitivity", "Drug(s)": "etoposide", ...}]
ground_truths = [{"Phenotype": "sensitivity", "Drug(s)": "etoposide", ...}]
score = benchmark_phenotype_annotations(predictions, ground_truths)
# score ≈ 100.0
```

#### Partial Match
```python
predictions = [{"Phenotype": "increased sensitivity", "Drug(s)": "etoposide", ...}]
ground_truths = [{"Phenotype": "sensitivity to drug", "Drug(s)": "etoposide", ...}]
score = benchmark_phenotype_annotations(predictions, ground_truths)
# score ≈ 70-85 (depends on other fields)
```

#### Missing Annotation
```python
predictions = [{"Phenotype": "sensitivity", ...}]
ground_truths = [
    {"Phenotype": "sensitivity", ...},
    {"Phenotype": "toxicity", ...}  # Not captured by model
]
score = benchmark_phenotype_annotations(predictions, ground_truths)
# score ≈ 50 (only 1 of 2 ground truths matched)
```

### Key Features

- **Flexible matching**: Handles case differences, punctuation, word order
- **Many-to-one support**: Multiple predictions can match the same ground truth
- **Weighted fields**: Important fields (Phenotype, Drug) have higher impact
- **Normalized scoring**: Score normalized by ground truth count for consistency

### Required Fields

The benchmark compares these fields from your annotation dictionaries:
- `Variant/Haplotypes`
- `Gene`
- `Drug(s)`
- `Phenotype Category`
- `Alleles`
- `Is/Is Not associated`
- `Direction of effect`
- `Phenotype`
- `When treated with/exposed to/when assayed with`
- `Comparison Allele(s) or Genotype(s)`

Missing or `None` values are handled automatically.
