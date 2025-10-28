# AutoGKB Functional Analysis Benchmark

This benchmark evaluates the performance of Large Language Models (LLMs) on functional analysis annotation tasks in pharmacogenomics. It compares model predictions against expert-curated ground truth annotations from the PharmGKB database.

## Overview

The benchmark focuses on functional analysis annotations (`var_fa_ann`) which describe how genetic variants affect drug metabolism, efficacy, and toxicity. These annotations are critical for personalized medicine and pharmacogenomic decision-making.

## Scoring System

The benchmark uses a comprehensive scoring system that evaluates 20 different fields across each annotation. The overall score is calculated as the average of all field scores.

### Field Evaluation Methods

#### 1. Exact Match Fields
**Fields:** `PMID`, `isPlural`, `Is/Is Not associated`, `Direction of effect`, `When treated with/exposed to/when assayed with`, `Multiple drugs And/or`

**Scoring:** Binary (1.0 for exact match, 0.0 for mismatch)
- Case-insensitive comparison
- Handles null values appropriately
- Perfect precision required

#### 2. Category Classification Fields
**Fields:** `Phenotype Category`, `Significance`

**Scoring:** Binary (1.0 for correct category, 0.0 for incorrect)

**Valid Categories:**
- **Phenotype Category:** `efficacy`, `toxicity`, `dosage`, `metabolism/pk`, `pd`, `other`
- **Significance:** `yes`, `no`, `not stated`

#### 3. Semantic Similarity Fields
**Fields:** `Gene`, `Drug(s)`, `Alleles`, `Specialty Population`, `Assay type`, `Metabolizer types`, `Functional terms`, `Gene/gene product`, `Cell type`, `Comparison Allele(s) or Genotype(s)`, `Comparison Metabolizer types`

**Scoring:** Continuous (0.0 to 1.0) using PubMedBERT embeddings
- Uses `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` model
- Cosine similarity between embeddings
- Fallback to sequence matching if embedding fails
- Handles biomedical terminology effectively

#### 4. Variant Coverage Field
**Field:** `Variant/Haplotypes`

**Scoring:** Coverage-based (0.0 to 1.0)
- **Parsing:** Handles multiple separators (`,`, `;`, `|`, `+`)
- **Wild-type handling:** Excludes wild-type variants from penalty calculation
- **Variant types:**
  - **rsIDs:** Exact match required (e.g., `rs123456`)
  - **Star alleles:** Exact match required (e.g., `CYP2D6*2`)
  - **Phenotype descriptions:** Semantic similarity with 0.8 threshold
- **Coverage calculation:** `covered_variants / total_non_wildtype_variants`

### Score Calculation

#### Individual Field Scores
Each field is evaluated independently for every annotation:
```python
field_score = evaluator(ground_truth_value, predicted_value)
```

#### Overall Score
The overall benchmark score is the arithmetic mean of all field scores:
```python
overall_score = sum(field_scores) / len(field_scores)
```

#### Sample-Level Scores
Each annotation receives individual scores for all fields, allowing for detailed error analysis.

## Example Scoring

### Perfect Match (Score: 1.0)
```
Ground Truth: CYP2D6*2 is associated with decreased metabolism of codeine
Prediction:   CYP2D6*2 is associated with decreased metabolism of codeine
```

### Partial Match (Score: 0.85)
```
Ground Truth: CYP2D6*2 is associated with decreased metabolism of codeine
Prediction:   CYP2D6*2 is associated with reduced metabolism of codeine
```
- Variant/Haplotypes: 1.0 (exact match)
- Gene: 1.0 (exact match)
- Drug(s): 1.0 (exact match)
- Direction of effect: 1.0 (exact match)
- Functional terms: 0.7 (semantic similarity: "decreased" vs "reduced")

### Poor Match (Score: 0.2)
```
Ground Truth: CYP2D6*2 is associated with decreased metabolism of codeine
Prediction:   CYP3A4*1 is associated with increased clearance of warfarin
```
- Variant/Haplotypes: 0.0 (no overlap)
- Gene: 0.0 (different gene)
- Drug(s): 0.0 (different drug)
- Direction of effect: 0.0 (opposite direction)
- Functional terms: 0.0 (different functional terms)

## Benchmark Data

The benchmark uses functional analysis annotations from PharmGKB, containing:
- **Total annotations:** ~2,000+ functional analysis records
- **Fields per annotation:** 20 standardized fields
- **Coverage:** Multiple genes, drugs, and variant types
- **Quality:** Expert-curated ground truth

## Usage

### Running the Benchmark
```bash
# Install dependencies
pixi install

# Run benchmark with mock predictions
pixi run python run_benchmark.py

# Run with custom predictions
python -c "
from src.fa_benchmark.fa_benchmark import evaluate_functional_analysis
import json

# Load your predictions
with open('your_predictions.json', 'r') as f:
    predictions = json.load(f)

# Load ground truth
with open('data/benchmark_annotations.json', 'r') as f:
    data = json.load(f)

# Extract functional analysis annotations
gt_annotations = []
for pmcid, article_data in data.items():
    if 'var_fa_ann' in article_data:
        gt_annotations.extend(article_data['var_fa_ann'])

# Run evaluation
results = evaluate_functional_analysis(gt_annotations, predictions)
print(f'Overall Score: {results[\"overall_score\"]:.3f}')
"
```

### Interpreting Results

#### Overall Score Interpretation
- **0.9-1.0:** Excellent performance
- **0.8-0.9:** Good performance
- **0.7-0.8:** Moderate performance
- **0.6-0.7:** Poor performance
- **<0.6:** Very poor performance

#### Field-Specific Analysis
Focus on fields with low scores to identify model weaknesses:
- **Variant/Haplotypes:** Model struggles with variant identification
- **Gene:** Model has difficulty with gene name recognition
- **Drug(s):** Model fails to identify correct drugs
- **Direction of effect:** Model confuses effect directions

## Technical Details

### Dependencies
- `sentence-transformers`: For PubMedBERT embeddings
- `numpy`: For numerical computations
- `difflib`: For sequence matching fallback

### Performance Considerations
- PubMedBERT model loads once per evaluation
- Embedding computation is the main bottleneck
- Consider caching embeddings for repeated evaluations

### Error Handling
- Graceful fallback to sequence matching if embedding fails
- Proper handling of null values
- Robust parsing of variant strings

## Contributing

To improve the benchmark:
1. Add new evaluation metrics
2. Expand the dataset
3. Improve scoring algorithms
4. Add visualization tools

## Citation

If you use this benchmark, please cite:
```
AutoGKB Functional Analysis Benchmark
[Your citation details here]
```
