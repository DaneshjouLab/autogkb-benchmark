# AutoGKB Functional Analysis Benchmark

## Scoring System

### Field Evaluation Methods

#### 1. Exact Match Fields
**Fields:** `PMID`, `isPlural`, `Is/Is Not associated`, `Direction of effect`, `When treated with/exposed to/when assayed with`, `Multiple drugs And/or`

**Scoring:** Binary (1.0 for exact match, 0.0 for mismatch)

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
- Uses a singleton pattern to cache the PubMedBERT model, avoiding reloading on multiple evaluations

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

#### Overall Score
The overall benchmark score is the arithmetic mean of all field scores, and also includes dependency validation. 

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