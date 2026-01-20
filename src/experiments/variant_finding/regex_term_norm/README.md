# Regex + Term Normalization Hybrid Extraction

## Overview

This approach combines:
1. **Regex extraction (v5)**: Pattern-based variant extraction with BioC supplements and SNP expansion
2. **Post-extraction normalization**: Fuzzy matching and normalization using `VariantLookup` from `term_normalization`

The goal is to catch format variations and map variants to canonical PharmGKB/ClinPGx identifiers.

## Pipeline

```
Article Text
    ↓
[1] SNP Expansion (pre-extraction)
    ↓ (CYP2B6 516G>T → CYP2B6 516G>T (rs3745274))
    ↓
[2] Regex Extraction
    ↓ (Pattern matching for rsIDs, star alleles, HLA)
    ↓
[3] Variant Normalization (NEW)
    ↓ (VariantLookup with fuzzy matching)
    ↓
[4] Apply Normalization
    ↓ (Use normalized term if score >= threshold)
    ↓
Final Variants + Mapping Trail
```

## Key Features

### 1. Post-Extraction Normalization

For each extracted variant:
- Query `VariantLookup.search(variant, threshold=0.8, top_k=3)`
- Get top candidates with similarity scores
- Apply normalization if best match score >= 0.9
- Track all mappings: `original → normalized`

### 2. Mapping Trail

Every variant transformation is recorded:

```json
{
  "normalization_mappings": {
    "hla-b*15:24": [
      {
        "id": "PA165954811",
        "normalized_term": "HLA-B*15:24",
        "url": "https://www.clinpgx.org/haplotype/PA165954811",
        "score": 1.0
      }
    ]
  },
  "normalization_applied": {
    "rs3745274": "rs3745274"  // Only if normalized form differs
  }
}
```

### 3. Impact Tracking

For each article, we track:
- Whether normalization was applied
- Whether normalization helped improve recall
- New matches found due to normalization
- Comparison with non-normalized extraction

## Usage

```bash
cd /path/to/autogkb-benchmark
python -m src.experiments.variant_finding.regex_term_norm.extract_variants_term_norm
```

### Parameters

You can adjust normalization behavior:

```python
run_experiment(
    normalization_threshold=0.8,  # Fuzzy matching threshold (0-1)
    normalization_min_score=0.9,  # Only apply if score >= this
    normalization_top_k=3,        # Number of candidates to retrieve
)
```

## Output

Results are saved to `results_term_norm.json`:

```json
{
  "run_name": "regex_term_norm_hybrid",
  "avg_recall": 0.934,
  "avg_precision": 0.419,
  "normalization_stats": {
    "total_variants_extracted": 2000,
    "total_normalizations_applied": 150,
    "normalization_rate": 0.075,
    "articles_where_normalization_helped": 5,
    "details": [...]
  },
  "per_article_results": [
    {
      "pmcid": "PMC5561238",
      "recall": 0.86,
      "normalization_mappings": {...},
      "normalization_applied": {...},
      "normalization_helped": true
    }
  ]
}
```

## Expected Benefits

### Format Variation Handling

- **HLA variants**: `B*5606` → `HLA-B*56:06`
- **Case variations**: `RS12345` → `rs12345`
- **Spacing differences**: `CYP 2D6*4` → `CYP2D6*4`

### Database Alignment

- Maps to canonical PharmGKB/ClinPGx IDs
- Validates against curated databases
- Provides confidence scores

### Fuzzy Matching

- Catches typos and OCR errors
- Handles synonyms and alternative representations
- Adjustable similarity threshold

## Performance Considerations

**API Calls**: The normalization step makes PharmGKB API calls for each variant, which can be slow.

**Optimization strategies**:
1. Batch processing: Group similar variants
2. Caching: Store normalization results
3. Local-first: Use local ClinPGx database before API
4. Selective normalization: Only normalize when confidence is low

## Potential Improvements

### V6+ Ideas

1. **Aggressive fuzzy matching**: Lower threshold to 0.7 for specific cases
2. **Synonym expansion**: Use PharmGKB synonym data
3. **Contextual normalization**: Use gene context for star alleles
4. **Error correction**: Detect and fix common OCR errors
5. **Batch API calls**: Reduce latency with batch lookups

## Related Files

- `extract_variants_term_norm.py`: Main implementation
- `results_term_norm.json`: Experiment results (generated)
- `../regex_variants/extract_variants_v5.py`: Base regex extraction
- `../../term_normalization/variant_search.py`: VariantLookup implementation

## Comparison with V5

| Feature | V5 | Term Norm Hybrid |
|---------|-----|------------------|
| Regex extraction | ✓ | ✓ |
| BioC supplements | ✓ | ✓ |
| SNP expansion | ✓ | ✓ |
| Format normalization | Basic | Advanced (fuzzy) |
| Database validation | No | Yes (PharmGKB/ClinPGx) |
| Mapping trail | No | Yes |
| Format variations | Limited | Comprehensive |

Expected recall improvement: +0.5-2% (targeting the 6 format-related misses in PMC5561238)
