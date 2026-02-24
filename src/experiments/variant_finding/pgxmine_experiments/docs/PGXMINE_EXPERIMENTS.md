# PGxMine Variant Extraction Experiments

## Overview

This implementation adds three new variant extraction methods to the AutoGKB benchmark system, each testing a specific aspect of PGxMine's methodology.

## Implemented Methods

### 1. `pgxmine_context_aware`

**Tests:** Context-aware star allele detection (PGxMine's core innovation)

**Methodology:**
1. Fetch article text (markdown + BioC supplements)
2. Use PubTator API to identify Gene entities with positions
3. Apply PGxMine's star allele regex ONLY after gene mentions (50-char window)
   - Regex: `^(,|and|or|/|\s|\+)*(?P<main>\*\s*[0-9]([\w:]*\w+)?)`
   - Source: `pgxmine/findPGxSentences.py:33`
4. Extract rsIDs globally using `\brs\d{4,}\b`
5. Format star alleles as `GENE*ALLELE` (e.g., `CYP2D6*4`)

**Research Question:** Does narrow, gene-aware context improve precision vs. broad extraction?

**Expected Performance:**
- Higher precision (fewer false positives from random `*` characters)
- Potential recall loss if star alleles mentioned far from gene names

---

### 2. `pgxmine_normalized`

**Tests:** Impact of comprehensive normalization

**Methodology:**
1. Fetch article text
2. Extract variants with broad regex patterns:
   - Star alleles: `\*\s*[0-9][\w:]*` (anywhere in text)
   - rsIDs: `\brs\d{4,}\b`
   - HLA alleles: `(?:HLA-)?([ABC]|DRB[1345]|DQ[AB]1|DP[AB]1)\*\d{2,}:?\d{0,2}`
3. Apply PGxMine's `normalize_mutation()` to each candidate
   - 157 regex patterns for variant forms
   - Source: `pgxmine/utils/__init__.py:11-235`
4. Return normalized variants

**Research Question:** Does aggressive normalization compensate for messier extraction?

**Expected Performance:**
- Lower precision (broad extraction catches noise)
- Higher recall (captures variants in non-standard formats)
- Normalization may rescue some false positives into true positives

---

### 3. `pgxmine_full`

**Tests:** Complete PGxMine pipeline end-to-end

**Methodology:**
1. Fetch article text, split into sentences
2. Get PubTator annotations for Genes, Chemicals, Mutations
3. Filter to sentences containing BOTH Chemical AND (Mutation OR Gene)
   - This implements PGxMine's co-occurrence filtering
4. Extract star alleles (context-aware) + rsIDs from filtered sentences only
5. Apply normalization
6. Return unique variants

**Research Question:** How does the complete PGxMine pipeline compare to regex_v5 baseline (93.4% recall, 41.9% precision)?

**Expected Performance:**
- Moderate-to-high precision (sentence filtering removes noise)
- Lower recall (strict filtering may exclude valid mentions)
- Good balance for high-confidence extractions

---

## File Structure

```
src/modules/variant_finding/
├── pgxmine_normalization.py       # Normalization logic (157 patterns)
├── methods/
│   └── pgxmine_flow.py            # Three extraction methods
├── variant_extractor.py           # Method registration (updated)
└── run.py                         # CLI choices (updated)
```

### Key Components

**pgxmine_normalization.py:**
- `normalize_mutation(mention: str) -> str | None`
- Amino acid mappings (3-letter, full names, single-letter)
- 157 regex patterns for:
  - Star alleles (`*4`, `* 4`)
  - rsIDs (`rs9923231`, `rs 9923231`)
  - Protein variants (`p.T790M`, `THR790MET`, `THREONINE 790 to METHIONINE`)
  - DNA/cDNA variants (`c.93G>A`, `93G->A`, `g.93delG`)
  - Frameshifts (`T790fs`, `p.T790fsX791`)

**pgxmine_flow.py:**
- `_fetch_pubtator_annotations()` - Rate-limited PubTator API calls
- `_extract_entities_from_biocjson()` - Parse Gene/Chemical/Mutation entities
- `_split_into_sentences()` - Sentence segmentation with offsets
- `_filter_sentences_with_chem_variant()` - Co-occurrence filtering
- `_extract_star_alleles_after_genes()` - Context-aware detection
- `pgxmine_context_aware_extract()` - Experiment 1
- `pgxmine_normalized_extract()` - Experiment 2
- `pgxmine_full_extract()` - Experiment 3

---

## Running Experiments

### Test on Subset (5 Articles)

```bash
# Context-aware extraction
pixi run python -m src.modules.variant_finding.run \
    --method pgxmine_context_aware \
    --max-articles 5 \
    --eval

# Normalized extraction
pixi run python -m src.modules.variant_finding.run \
    --method pgxmine_normalized \
    --max-articles 5 \
    --eval

# Full pipeline
pixi run python -m src.modules.variant_finding.run \
    --method pgxmine_full \
    --max-articles 5 \
    --eval
```

### Full Benchmark (32 Articles)

```bash
# Run all three experiments
for method in pgxmine_context_aware pgxmine_normalized pgxmine_full; do
    pixi run python -m src.modules.variant_finding.run \
        --method $method \
        --eval
done
```

### Single Article Test

```bash
# Manually test on PMC5508045 (4 rsID variants)
pixi run python test_pgxmine_implementation.py
```

---

## Output Files

### Variants Output

**Location:** `outputs/<method>_<timestamp>/variants.json`

**Format:**
```json
{
  "metadata": {
    "method": "pgxmine_context_aware",
    "timestamp": "2025-02-04T10:30:00",
    "num_articles": 32
  },
  "variants": {
    "PMC5508045": ["rs9923231", "rs887829", "rs2108622", "rs1057910"],
    "PMC4916189": ["CYP2B6*1", "CYP2B6*9", "rs3745274", ...]
  }
}
```

### Results Output

**Location:** `results/<method>_<timestamp>.json`

**Format:**
```json
{
  "method": "pgxmine_context_aware",
  "overall": {
    "precision": 0.55,
    "recall": 0.85,
    "f1": 0.67,
    "perfect_recall_count": 15
  },
  "per_article": {
    "PMC5508045": {
      "ground_truth": ["rs9923231", "rs887829", "rs2108622", "rs1057910"],
      "extracted": ["rs9923231", "rs887829", "rs2108622", "rs1057910"],
      "matches": 4,
      "misses": 0,
      "extras": 0,
      "precision": 1.0,
      "recall": 1.0
    }
  }
}
```

---

## Evaluation Metrics

1. **Recall:** `matches / ground_truth_count`
   - % of ground truth variants found

2. **Precision:** `matches / extracted_count`
   - % of extracted variants that are correct

3. **F1 Score:** `2 * (precision * recall) / (precision + recall)`
   - Harmonic mean of precision and recall

4. **Perfect Recall Count:** Number of articles with 100% recall

---

## Comparison Baseline

Compare against existing methods:

| Method | Recall | Precision | F1 | Perfect Recall |
|--------|--------|-----------|-----|----------------|
| **regex_v5** | 93.4% | 41.9% | 57.8% | 24/32 |
| pubtator | 36.3% | 23.4% | 28.5% | 6/32 |
| just_ask (Claude) | 72.0% | 45.7% | 56.0% | 14/32 |
| just_ask (GPT-4o) | 66.1% | 42.4% | 51.7% | 11/32 |

**Target:** Beat regex_v5's recall while improving precision

---

## Expected Insights

### 1. Context-Awareness Impact

**Question:** Does detecting star alleles only after genes reduce false positives?

**Metrics to Check:**
- Precision vs. regex_v5
- False positive analysis (extracted but not in ground truth)
- Missed variants that appear far from gene names

**Example Case:**
- Text: "...CYP2D6 is important. Patients with *4 or *10..."
- Context-aware: ✓ Detects `CYP2D6*4`, `CYP2D6*10`
- Broad regex: ✓ Detects but might miss gene association

### 2. Normalization Value

**Question:** Which of the 157 patterns most frequently improve matches?

**Metrics to Check:**
- Recall improvement from normalization
- Most useful pattern categories (protein vs DNA vs star alleles)
- Cases where normalization rescues matches

**Example Cases:**
- `THR790MET` → `p.T790M` (3-letter to standard)
- `93G->A` → `c.93G>A` (informal to HGVS)
- `* 4` → `*4` (space removal)

### 3. Full Pipeline Performance

**Question:** Is sentence-level filtering worth the recall cost?

**Metrics to Check:**
- Precision vs. other PGxMine methods
- Recall loss from filtering
- Types of variants lost (mention-only vs. association)

**Example Case:**
- Sentence: "CYP2D6*4 increases warfarin sensitivity"
  - Has Chemical (warfarin) ✓
  - Has Gene (CYP2D6) ✓
  - Kept by filter ✓
- Sentence: "The CYP2D6*4 allele is common"
  - Has Gene (CYP2D6) ✓
  - No Chemical ✗
  - Filtered out ✗

---

## Error Analysis

### Expected False Positives

1. **Non-variant asterisks:**
   - Mathematical notation: "p < 0.05*"
   - Footnote markers: "*significant"
   - Mitigated by: context-awareness

2. **Protein mentions without mutations:**
   - "p53 protein levels"
   - Mitigated by: normalization patterns

3. **HLA typing context:**
   - "HLA typing was performed..."
   - Mitigated by: sentence filtering

### Expected False Negatives

1. **Star alleles far from genes:**
   - "CYP2D6 genotyping... The *4 allele frequency..."
   - Lost by: context window limits

2. **Non-pharmacogenomic variants:**
   - Cancer mutations not in PGx genes
   - Intentionally excluded

3. **Informal notations:**
   - "2D6-4" instead of "CYP2D6*4"
   - Normalization may not cover all forms

---

## Next Steps

1. **Run Experiments:**
   - Test on 5-article subset first
   - Verify outputs are sensible
   - Run full 32-article benchmark

2. **Analyze Results:**
   - Compare precision/recall with baselines
   - Identify variant types where each method excels
   - Analyze per-article performance patterns

3. **Error Analysis:**
   - Categorize false positives by type
   - Categorize false negatives by type
   - Identify areas for improvement

4. **Method Refinement:**
   - Adjust context window size if needed
   - Add missing normalization patterns
   - Tune sentence filtering criteria

5. **Documentation:**
   - Update MEMORY.md with key findings
   - Document which method works best for which variant types
   - Record optimal parameters

---

## Implementation Notes

### Dependencies

All required packages are in `pixi.toml`:
- `requests` - PubTator API calls
- `loguru` - Logging
- `re` - Regex operations (standard library)

### Rate Limiting

PubTator API calls are rate-limited to 0.35s between requests (enforced in `_fetch_pubtator_annotations()`).

### Text Sources

Combines two sources for comprehensive coverage:
1. Article markdown (from `src.utils.get_markdown_text()`)
2. BioC supplement (from `src.modules.utils_bioc.fetch_bioc_supplement()`)

### Entity Tracking

All entity positions are tracked relative to the full document text, enabling:
- Mapping entities to sentences
- Context-aware extraction windows
- Offset-based filtering

### Normalization Edge Cases

- Star alleles and rsIDs: spaces removed, passed through
- Unknown patterns: returns None (variant kept as-is)
- Amino acid codes: case-insensitive matching

---

## References

- **PGxMine Repository:** https://github.com/jakelever/pgxmine
- **PGxMine Data:** https://zenodo.org/records/6617348
- **PubTator3 API:** https://www.ncbi.nlm.nih.gov/research/pubtator3-api/
- **HGVS Nomenclature:** https://varnomen.hgvs.org/

---

## Troubleshooting

### Rosetta Error on macOS

If you see `rosetta error: Attachment of code signature supplement failed`, this is a macOS-specific issue with Conda packages. The code itself is correct. Try:

```bash
# Use native Python if available
python3 test_pgxmine_implementation.py

# Or create a fresh environment
conda create -n pgxmine python=3.11
conda activate pgxmine
pip install -r <requirements>
```

### PubTator API Timeout

If PubTator API calls timeout:
1. Check network connectivity
2. Verify PMID exists in mapping
3. API may be temporarily down (retry later)

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Ensure dependencies are installed
pixi install

# Or check Python path
PYTHONPATH=src pixi run python ...
```

---

## Success Criteria

✅ **Implementation Complete:**
- [x] Three methods implemented
- [x] Methods registered in variant_extractor.py
- [x] CLI choices updated in run.py
- [x] Normalization module ported (157 patterns)
- [x] PubTator integration with rate limiting
- [x] Context-aware star allele detection
- [x] Sentence-level filtering

🎯 **Evaluation Goals:**
- [ ] All three methods run successfully on 32 articles
- [ ] Results saved in standard format
- [ ] Precision/recall calculated
- [ ] Comparison with regex_v5 baseline
- [ ] Per-article analysis completed
- [ ] Error patterns identified and categorized

📊 **Target Metrics:**
- Recall ≥ 90% (match or beat regex_v5's 93.4%)
- Precision > 50% (improve on regex_v5's 41.9%)
- F1 Score > 60%
- At least one method finds a good precision/recall balance
