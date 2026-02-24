# PGxMine Experiments - Implementation Summary

## ✅ Implementation Complete

All three PGxMine variant extraction experiments have been successfully implemented and integrated into the AutoGKB benchmark system.

---

## 📁 Files Created/Modified

### New Files Created

1. **`src/modules/variant_finding/pgxmine_normalization.py`** (320 lines)
   - Port of PGxMine's normalization function
   - 157 regex patterns for variant forms
   - Amino acid mappings (3-letter, full names, single-letter)

2. **`src/modules/variant_finding/methods/pgxmine_flow.py`** (370 lines)
   - Three extraction methods (context_aware, normalized, full)
   - PubTator integration with rate limiting
   - Context-aware star allele detection
   - Sentence-level filtering logic

3. **`PGXMINE_EXPERIMENTS.md`**
   - Comprehensive documentation
   - Usage instructions
   - Expected results analysis

4. **`test_pgxmine_implementation.py`**
   - Quick test script for single article

### Files Modified

1. **`src/modules/variant_finding/variant_extractor.py`**
   - Added imports for three new methods
   - Registered methods in METHODS dict

2. **`src/modules/variant_finding/run.py`**
   - Added three method names to CLI choices

---

## 🧪 Implemented Methods

### 1. `pgxmine_context_aware`

**Innovation:** Context-aware star allele detection

**How it works:**
- Uses PubTator to find Gene entities
- Applies star allele regex ONLY after gene mentions (50-char window)
- Extracts rsIDs globally
- **Research Question:** Does narrow context improve precision?

**Expected:** Higher precision, potential recall loss

### 2. `pgxmine_normalized`

**Innovation:** Comprehensive normalization

**How it works:**
- Broad variant extraction with regex
- Applies 157-pattern normalization to each candidate
- **Research Question:** Does normalization rescue messy extraction?

**Expected:** Higher recall, lower precision (improved by normalization)

### 3. `pgxmine_full`

**Innovation:** Complete PGxMine pipeline

**How it works:**
- Sentence-level filtering (Chemical AND Variant co-occurrence)
- Context-aware extraction on filtered sentences
- Normalization applied
- **Research Question:** How does full pipeline compare to baselines?

**Expected:** Balanced precision/recall

---

## 🚀 How to Run

### Quick Test (5 Articles)

```bash
# Test context-aware extraction
pixi run python -m src.modules.variant_finding.run \
    --method pgxmine_context_aware \
    --max-articles 5 \
    --eval

# Test normalized extraction
pixi run python -m src.modules.variant_finding.run \
    --method pgxmine_normalized \
    --max-articles 5 \
    --eval

# Test full pipeline
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

### Expected Runtime

- Each article: ~2-5 seconds (PubTator API rate limiting)
- 5 articles: ~15-30 seconds
- 32 articles: ~2-3 minutes per method
- Total (all 3 methods): ~6-9 minutes

---

## 📊 Output Files

### Variants

**Location:** `outputs/<method>_<timestamp>/variants.json`

Contains extracted variants for each article.

### Results

**Location:** `results/<method>_<timestamp>.json`

Contains evaluation metrics:
- Overall precision, recall, F1
- Per-article breakdown
- Matched/missed/extra variants

---

## 🎯 Success Criteria

### Implementation Checklist

- [x] `pgxmine_context_aware` method implemented
- [x] `pgxmine_normalized` method implemented
- [x] `pgxmine_full` method implemented
- [x] Normalization module ported (157 patterns)
- [x] PubTator integration with rate limiting
- [x] Methods registered in variant_extractor.py
- [x] CLI updated in run.py
- [x] All Python files pass syntax checks
- [x] Documentation created

### Evaluation Goals

After running experiments:

1. **Compare with regex_v5 baseline:**
   - regex_v5: 93.4% recall, 41.9% precision
   - Target: Match or improve recall, improve precision

2. **Analyze per-method performance:**
   - Which method has best precision?
   - Which method has best recall?
   - Which method has best F1 score?

3. **Identify variant type patterns:**
   - Which method works best for star alleles?
   - Which method works best for rsIDs?
   - Which method works best for HLA alleles?

4. **Error analysis:**
   - Categorize false positives
   - Categorize false negatives
   - Identify improvement opportunities

---

## 🔍 Key Implementation Details

### Context-Aware Extraction

- **Window size:** 50 characters after gene mention
- **Regex:** `^(,|and|or|/|\s|\+)*(?P<main>\*\s*[0-9]([\w:]*\w+)?)`
- **Source:** PGxMine's `findPGxSentences.py:33`

### Normalization Patterns

- **Star alleles:** Space removal only
- **rsIDs:** Space removal only
- **Protein variants:** 90+ patterns
  - `THR790MET` → `p.T790M`
  - `THREONINE to METHIONINE at position 790` → `p.T790M`
- **DNA variants:** 40+ patterns
  - `93G->A` → `c.93G>A`
  - `G to A substitution at nucleotide 93` → `c.93G>A`

### Sentence Filtering

- **Requirement:** Chemical entity AND (Gene OR Mutation) in same sentence
- **Purpose:** Focus on pharmacogenomic associations
- **Trade-off:** Higher precision, lower recall

### PubTator Integration

- **API:** NCBI PubTator3 BioC JSON endpoint
- **Rate limit:** 0.35s between requests
- **Entities extracted:** Gene, Chemical, Mutation, SNP, DNAMutation, ProteinMutation

---

## 📖 Documentation

See **`PGXMINE_EXPERIMENTS.md`** for:
- Detailed methodology descriptions
- Expected insights per experiment
- Error analysis guidelines
- Troubleshooting tips
- Comparison with baselines

---

## 🧪 Verification

All Python files have been verified:

```
✓ src/modules/variant_finding/pgxmine_normalization.py - syntax OK
✓ src/modules/variant_finding/methods/pgxmine_flow.py - syntax OK
✓ src/modules/variant_finding/variant_extractor.py - syntax OK
✓ src/modules/variant_finding/run.py - syntax OK
```

---

## 🔬 Next Steps

1. **Run experiments on 5-article subset:**
   ```bash
   for method in pgxmine_context_aware pgxmine_normalized pgxmine_full; do
       pixi run python -m src.modules.variant_finding.run \
           --method $method \
           --max-articles 5 \
           --eval
   done
   ```

2. **Review initial results:**
   - Check `results/<method>_<timestamp>.json`
   - Verify metrics are reasonable
   - Inspect per-article performance

3. **Run full benchmark:**
   ```bash
   for method in pgxmine_context_aware pgxmine_normalized pgxmine_full; do
       pixi run python -m src.modules.variant_finding.run \
           --method $method \
           --eval
   done
   ```

4. **Analyze results:**
   - Compare precision/recall across methods
   - Identify best-performing method
   - Categorize errors by variant type
   - Document findings in MEMORY.md

5. **Generate comparison table:**
   ```
   | Method                  | Recall | Precision | F1   |
   |-------------------------|--------|-----------|------|
   | regex_v5 (baseline)     | 93.4%  | 41.9%     | 57.8%|
   | pgxmine_context_aware   | ?      | ?         | ?    |
   | pgxmine_normalized      | ?      | ?         | ?    |
   | pgxmine_full            | ?      | ?         | ?    |
   ```

---

## 💡 Key Insights

### Design Decisions

1. **50-character context window:**
   - Based on PGxMine's iterative search approach
   - Balances precision (narrow context) vs recall (finding alleles)

2. **157 normalization patterns:**
   - Direct port from PGxMine's production code
   - Covers informal notations common in literature
   - Example: "THR790MET" → "p.T790M"

3. **Sentence-level filtering:**
   - Requires both Chemical and Variant entities
   - Focuses on pharmacogenomic associations (not just mentions)
   - Trade-off: precision vs recall

### Expected Trade-offs

- **Context-aware:** ⬆️ Precision, ⬇️ Recall (if alleles far from genes)
- **Normalized:** ⬆️ Recall, ⬇️ Precision (broad extraction + normalization)
- **Full pipeline:** ⚖️ Balanced (filtering + context + normalization)

---

## ✨ Innovation Summary

This implementation tests three core PGxMine innovations:

1. **Context-aware detection** - Apply extraction only near relevant entities
2. **Comprehensive normalization** - 157 patterns to handle variant notation diversity
3. **Co-occurrence filtering** - Focus on sentences with both drug and variant mentions

Each method isolates one innovation to measure its individual contribution to performance.

---

## 🎉 Ready to Run!

The implementation is complete and ready for testing. All methods are registered, documented, and syntax-verified. You can now run the experiments and compare PGxMine's methodology against the existing AutoGKB baseline methods.
