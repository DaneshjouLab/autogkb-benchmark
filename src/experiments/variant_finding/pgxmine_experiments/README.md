# PGxMine Variant Extraction Experiments

This folder contains all files related to the PGxMine variant extraction experiments conducted on the AutoGKB benchmark.

## Experiment Summary

**Date:** 2026-02-04
**Goal:** Test PGxMine's variant extraction methodology on AutoGKB benchmark
**Outcome:** All methods significantly underperformed the regex_v5 baseline

### Results at a Glance

| Method | Recall | Precision | F1 Score |
|--------|--------|-----------|----------|
| **regex_v5 (baseline)** | **93.4%** | **41.9%** | **57.8%** |
| pgxmine_context_aware | 39.1% | 23.4% | 29.3% |
| pgxmine_normalized | 45.3% | 8.8% | 14.9% |
| pgxmine_full | 19.7% | 17.2% | 18.4% |

**Key Finding:** 0 star alleles detected by any method (major failure mode)

---

## Folder Structure

```
pgxmine_experiments/
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md      # Implementation details & how to run
│   ├── PGXMINE_EXPERIMENTS.md         # Detailed methodology & expected results
│   └── PGXMINE_RESULTS_SUMMARY.md     # Complete results analysis
├── results/                           # Experimental results
│   ├── pgxmine_context_aware_*.json   # Context-aware method results
│   ├── pgxmine_normalized_*.json      # Normalized method results
│   ├── pgxmine_full_*.json            # Full pipeline results
│   └── pgxmine_*_*/                   # Output directories with variants
└── tests/                             # Test scripts
    └── test_pgxmine_implementation.py # Quick test on single article
```

---

## Source Code Location

The actual implementation code remains in the main codebase:

- **Normalization:** `src/modules/variant_finding/pgxmine_normalization.py`
- **Extraction methods:** `src/modules/variant_finding/methods/pgxmine_flow.py`
- **Method registration:** `src/modules/variant_finding/variant_extractor.py`
- **CLI:** `src/modules/variant_finding/run.py`

---

## Quick Links

### Documentation

1. **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)**
   - What was implemented
   - How to run the experiments
   - Expected outputs
   - Success criteria

2. **[PGXMINE_EXPERIMENTS.md](docs/PGXMINE_EXPERIMENTS.md)**
   - Detailed methodology for each method
   - Expected insights
   - Comparison with baselines
   - Troubleshooting guide

3. **[PGXMINE_RESULTS_SUMMARY.md](docs/PGXMINE_RESULTS_SUMMARY.md)**
   - Complete results analysis
   - Root cause analysis
   - Lessons learned
   - Recommendations

---

## Running the Experiments

### Quick Test (5 articles)

```bash
source .venv/bin/activate
PYTHONPATH=src python -m src.modules.variant_finding.run \
    --method pgxmine_context_aware \
    --max-articles 5 \
    --eval
```

### Full Benchmark (32 articles)

```bash
for method in pgxmine_context_aware pgxmine_normalized pgxmine_full; do
    PYTHONPATH=src python -m src.modules.variant_finding.run \
        --method $method \
        --eval
done
```

---

## Key Findings

### What Worked

- ✅ rsID extraction (basic regex)
- ✅ Some HLA allele detection (normalized method)
- ✅ Clean implementation (no bugs)

### What Failed

- ❌ Star allele detection (0 found across all methods)
- ❌ PubTator Mutation entities (missing in 28/32 articles)
- ❌ Context-aware extraction (window too narrow)
- ❌ Sentence filtering (too aggressive, 19.7% recall)
- ❌ Normalization (no benefit for already-standard variants)

### Root Causes

1. **Methodology mismatch:** PGxMine designed for association extraction, not variant mention extraction
2. **Entity dependency:** Relying on PubTator entities proved fragile
3. **Context limitations:** 50-char window insufficient for star alleles
4. **Over-filtering:** Chemical + Variant co-occurrence requirement too strict

---

## Recommendations

### For Future Work

1. **Don't use these methods** - regex_v5 is far superior (93.4% vs 19.7-45.3% recall)
2. **If improving PGxMine approaches:**
   - Fix star allele detection (gene-specific regex, wider context)
   - Remove sentence filtering
   - Use PubTator for validation, not extraction
3. **Key lesson:** Simple pattern matching > sophisticated NLP for this task

### For Similar Experiments

1. **Validate components first** - test simple baseline before complex pipeline
2. **Check entity coverage** - ensure NER tool detects target entity types
3. **Measure incrementally** - add complexity only if it improves metrics
4. **Match methodology to task** - PGxMine optimized for different problem

---

## Comparison with Baseline

### regex_v5 (Winner)

**Approach:**
- Direct gene-specific patterns: `CYP2D6\*(\d+)`
- No entity dependencies
- No sentence filtering
- No normalization

**Why it wins:**
- ✅ Finds star alleles reliably
- ✅ High recall (93.4%)
- ✅ Faster (no API calls)
- ✅ Robust (no entity dependencies)
- ✅ Debuggable (simple patterns)

### PGxMine Methods (Failed)

**Common issues:**
- ❌ 0 star alleles found
- ❌ Depends on unreliable entity detection
- ❌ Complex pipeline with multiple failure points
- ❌ Slower (PubTator API calls)

---

## Methodology Details

### Method 1: pgxmine_context_aware

**Concept:** Detect star alleles only after Gene entities (PGxMine's innovation)

**Implementation:**
1. Get Gene entities from PubTator
2. Apply star allele regex in 50-char window after each gene
3. Extract rsIDs globally

**Expected:** Higher precision (narrow context)
**Actual:** 39.1% recall, 23.4% precision (poor on both)

**Failure mode:** Star alleles not within 50 chars of genes

---

### Method 2: pgxmine_normalized

**Concept:** Broad extraction + comprehensive normalization (157 patterns)

**Implementation:**
1. Extract variants with broad regex
2. Apply PGxMine's normalization to each candidate
3. Return normalized variants

**Expected:** Higher recall (broad extraction)
**Actual:** 45.3% recall, 8.8% precision (many false positives)

**Failure mode:** Broad regex too noisy, normalization doesn't help standard variants

---

### Method 3: pgxmine_full

**Concept:** Complete PGxMine pipeline (co-occurrence filtering)

**Implementation:**
1. Split into sentences
2. Filter to sentences with Chemical AND (Gene OR Mutation)
3. Extract from filtered sentences
4. Apply normalization

**Expected:** Balanced precision/recall
**Actual:** 19.7% recall, 17.2% precision (worst performer)

**Failure mode:** Filtering too aggressive, Mutation entities missing

---

## Lessons Learned

1. **Entity-based methods are fragile** - pattern matching more reliable
2. **Context windows miss long-range references** - star alleles mentioned far from genes
3. **Sentence filtering loses recall** - valid mentions in non-drug sentences
4. **Normalization not always needed** - depends on input format
5. **Method-task alignment critical** - PGxMine optimized for different problem

---

## Files Reference

### Documentation Files

- **IMPLEMENTATION_SUMMARY.md** - Quick reference, how to run
- **PGXMINE_EXPERIMENTS.md** - Detailed methodology, expected insights
- **PGXMINE_RESULTS_SUMMARY.md** - Complete analysis, recommendations

### Results Files

- **pgxmine_context_aware_*.json** - Evaluation results (recall, precision, per-article)
- **pgxmine_normalized_*.json** - Evaluation results
- **pgxmine_full_*.json** - Evaluation results
- **pgxmine_*_*/variants.json** - Extracted variants for each article

### Test Files

- **test_pgxmine_implementation.py** - Quick test script for single article

---

## Citation

If referencing this experiment:

```
PGxMine Variant Extraction Experiments on AutoGKB Benchmark
Date: 2026-02-04
Methods: Context-aware, Normalized, Full pipeline
Baseline: regex_v5 (93.4% recall, 41.9% precision)
Result: All methods underperformed baseline (19.7-45.3% recall)
Key finding: Star allele detection failed (0 found)
Conclusion: Pattern matching superior to entity-based NLP for this task
```

---

## Contact

For questions about this experiment:
- See detailed analysis in `docs/PGXMINE_RESULTS_SUMMARY.md`
- Check implementation in `src/modules/variant_finding/methods/pgxmine_flow.py`
- Review methodology in `docs/PGXMINE_EXPERIMENTS.md`

---

**Experiment Status:** ✅ Complete
**Outcome:** ❌ Methods not viable for AutoGKB benchmark
**Recommendation:** Use regex_v5 baseline instead
