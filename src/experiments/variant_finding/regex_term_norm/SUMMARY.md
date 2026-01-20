# Regex + Term Normalization Hybrid - Executive Summary

## Question
**Do existing regex finding systems benefit from term_normalization packages?**

## Answer
**Yes, but selectively.** Term normalization helps in two ways:

1. ✅ **Pre-extraction normalization** (SNP expansion) - Already used in v4/v5, significantly helps
2. ❌ **Post-extraction normalization** (fuzzy matching) - Doesn't help recall, but provides data quality benefits

## What We Built

Created a hybrid system combining:
- V5 regex extraction (with BioC supplements + SNP expansion)
- Post-extraction normalization using `VariantLookup` from `term_normalization`
- Complete mapping trail tracking all transformations

## Results

### Performance (Compared to V5)
- **Recall**: 93.4% (same as V5, no improvement)
- **Precision**: 41.9% (same as V5)
- **Perfect recall articles**: 25/32 (78%, same as V5)

### Normalization Activity
- **17 variants normalized** across 7 articles
- **3.2% normalization rate**
- **0 articles** had improved recall

### Runtime
- **~19 minutes** (vs ~2 minutes for V5)
- Slowdown due to PharmGKB API calls for each variant

## Why Didn't Post-Extraction Normalization Help Recall?

### Finding 1: Typos in Benchmark
Articles with normalizations applied **already had 100% recall**:

| Article | Normalization | Original Match? | After Normalization |
|---------|---------------|-----------------|---------------------|
| PMC12331468 | `rs180131` → `rs1801131` | ✓ (typo in benchmark) | Still matches |
| PMC6435416 | `rs1694` → `rs16944` | ✓ (typo in benchmark) | Still matches |
| PMC12319246 | `rs3745275` → `rs374527` | ✓ (typo in benchmark) | Still matches |

**Conclusion**: Normalizing typos doesn't help when the benchmark contains the same typos.

### Finding 2: Missed Variants Never Extracted
All 7 articles with imperfect recall had **zero normalizations** because missed variants were never extracted:

| Article | Recall | Missed Variants | Extracted? |
|---------|--------|-----------------|------------|
| PMC5561238 | 86% | `hla-drb1*10:01`, `rs28399499` | No (not in text) |
| PMC10946077 | 67% | `ugt1a1*1` | No (wildtype inference needed) |
| PMC10880264 | 33% | `cyp2d6 poor metabolizer` | No (phenotype extraction needed) |
| PMC4916189 | 71% | `cyp2b6*9` | No (SNP→star mapping gap) |

**Conclusion**: Fuzzy matching can't recover variants that regex never extracted.

## What Post-Extraction Normalization DID Accomplish

Even though it didn't improve recall, it provided valuable data quality improvements:

### 1. Typo Correction
```
rs180131 → rs1801131 (missing digit)
rs6737679 → rs67376798 (wrong digits)
rs1694 → rs16944 (missing digit)
```

### 2. Format Standardization
```
DPYD*9 → rs1801265 (star allele to rsID)
CYP2B6*516 → rs28399499 (incorrect notation)
```

### 3. Database Validation
- Validated all 536 extracted variants against PharmGKB/ClinPGx
- Provided PharmGKB IDs and URLs for each variant
- Confirmed variants exist in pharmacogenomics databases

### 4. Transparency
- Complete mapping trail saved for every variant
- Top 3 candidates with similarity scores
- Clear record of which normalizations were applied

## The Two Types of Term Normalization

| Type | When Applied | V5 Usage | Helps Recall? | Use Case |
|------|--------------|----------|---------------|----------|
| **Pre-extraction** | Before pattern matching | ✓ (SNP expansion) | Yes | Enriches text with alternative forms |
| **Post-extraction** | After pattern matching | ✗ (not used) | No | Validates and standardizes extracted variants |

### Pre-Extraction Normalization (Already in V5)
```
Article: "CYP2B6 516G>T"
    ↓ SNP expansion
Article: "CYP2B6 516G>T (rs3745274)"  ← Enriched text
    ↓ Regex extraction
Extracted: ["rs3745274"]  ← New variant found!
```
**Result**: Recovers variants mentioned only as SNP notations

### Post-Extraction Normalization (This experiment)
```
Article: "rs3745275" (typo)
    ↓ Regex extraction
Extracted: ["rs3745275"]  ← Matches benchmark typo
    ↓ Normalization
Normalized: ["rs3745274"]  ← Breaks match!
```
**Result**: Doesn't help when benchmark has typos, can't recover missed variants

## Recommendations

### ✅ Keep Using
1. **Pre-extraction normalization** (SNP expansion) - Already proven valuable in v4/v5
2. **Format normalization** (during extraction) - Basic cleanup like HLA format fixes
3. **Post-extraction for production** - Useful for data quality even if not for recall

### ❌ Don't Expect
1. **Recall improvement from fuzzy matching** - Won't recover missed variants
2. **Help with algorithmic gaps** - Need wildtype inference, phenotype extraction, etc.

### 🎯 Focus On Instead
To improve beyond 93.4% recall, implement **V6 algorithmic improvements**:

| Improvement | Variants Targeted | Expected Gain |
|-------------|-------------------|---------------|
| Wildtype inference | 4 | +1.2% |
| Metabolizer phenotypes | 2 | +0.6% |
| Better table extraction | 6+ | +1.8% |
| SNP→star mapping (PharmVar) | 2 | +0.6% |

## Files Created

```
regex_term_norm/
├── extract_variants_term_norm.py     # Main hybrid implementation
├── README.md                          # Usage documentation
├── results_term_norm.json            # Complete results with mappings
├── results_analysis.md               # Detailed analysis
└── SUMMARY.md                        # This file
```

## Key Takeaway

**Term normalization is valuable, but in different ways:**

- ✅ **SNP expansion** (pre-extraction): Recovers variants → Used in v4/v5 → Helps recall
- ✅ **Database validation** (post-extraction): Ensures quality → This experiment → Good for production
- ❌ **Fuzzy matching** (post-extraction): Can't recover missed variants → This experiment → Doesn't help recall

**The right tool for the right job.** Post-extraction term normalization should be used for **data quality and validation**, not as a strategy to improve recall. To improve recall, focus on better **extraction algorithms** that capture more variants upfront.

## Next Steps

1. **Keep term_norm hybrid for production** - Provides valuable validation and standardization
2. **Implement V6 improvements** - Wildtype inference, metabolizer patterns, better table extraction
3. **Consider alternative normalization strategies**:
   - PharmVar for comprehensive SNP↔star mappings
   - Context-aware star allele disambiguation
   - Table-specific extraction strategies
