# Regex Extraction V5 - Missed Variants Analysis

**Overall Performance:** 93.4% recall, 41.9% precision across 32 articles

This document analyzes the **missed variants** (false negatives) from the regex extraction v5 run.

**V5 Improvements over V4:**
- BioC API integration for supplementary materials
- Extracts variants from article + supplement text combined
- 11 articles have BioC supplement data available

---

## Performance Comparison

| Version | Recall | Precision | Perfect Recall Articles |
|---------|--------|-----------|------------------------|
| V3 | 87.8% | 42.8% | 22/32 (69%) |
| V4 | 91.3% | 43.4% | 24/32 (75%) |
| **V5** | **93.4%** | 41.9% | **25/32 (78%)** |

**Total improvement V3→V5: +5.6% recall**

---

## Summary of Miss Categories

| Category | Count | Example |
|----------|-------|---------|
| Wildtype alleles (`*1`) | 4 | `UGT1A1*1`, `CYP2D6*1`, `CYP2C9*1` |
| Metabolizer status as variant | 2 | `CYP2C19 intermediate metabolizer` |
| HLA variants (format issues) | 4 | `HLA-B*35:10`, `HLA-DRB1*10:01` |
| rsIDs (format issues) | 2 | `rs28399499`, `rs3745274` in PMC5561238 |
| CYP2B6 star alleles | 2 | `CYP2B6*9`, `CYP2B6*1` |

**Total misses: 14 variants across 7 articles (of 32 total)**

---

## BioC Supplement Integration Results

### Impact

| Metric | Before (V4) | After (V5) | Change |
|--------|-------------|------------|--------|
| PMC6435416 recall | 33.3% | **100%** | **+66.7%** |
| Overall recall | 91.3% | 93.4% | +2.1% |
| Perfect recall articles | 24 | 25 | +1 |

### Variants Recovered from Supplements

**PMC6435416** - All 10 missing CYP2D6 star alleles recovered from Supplemental Table S2:
- `CYP2D6*3`, `CYP2D6*4`, `CYP2D6*4xN`, `CYP2D6*5`, `CYP2D6*6`
- `CYP2D6*9`, `CYP2D6*17`, `CYP2D6*29`, `CYP2D6*35`, `CYP2D6*41`

### BioC Coverage

- Articles with BioC supplements: 11/32 (34%)
- Articles helped by supplements: 1
- Total variants recovered: 10

---

## Detailed Miss Analysis by Article

### PMC4916189
**Missed:** `cyp2b6*9`, `cyp2b6*1`
**Recall:** 71% (5/7)

**Analysis:** The article uses SNP nomenclature (516G>T, 983T>C) which V5 successfully expands to rsIDs (`rs3745274`, `rs28399499`). However, the benchmark also expects the corresponding star alleles which aren't explicitly mentioned.

**Has BioC supplement:** Yes, but supplement doesn't contain these star alleles.

**Investigation - SNP-to-star mapping:**
The `SNPExpander` already has `star_allele` data, but there's a mismatch:
```
CYP2B6 516G>T → rs3745274 ✓ (correctly found)
CYP2B6 516G>T → CYP2B6*36 (PharmGKB says)
CYP2B6 516G>T → CYP2B6*9  (benchmark expects)
```

PharmGKB maps this SNP to `*36`, not `*9`. This is a **data source disagreement**, not a code issue.

**Recommendation:** Either accept the discrepancy or add manual curated mappings for known benchmark-specific expectations.

---

### PMC5561238
**Missed:** `hla-b*56:06`, `hla-drb1*10:01`, `rs28399499`, `hla-b*35:10`, `rs3745274`, `hla-drb1*08:01`
**Recall:** 86% (37/43)

**Analysis:** This article has 43 expected variants (the most in the benchmark). 37 were found successfully. The 6 misses appear to be:
- 4 HLA variants that may have formatting/OCR issues in the source
- 2 rsIDs that should have been found (possibly in a format not captured)

**Has BioC supplement:** Yes, but the missing variants aren't in the supplement text.

**Recommendation:** Investigate the specific text around these variants in the article to understand why they weren't captured.

---

### PMC10946077 & PMC11062152
**Missed:** `ugt1a1*1` (in both articles)
**Recall:** 67% (2/7 and 2/3)

**Analysis:** The wildtype `*1` allele is not explicitly mentioned. Articles discuss UGT1A1*28 and *6 explicitly but reference wildtype implicitly.

**Has BioC supplement:** No

**Recommendation:** Implement wildtype inference - when `*6`, `*28` are found, infer `*1` exists.

---

### PMC10880264
**Missed:** `cyp2d6 poor metabolizer`, `cyp2c19 intermediate metabolizer`
**Recall:** 33% (1/3)

**Analysis:** The benchmark expects **metabolizer phenotype status** as the variant. This is a semantic concept, not a genetic variant identifier.

**Has BioC supplement:** No

**Recommendation:** Add metabolizer status extraction:
```python
METABOLIZER_PATTERN = r"(CYP\w+)\s+(poor|intermediate|normal|rapid|ultrarapid)\s+metabolizer"
```

---

### PMC3548984
**Missed:** `cyp2d6*1`
**Recall:** 83% (5/10)

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

**Has BioC supplement:** No

**Recommendation:** Same as UGT1A1*1 - implement wildtype inference.

---

### PMC10399933
**Missed:** `cyp2c9*1`
**Recall:** 80% (4/5)

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

**Has BioC supplement:** No

**Recommendation:** Same as above - implement wildtype inference.

---

## Remaining Gaps Analysis

### Why These Can't Be Solved with More Data

| Miss Type | Count | Root Cause | Solution Type |
|-----------|-------|------------|---------------|
| Wildtype `*1` | 4 | Implicit in text | **Algorithmic** (inference) |
| Metabolizer phenotypes | 2 | Semantic, not variant | **Algorithmic** (pattern) |
| HLA/rsID format | 6 | Unknown format in source | **Investigation needed** |
| CYP2B6 star alleles | 2 | SNP→star mapping gap | **Algorithmic** (mapping) |

### Key Insight

**12 of 14 remaining misses (86%) require algorithmic solutions, not more data:**
- Wildtype inference: 4 variants
- Metabolizer pattern matching: 2 variants
- SNP-to-star-allele mapping: 2 variants
- Format investigation: 4 variants (may be data quality issues)

---

## Recommended Improvements for V6

### 1. Wildtype Inference (High Impact)
When star alleles like `*2`, `*3`, etc. are found for a gene, automatically add `*1`:

```python
def infer_wildtype(gene: str, found_alleles: list[str]) -> list[str]:
    """Add *1 if other alleles found and *1 not present."""
    if any(f"{gene}*" in a for a in found_alleles):
        if f"{gene}*1" not in [a.lower() for a in found_alleles]:
            return [f"{gene}*1"]
    return []
```

**Expected recovery:** 4 variants (+1.2% recall)

### 2. Metabolizer Status Extraction (Medium Impact)
Add pattern to capture metabolizer phenotypes:

```python
METABOLIZER_PATTERN = r"(CYP\w+)\s+(poor|intermediate|normal|rapid|ultrarapid)\s+metabolizer"
```

**Expected recovery:** 2 variants (+0.6% recall)

### 3. SNP-to-Star-Allele Mapping (Complicated)

The `SNPExpander` already has `star_allele` data from PharmGKB, but there's a mismatch:

```python
# What PharmGKB says:
expander.lookup("CYP2B6", "516G>T")
# → star_allele: "CYP2B6*36" (not *9!)

# What the benchmark expects:
# → "CYP2B6*9"
```

**The issue:** PharmGKB maps `516G>T` to `CYP2B6*36`, not `*9`. And `CYP2B6*9` isn't in PharmGKB's haplotype data at all.

**Options:**
1. Add manual curated mappings for known discrepancies
2. Accept that PharmGKB star allele assignments differ from benchmark
3. Use a different data source (e.g., PharmVar) for star allele definitions

**Expected recovery:** 0-2 variants (depends on data source alignment)

---

## Impact Analysis

If recommended improvements were implemented:

| Improvement | Variants Recovered | Est. Recall Gain | Complexity |
|-------------|-------------------|------------------|------------|
| Wildtype inference | 4 | +1.2% | Low |
| Metabolizer extraction | 2 | +0.6% | Low |
| SNP-to-star mapping | 0-2 | +0-0.6% | High (data mismatch) |
| **Total** | **6-8** | **+1.8-2.4%** | |

**Estimated V6 recall: ~95-96%** (up from 93.4%)

The remaining misses:
- 6 HLA/rsID in PMC5561238 - require investigation into source article
- 2 CYP2B6 star alleles - PharmGKB maps to different star alleles than benchmark expects

---

## Articles with Perfect Recall (25/32)

PMC5508045, PMC12036300, PMC554812, PMC6465603, PMC12038368, PMC12331468, **PMC6435416**, PMC12319246, PMC10275785, PMC11971672, PMC11430164, PMC8790808, PMC3839910, PMC3113609, PMC10786722, PMC384715, PMC3584248, PMC12035587, PMC10993165, PMC4706412, PMC6714829, PMC2859392, PMC11603346, PMC8973308, PMC3387531

**Notable:** PMC6435416 achieved perfect recall in V5 (was 33% in V4) thanks to BioC supplement integration.

---

## Conclusion

V5 successfully integrated BioC supplementary materials, improving recall from 91.3% to 93.4%. The major win was PMC6435416, which went from 33% to 100% recall by extracting CYP2D6 star alleles from Supplemental Table S2.

The remaining 14 misses fall into predictable categories that require algorithmic solutions rather than more data:
- **Wildtype inference** for implicit `*1` alleles
- **Metabolizer pattern matching** for phenotype terms
- **SNP-to-star mapping** for indirect variant references

The BioC API integration is complete and working. Future improvements should focus on the algorithmic enhancements outlined above.
