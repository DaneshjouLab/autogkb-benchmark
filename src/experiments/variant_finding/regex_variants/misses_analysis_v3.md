# Regex Extraction V3 - Missed Variants Analysis

**Overall Performance:** 87.8% recall, 42.8% precision across 32 articles

This document analyzes the **missed variants** (false negatives) from the regex extraction v3 run.

---

## Summary of Miss Categories

| Category | Count | Example |
|----------|-------|---------|
| SNP notation instead of rs number | 3 | `516G>T` instead of `rs3745274` |
| Star alleles in phenotype-only articles | 12 | Article discusses "poor metabolizer" but benchmark expects `*4/*4` |
| Metabolizer status as variant | 2 | `CYP2C19 intermediate metabolizer` |
| Wildtype alleles (`*1`) | 5 | `UGT1A1*1`, `CYP2D6*1` |
| Copy number variants | 1 | `CYP2D6*4xN` |
| HLA variants | 4 | `HLA-B*35:10` |

---

## Detailed Miss Analysis by Article

### PMC4916189
**Missed:** `cyp2b6*9`, `cyp2b6*1`

**Location in article:** The article uses SNP nomenclature throughout:
- "CYP2B6 516G>T" (corresponds to `*9`)
- "CYP2B6 983T>C"
- "CYP2B6 15582C>T"

The text never uses star allele notation—only positional SNP notation like "516G>T".

**Recommendation:** Create a mapping table of common SNP notations to star alleles:
- `CYP2B6 516G>T` → `CYP2B6*9`
- `CYP2B6 983T>C` → `CYP2B6*18`

---

### PMC2859392
**Missed:** `rs3745274`

**Location in article:** The article title and text use "CYP2B6 G516T" or "CYP2B6-G516T" notation rather than the rs number.

**Recommendation:** Add SNP-to-rsID mapping:
- `G516T` (CYP2B6) → `rs3745274`

---

### PMC4706412
**Missed:** `rs9923231`

**Location in article:** The article uses "VKORC1-1639 G>A" notation instead of the rs number. This is a common alternative nomenclature for this VKORC1 variant.

**Recommendation:** Add mapping:
- `VKORC1 -1639G>A` or `VKORC1-1639 G>A` → `rs9923231`

---

### PMC5561238
**Missed:** `hla-b*35:10`, `rs3745274`, `rs28399499`, `hla-drb1*10:01`, `hla-b*56:06`, `hla-drb1*08:01`

**Analysis:**
- 4 HLA variants were missed—these follow standard HLA nomenclature so they should have been caught
- 2 rs numbers were missed—likely appear in alternative notation in the text

**Recommendation:**
- Investigate why these specific HLA alleles weren't matched (possible OCR/formatting issues)
- Check for variant notation alternatives

---

### PMC10946077 & PMC11062152
**Missed:** `ugt1a1*1` (in both articles)

**Analysis:** The wildtype `*1` allele is often implied rather than explicitly stated. Articles may say "wild-type" or "normal function" instead of `*1`.

**Location:** Articles discuss UGT1A1*28 and *6 explicitly but reference wildtype as comparison without using `*1` notation.

**Recommendation:**
- Search for "wild-type", "wildtype", "WT", "normal function" in context of gene names
- Consider if `*1` should always be inferred when other star alleles are present

---

### PMC10880264
**Missed:** `cyp2c19 intermediate metabolizer`, `cyp2d6 poor metabolizer`

**Location in article:**
- Line 158: "intermediate metabolizer (IM n = 17)"
- Line 174: Table header "PM" (poor metabolizer), "IM" (intermediate metabolizer)

**Analysis:** The benchmark expects **metabolizer phenotype status** as the variant, not specific alleles. This is a fundamentally different annotation type that regex cannot capture without understanding the semantic meaning.

**Recommendation:** This requires a separate extraction approach:
- Pattern: `(CYP\w+)\s*(poor|intermediate|normal|rapid|ultrarapid)\s*metabolizer`
- Or: Look for metabolizer abbreviations (PM, IM, NM, RM, UM) in context of gene names

---

### PMC6435416
**Missed:** `cyp2d6*17`, `cyp2d6*35`, `cyp2d6*41`, `cyp2d6*3`, `cyp2d6*4`, `cyp2d6*29`, `cyp2d6*6`, `cyp2d6*9`, `cyp2d6*5`, `cyp2d6*4xn`

**Location in article:** The article discusses CYP2D6 **metabolizer phenotypes** (poor, intermediate, normal, ultrarapid) but does NOT list individual star alleles in the main text. The benchmark annotation includes all alleles that comprise these phenotypes (from supplementary materials or PharmGKB's interpretation).

**Analysis:** This is a mismatch between:
- What the article text contains: phenotype classifications
- What the benchmark expects: underlying genotypes

**Recommendation:**
- Accept that regex extraction from article text cannot capture alleles not mentioned
- Consider flagging articles that use phenotype-only language for manual review
- Or: add metabolizer-to-allele expansion as post-processing

---

### PMC3548984
**Missed:** `cyp2d6*1`

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

**Recommendation:** Same as UGT1A1*1—consider wildtype inference.

---

### PMC10399933
**Missed:** `cyp2c9*1`

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

**Recommendation:** Same as above.

---

## Recommended Improvements for V4

### 1. SNP Notation Mapping
Add a dictionary mapping common positional SNP notations to rs numbers and star alleles:

```python
SNP_MAPPINGS = {
    # CYP2B6
    ("CYP2B6", "516G>T"): ["rs3745274", "CYP2B6*9"],
    ("CYP2B6", "G516T"): ["rs3745274", "CYP2B6*9"],
    ("CYP2B6", "983T>C"): ["rs28399499", "CYP2B6*18"],

    # VKORC1
    ("VKORC1", "-1639G>A"): ["rs9923231"],
    ("VKORC1", "1639G>A"): ["rs9923231"],
}
```

### 2. Metabolizer Status Extraction
Add patterns to capture metabolizer phenotypes as variants:

```python
METABOLIZER_PATTERN = r"(CYP\w+)\s+(poor|intermediate|normal|rapid|ultrarapid)\s+metabolizer"
```

### 3. Wildtype Inference (Optional)
When star alleles like `*2`, `*3`, etc. are found for a gene, consider automatically adding `*1` if:
- The article discusses genotype comparisons
- Terms like "wild-type", "reference", or "normal function" appear

### 4. Alternative Notation Patterns
Expand patterns to catch:
- `516G>T`, `G516T`, `516 G>T` (various spacings)
- `-1639G>A`, `1639G>A`, `-1639 G>A`
- Gene names with various separators: `CYP2B6-G516T`, `CYP2B6 G516T`, `CYP2B6(G516T)`

---

## Impact Analysis

If all recommended improvements were implemented:

| Improvement | Variants Recovered | Est. Recall Gain |
|-------------|-------------------|------------------|
| SNP notation mapping | ~5 | +1.5% |
| Metabolizer status extraction | ~2 | +0.6% |
| HLA format fixes | ~4 | +1.2% |
| Wildtype inference | ~5 | +1.5% |

**Estimated new recall: ~93%** (up from 87.8%)

---

## Notes

- Some misses are due to fundamental differences between article text and benchmark expectations (e.g., phenotypes vs genotypes)
- Precision would remain low due to over-extraction—consider adding a relevance filter
- Articles using only metabolizer phenotypes may require LLM-based extraction for accurate allele assignment
