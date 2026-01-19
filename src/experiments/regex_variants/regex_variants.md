# Regex Variants Experiment Log

Started: 2025-01-16 7:21 PST

## Overview
Trying to extract variants from articles using regex patterns and seeing if that at least gets full coverage of the variants. This is a bit more deterministic than LLM-based extraction which is nice and will be much faster/cheaper.

## Process
1. Get the markdown for the article but only the methods and conclusions sections
2. Extract variants using regex patterns
3. Evaluate coverage against ground truth using benchmark_v2/variant_bench.py
4. Iterate on regex patterns to improve coverage until it doesn't make sense to continue (high accuracy or an absurd number of patterns needed)

---

## Results Summary

| Version | Avg Recall | Avg Precision | Perfect Recall | Key Changes |
|---------|------------|---------------|----------------|-------------|
| v1      | 42.0%      | 31.4%         | -              | Basic patterns (rsIDs, CYP star alleles, HLA with prefix) |
| v2      | 69.9%      | 37.5%         | -              | Full text, HLA without prefix, broader gene list |
| v3      | 87.8%      | 42.8%         | 22/32 (69%)    | Standalone star alleles, diplotypes, NUDT15, HLA parenthetical |

---

## Version 1 (v1)
- **Recall: 42.0%**, Precision: 31.4%
- Used only methods/conclusions sections
- Basic patterns:
  - rsIDs: `rs\d{4,}`
  - Star alleles: `CYP\w+\*\d+`
  - HLA: `HLA-[A-Z]+\d*\*\d+:\d+`
- Issues: Many variants missed due to limited text, missing gene families

## Version 2 (v2)
- **Recall: 69.9%**, Precision: 37.5%
- Improvements:
  - Uses full article text instead of just methods/conclusions
  - Handles HLA alleles without HLA- prefix (e.g., B*5801 → HLA-B*58:01)
  - Broader star allele gene patterns (UGT, NUDT, DPYD, TPMT, NAT, SLCO, ABCB)
  - Format normalization
- Issues: Missing standalone star alleles, NUDT15, copy number variants

## Version 3 (v3) - Current Best
- **Recall: 87.8%**, Precision: 42.8%
- Perfect recall on 22/32 articles (69%)
- Improvements:
  - Handles star alleles with space between gene and allele (e.g., "NUDT15 *3")
  - Handles standalone star alleles (*3, *4) by finding nearby gene context (200 char window)
  - Handles diplotype patterns with Unicode multiplication sign (*1×N/*2, *1/*10×N) with 800 char window
  - Added NUDT15 to gene list
  - Handles copy number variants (*1xN, *2xN)
  - Expanded pharmacogene list to 25+ genes
  - HLA parenthetical notation: HLA-B*38:(01/02), B*39:(01/05/06/09)

### Articles with Remaining Issues

1. **PMC2859392 (0%)** - Uses G516T notation instead of rs3745274
2. **PMC6435416 (33%)** - CYP2D6 star alleles only in supplemental tables
3. **PMC10880264 (33%)** - Metabolizer phenotypes ("CYP2D6 poor metabolizer") not extractable
4. **PMC4706412 (88%)** - Uses -1639G>A notation instead of rs9923231
5. **PMC5561238 (86%)** - Some HLA alleles still in non-standard format
6. **UGT1A1*1** - Wild-type allele often not explicitly mentioned

### Limitations Identified

1. **Variant notation mapping**: Articles use different notations for the same variant:
   - rsID: rs3745274
   - cDNA change: G516T, c.516G>T, 516G>T
   - Position notation: -1639G>A, VKORC1-1639
   - Would require a variant-to-rsID mapping database

2. **Supplemental tables**: Variants listed only in supplemental materials are not accessible

3. **Metabolizer phenotypes**: Ground truth includes "CYP2D6 poor metabolizer" which are phenotype descriptions, not variant identifiers

4. **Wild-type alleles**: Reference alleles like *1 are often not explicitly mentioned in text

---

## Next Steps (if needed)

1. Add cDNA change pattern matching (G516T, -1639G>A) with variant-to-rsID mapping
2. Consider pharmacogene-specific patterns for wild-type detection
3. Explore PDF supplemental table extraction
