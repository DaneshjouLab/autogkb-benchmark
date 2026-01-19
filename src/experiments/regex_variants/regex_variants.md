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

---

## Detailed Breakdown

### Version 1 (v1) Process

1.  **Load Benchmark Data**: The process starts by loading the ground-truth data from `benchmark_v2`, which contains a dictionary mapping PMCIDs to a list of known variant annotations for each article.
2.  **Filter Article Text**: For each article (PMCID), the system reads the corresponding markdown file. It then extracts only the text from the "Methods" and "Conclusions" sections. This was done to focus on the most relevant parts of the article where variants are typically discussed.
3.  **Apply Regex Patterns**: A set of basic regular expressions are applied to the filtered text to find variant mentions:
    *   **rsIDs**: `\brs\d{4,}\b` - Matches "rs" followed by four or more digits.
    *   **Star Alleles**: `\b(CYP\w+)\*(\d+)\b` - Specifically targets star alleles for genes starting with "CYP".
    *   **HLA Alleles**: `\bHLA-[A-Z]+\d*\*\d+:\d+\b` - Matches formal HLA allele notation that includes the "HLA-" prefix.
4.  **Score and Evaluate**: The variants extracted by the regex patterns are compared against the ground-truth variants for that article. The evaluation calculates:
    *   **Recall (Match Rate)**: The percentage of ground-truth variants that were successfully found by the regex patterns.
    *   **Precision**: The percentage of extracted variants that were correct (i.e., present in the ground-truth).
5.  **Aggregate Results**: The recall and precision are calculated for each article, and then an average is taken across all articles in the benchmark to produce the final summary statistics (Average Recall: 42.0%, Average Precision: 31.4%).

### Version 2 (v2) Process

1.  **Load Benchmark Data**: Same as v1, the process begins by loading the ground-truth data.
2.  **Use Full Article Text**: Unlike v1, this version uses the **entire text** of the article markdown, not just the "Methods" and "Conclusions" sections. This was a key change to improve coverage, as variants can be mentioned in any part of the text.
3.  **Apply Enhanced Regex Patterns**: The regex patterns were expanded and improved:
    *   **Broader Gene List for Star Alleles**: The pattern was updated to include more gene families beyond `CYP`, such as `UGT`, `NUDT`, `DPYD`, `TPMT`, `NAT`, `SLCO`, and `ABCB`.
    *   **Flexible HLA Allele Matching**: A new pattern was introduced to handle HLA alleles that are written without the "HLA-" prefix (e.g., `B*5801`). The system then normalizes these finds into the standard `HLA-B*58:01` format.
    *   **rsIDs**: The rsID pattern remained the same.
4.  **Normalize and Score**: After extraction, the found variants are normalized to a standard format to ensure accurate comparison with the ground-truth data. The scoring and evaluation process for recall and precision is the same as in v1.
5.  **Aggregate Results**: The final statistics are aggregated across all articles, showing a significant improvement in recall (Average Recall: 69.9%, Average Precision: 37.5%). The increase in recall was primarily due to using the full text and the broader set of regex patterns.

### Version 3 (v3) Process

1.  **Load Data and Use Full Text**: This version continues to use the full text of the articles, similar to v2.
2.  **Advanced Regex and Contextual Analysis**: V3 introduces much more sophisticated patterns and contextual analysis to handle complex and implicit variant mentions.
    *   **Expanded Pharmacogene List**: The list of target genes was expanded to over 25 pharmacogenes, significantly broadening the search space for star alleles.
    *   **Standalone Star Alleles**: To find mentions like `*3` or `*4` that are not directly attached to a gene name, the system searches for the nearest gene name within a 200-character window. If a relevant gene is found, the allele is associated with it (e.g., finding "NUDT15" near "*3" results in "NUDT15*3").
    *   **Spaced Star Alleles**: Handles cases where a space exists between the gene and the allele (e.g., "NUDT15 *3").
    *   **Diplotype and Copy Number Variants**: New patterns were added to recognize diplotypes (e.g., `*1×N/*2` or `*1/*10×N`) and copy number variations (e.g., `*1xN`, `*2xN`), often searching within a larger 800-character window to link them correctly.
    *   **HLA Parenthetical Notation**: A specific pattern was developed to parse complex HLA notations where multiple alleles are grouped, such as `HLA-B*38:(01/02)` or `B*39:(01/05/06/09)`.
3.  **Normalize, Score, and Evaluate**: The extracted variants, including those inferred from context, are normalized and then scored against the ground truth for recall and precision.
4.  **Aggregate Results**: This version achieved the highest performance, with an average recall of 87.8% and an average precision of 42.8%. Crucially, it achieved perfect recall on 69% of the articles (22 out of 32), demonstrating the effectiveness of the advanced contextual patterns.
