# PubTator Variant Finding Experiment

## Overview

This experiment evaluates the PubTator3 API for extracting pharmacogenetic variants from biomedical literature. PubTator is NCBI's automatic annotation service that identifies biomedical concepts including variants (mutations) from PubMed abstracts and full-text articles.

## Approaches Tested

### 1. PubTator API with Full Text (`pubtator_api_v1`)
- Query the PubTator3 API with PMIDs
- Request full-text annotations (`full=true`)
- Extract variant annotations from the BioC JSON response

### 2. PubTator API with Abstract Only (`pubtator_api_abstract_only`)
- Query the PubTator3 API with PMIDs
- Request abstract-only annotations (`full=false`)
- Extract variant annotations from the BioC JSON response

### 3. PubTator on Parsed Markdown (Not Feasible)
- **Finding**: PubTator does not provide an API to run annotation on arbitrary text
- The API only works on PMIDs/PMCIDs that are already indexed in PubMed/PMC
- Therefore, we cannot use PubTator on our parsed markdown text directly

## Results Summary

| Method | Recall | Precision | Perfect Recall | Description |
|--------|--------|-----------|----------------|-------------|
| **PubTator (Full Text)** | 36.3% | 19.4% | 10/32 (31%) | Full-text annotations via API |
| **PubTator (Abstract Only)** | 26.4% | 31.8% | 6/32 (19%) | Abstract-only annotations |

### Comparison with Other Methods

| Method | Recall | Precision | Perfect Recall |
|--------|--------|-----------|----------------|
| **Regex v5** | **93.4%** | 41.9% | **25/32 (78%)** |
| LLM-Only (Sonnet 4.5) | 71.5% | **46.3%** | 16/32 (50%) |
| LLM-Only (GPT-4o) | 65.7% | 42.1% | 14/32 (44%) |
| **PubTator (Full Text)** | 36.3% | 19.4% | 10/32 (31%) |
| **PubTator (Abstract Only)** | 26.4% | 31.8% | 6/32 (19%) |

**PubTator significantly underperforms** compared to both regex and LLM methods.

## Key Findings

### What PubTator Detects Well
- **rsIDs**: PubTator reliably identifies rs#### identifiers (e.g., rs9923231, rs2108622)
- It also normalizes HGVS notations to rsIDs (e.g., "-1639G>A" → rs9923231)

### What PubTator Misses (Critical Gap)

PubTator **does NOT recognize** pharmacogenetic star alleles and HLA alleles:

1. **Star alleles** (0% detection):
   - CYP2C9*3, CYP2B6*1, CYP2C19*17
   - UGT1A1*28, NUDT15*3
   - CYP2D6*1, CYP2D6*10, CYP2D6*41

2. **HLA alleles** (0% detection):
   - HLA-B*58:01, HLA-A*31:01
   - HLA-DRB1*03:01, HLA-C*04:01

3. **Phenotype descriptors** (0% detection):
   - "CYP2D6 poor metabolizer"
   - "CYP2C19 intermediate metabolizer"

PubTator recognizes gene names (e.g., "CYP2C19") but NOT the star allele notation (e.g., "CYP2C19*2").

### Important Nuance: rsIDs vs Star Alleles

Many star alleles are defined by specific rsIDs (e.g., NUDT15*3 = rs116855232, CYP2C9*3 = rs1057910). We analyzed whether PubTator finds the corresponding rsIDs when star alleles are missed:

| Outcome | Count | Percentage |
|---------|-------|------------|
| rsID found as "extra" | 6 | 27.3% |
| rsID found as "match" | 1 | 4.5% |
| rsID NOT found | 15 | 68.2% |

**Examples:**
- **PMC12035587**: Missed `NUDT15*3` but found `rs116855232` (the defining rsID) - counted as extra
- **PMC10399933**: Missed `CYP2C9*2` and `CYP2C9*3` but found `rs1799853` and `rs1057910` - counted as extras
- **PMC11971672**: Missed `CYP2C19*2`, `CYP2C19*3`, `CYP2C19*17` and did NOT find any corresponding rsIDs

**Implications:**
1. ~27% of "extras" are actually correct variants in rsID format
2. This is a **nomenclature mismatch** problem, not just a detection problem
3. However, 68% of corresponding rsIDs are still not found at all
4. A star-allele-to-rsID mapping layer could improve apparent precision but recall would still be poor

### Example: PMC12036300

**Ground truth variants**: CYP2C19*1, CYP2C19*2, CYP2C19*17

**PubTator response**:
- Genes detected: CYP2C19 (11 mentions)
- Variants detected: 0

**Result**: 0% recall - PubTator completely misses star allele notation.

## Per-Article Results (Full Text)

### Perfect Recall (10 articles)
- PMC5508045: 100% recall (4/4 rsIDs)
- PMC6465603: 100% recall (2/2 rsIDs)
- PMC12038368: 100% recall (2/2 rsIDs)
- PMC12331468: 100% recall (4/4 rsIDs)
- PMC12319246: 100% recall (8/8 rsIDs)
- PMC10275785: 100% recall (2/2 rsIDs)
- PMC6714829: 100% recall (2/2 rsIDs)
- PMC2859392: 100% recall (1/1 rsID)
- PMC8973308: 100% recall (3/3 rsIDs)
- PMC10786722: 100% recall (3/3 rsIDs)

**Pattern**: Perfect recall only when ground truth contains exclusively rsIDs.

### Complete Misses (17 articles with 0% recall)
All articles with star alleles or HLA alleles in the ground truth:
- PMC12036300: CYP2C19 star alleles
- PMC554812: HLA alleles
- PMC5561238: 43 HLA alleles
- PMC10946077: UGT1A1 star alleles
- PMC6435416: CYP2D6 star alleles
- And more...

## Analysis

### Why PubTator Underperforms for Pharmacogenomics

1. **Domain mismatch**: PubTator's tmVar3 system is optimized for general mutation detection (HGVS notation, rsIDs), not pharmacogenetic nomenclature

2. **Star allele format**: The `GENE*N` notation is specific to pharmacogenomics and not a standard HGVS format

3. **HLA nomenclature**: HLA allele naming (`HLA-B*58:01`) is domain-specific and not covered by standard variant detection

4. **Metabolizer phenotypes**: These are clinical descriptors, not sequence variants, and fall outside PubTator's scope

### Trade-off: Full Text vs Abstract Only

| Metric | Full Text | Abstract Only | Difference |
|--------|-----------|---------------|------------|
| Recall | 36.3% | 26.4% | +10.0% |
| Precision | 19.4% | 31.8% | -12.4% |
| Perfect Recall | 31% | 19% | +12% |

- **Full text**: Higher recall (+10%) but lower precision (-12%)
- **Abstract only**: Higher precision but misses variants in main text
- **Recommendation**: Use full text if recall is priority

## Limitations of PubTator for Pharmacogenomics

1. **Not designed for star alleles**: PubTator's variant detection is based on sequence-level mutations, not haplotype nomenclature

2. **No custom annotation API**: Cannot run PubTator on arbitrary text; must use pre-indexed PMIDs

3. **Limited pharmacogene coverage**: Gene-specific allele nomenclature not recognized

4. **No phenotype detection**: Metabolizer status descriptions not captured

## Recommendations

### When to Use PubTator
- **rsID extraction only**: If you only need rsIDs, PubTator is adequate
- **HGVS normalization**: PubTator normalizes various mutation formats to rsIDs
- **Cross-referencing**: Use as one data source combined with other methods

### When NOT to Use PubTator
- **Pharmacogenomics**: Star alleles and HLA alleles are critical
- **Comprehensive extraction**: When high recall is needed
- **Domain-specific variants**: For specialized nomenclature systems

### Better Alternatives for Pharmacogenomics

1. **Regex-based extraction (Regex v5)**: 93.4% recall
   - Handles star alleles, HLA alleles, rsIDs
   - Fast and deterministic

2. **LLM-based extraction**: 71.5% recall
   - Better at understanding context
   - Can handle novel formats

3. **Hybrid approach**: Combine regex + LLM for best results

## Star Allele to rsID Mapping: Background and Analysis

### How Star Alleles and rsIDs Relate

Star alleles (e.g., `CYP2C19*2`) and rsIDs (e.g., `rs4244285`) represent genetic variation at different levels:

| Concept | Star Allele | rsID |
|---------|-------------|------|
| **What it represents** | A haplotype (combination of variants) | A single nucleotide position |
| **Naming authority** | [PharmVar](https://www.pharmvar.org) | [dbSNP](https://www.ncbi.nlm.nih.gov/snp/) |
| **Example** | CYP2C19*2 | rs4244285 |
| **Complexity** | Can be defined by 1 or many SNPs | Single position |
| **Domain** | Pharmacogenomics | General genomics |

**Key insight**: Some star alleles are defined by a **single SNP** (simple 1:1 mapping), while others require **multiple SNPs** (complex haplotypes with no single rsID equivalent).

### Star Alleles with Simple 1:1 rsID Mappings

These star alleles are defined by a single functional variant:

| Star Allele | rsID | Variant | Function | Source |
|-------------|------|---------|----------|--------|
| CYP2C19*2 | rs4244285 | c.681G>A | No function (splicing defect) | [PharmVar](https://www.pharmvar.org/gene/CYP2C19), [SNPedia](https://snpedia.com/index.php/Rs4244285) |
| CYP2C19*3 | rs4986893 | c.636G>A | No function (stop codon) | [PharmVar](https://www.pharmvar.org/gene/CYP2C19) |
| CYP2C19*17 | rs12248560 | c.-806C>T | Increased function | [PharmVar](https://www.pharmvar.org/gene/CYP2C19), [SNPedia](https://snpedia.com/index.php/Rs12248560) |
| CYP2B6*6 | rs3745274 | c.516G>T | Decreased function | [SNPedia](https://snpedia.com/index.php/Rs3745274) |
| CYP2B6*9 | rs3745274 | c.516G>T | Decreased function | Same SNP as *6 |
| CYP2C9*2 | rs1799853 | c.430C>T | Decreased function | [PharmVar](https://www.pharmvar.org/gene/CYP2C9) |
| CYP2C9*3 | rs1057910 | c.1075A>C | Decreased function | [PharmVar](https://www.pharmvar.org/gene/CYP2C9) |
| CYP2C9*8 | rs7900194 | c.449G>A | Decreased function | [PharmVar](https://www.pharmvar.org/gene/CYP2C9) |
| UGT1A1*6 | rs4148323 | c.211G>A | Decreased function | [PharmVar](https://www.pharmvar.org/gene/UGT1A1) |
| UGT1A1*28 | rs8175347 | (TA)7 repeat | Decreased function | [SNPedia](https://snpedia.com/index.php/Rs8175347) |
| NUDT15*3 | rs116855232 | c.415C>T (R139C) | No function | [OMIM](https://omim.org/entry/615792), [PMC4375304](https://pmc.ncbi.nlm.nih.gov/articles/PMC4375304/) |

### Star Alleles WITHOUT Simple rsID Mappings

These cannot be represented by a single rsID:

#### 1. CYP2D6 Star Alleles (Complex Haplotypes)

CYP2D6 is notoriously complex. Most star alleles are defined by **multiple SNPs** that must occur together:

| Star Allele | Why No Single rsID | Reference |
|-------------|-------------------|-----------|
| CYP2D6*1 | Wild-type reference (no variant) | N/A |
| CYP2D6*4 | Defined by multiple SNPs including rs3892097 | [PharmVar](https://www.pharmvar.org/gene/CYP2D6) |
| CYP2D6*10 | Defined by rs1065852 + other variants | [PharmVar](https://www.pharmvar.org/gene/CYP2D6) |
| CYP2D6*41 | Multiple defining SNPs | [PharmVar](https://www.pharmvar.org/gene/CYP2D6) |
| CYP2D6*5 | Gene deletion (structural variant) | No SNP possible |
| CYP2D6*1xN | Gene duplication (CNV) | No SNP possible |

**Why CYP2D6 is different**: "The same SNP may be found on multiple genetic backgrounds, which, based on the totality of variants present, show different activities" ([PMC6441960](https://pmc.ncbi.nlm.nih.gov/articles/PMC6441960/)). For example, rs16947 appears in 19+ different CYP2D6 haplotypes.

#### 2. HLA Alleles (Different Naming System Entirely)

HLA alleles use a completely different nomenclature based on serological typing and sequence-based typing:

| HLA Allele | Why No rsID Mapping | Notes |
|------------|---------------------|-------|
| HLA-B*57:01 | HLA nomenclature based on protein sequence | Tag SNP rs2395029 is 99.9% predictive in Europeans, but not a definition |
| HLA-B*58:01 | HLA nomenclature based on protein sequence | No equivalent rsID |
| HLA-A*31:01 | HLA nomenclature based on protein sequence | No equivalent rsID |
| HLA-DRB1*07:01 | HLA class II, protein-based naming | No equivalent rsID |

**HLA alleles are named by protein sequence**, not by the underlying DNA variants. While SNPs like rs2395029 can be used as "tag SNPs" to predict HLA alleles in certain populations, this is an **imputation strategy**, not a definition. The linkage varies by population ([SNPedia](https://www.snpedia.com/index.php/Rs2395029)).

#### 3. Wild-type (*1) Alleles

| Star Allele | Why No rsID |
|-------------|-------------|
| CYP2C19*1 | Reference sequence (absence of variants) |
| CYP2D6*1 | Reference sequence |
| CYP2C9*1 | Reference sequence |
| UGT1A1*1 | Reference sequence |

Wild-type alleles are defined by the **absence** of known functional variants, so there's no SNP to detect.

#### 4. Metabolizer Phenotypes (Not Genetic Variants)

| Term | Why No rsID |
|------|-------------|
| "CYP2D6 poor metabolizer" | Clinical phenotype, not a variant |
| "CYP2C19 intermediate metabolizer" | Clinical phenotype, not a variant |

These describe the **functional outcome** of genotype combinations, not the genotype itself.

### How I Verified These Mappings

1. **Primary source**: [PharmVar](https://www.pharmvar.org/genes) - The official repository for pharmacogene star allele definitions

2. **Cross-reference**: [SNPedia](https://www.snpedia.com) - Community-curated SNP database with clinical annotations

3. **Literature**: PubMed/PMC articles on specific gene-drug associations

4. **Validation approach**:
   - For each star allele in our benchmark's ground truth, I searched PharmVar/SNPedia for the defining variant(s)
   - If a single rsID defines the allele, I added it to the mapping table
   - If multiple variants define it, or it's a structural variant, I noted it as "no simple mapping"

### Results from Our Analysis

Using the mappings above, I checked whether PubTator found corresponding rsIDs for missed star alleles:

| Star Allele Missed | Corresponding rsID | PubTator Found rsID? | Article |
|--------------------|-------------------|---------------------|---------|
| NUDT15*3 | rs116855232 | ✓ Yes (as extra) | PMC12035587 |
| CYP2C9*2 | rs1799853 | ✓ Yes (as extra) | PMC10399933 |
| CYP2C9*3 | rs1057910 | ✓ Yes (as extra) | PMC10399933 |
| CYP2B6*6 | rs3745274 | ✓ Yes (as extra) | PMC11603346 |
| CYP2B6*9 | rs3745274 | ✓ Yes (as match) | PMC4916189 |
| UGT1A1*6 | rs4148323 | ✓ Yes (as extra) | PMC10946077 |
| CYP2C9*8 | rs7900194 | ✓ Yes (as extra) | PMC4706412 |
| CYP2C19*2 | rs4244285 | ✗ No | PMC12036300, PMC11971672 |
| CYP2C19*17 | rs12248560 | ✗ No | PMC12036300, PMC11971672 |
| UGT1A1*28 | rs8175347 | ✗ No | PMC10946077, PMC11062152 |
| CYP2D6*4 | rs3892097 (partial) | ✗ No | PMC3548984, PMC6435416 |
| CYP2D6*10 | rs1065852 (partial) | ✗ No | PMC3548984, PMC3584248 |
| HLA-B*58:01 | None (HLA naming) | N/A | PMC554812 |
| HLA-A*31:01 | None (HLA naming) | N/A | PMC3113609 |

### Summary

- **7/22 (32%)** of missed star alleles had their corresponding rsID found by PubTator
- **15/22 (68%)** had rsIDs that were NOT found
- **All HLA alleles** have no rsID equivalent and cannot be found via rsID detection
- **CYP2D6 star alleles** have complex haplotype definitions that don't map to single rsIDs

## Technical Details

### API Endpoint
```
https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={PMID}&full={true|false}
```

### Rate Limit
- 3 requests per second
- 0.35 second delay between requests

### Response Format
BioC JSON with annotations containing:
- `type`: "Variant" for mutation annotations
- `infons.rsid`: Normalized rsID
- `text`: Original text mention

## Files

- `pubtator_variants.py`: Main experiment script
- `results/pubtator_api_v1.json`: Full-text results
- `results/pubtator_api_abstract_only.json`: Abstract-only results

## Conclusion

**PubTator is not suitable for pharmacogenomics variant extraction.** Its 36% recall significantly underperforms compared to regex (93%) and LLM (72%) methods. The critical gap is the complete lack of star allele and HLA allele recognition, which are fundamental to pharmacogenomics.

For pharmacogenomics applications, use the regex v5 approach or LLM-based extraction instead. PubTator may be useful as a supplementary source for rsID extraction and HGVS normalization, but should not be the primary extraction method.

---

*Experiment run: January 2026*
*Benchmark: 32 pharmacogenomics articles, 322 total variants*
