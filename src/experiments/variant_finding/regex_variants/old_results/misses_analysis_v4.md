# Regex Extraction V4 - Missed Variants Analysis

**Overall Performance:** 91.3% recall, 43.4% precision across 32 articles

This document analyzes the **missed variants** (false negatives) from the regex extraction v4 run.

**V4 Improvements over V3:**
- SNP notation expansion (516G>T -> rs3745274)
- PharmGKB-derived mappings for 739 SNP notations
- Various SNP formats: GENE 516G>T, GENE-G516T

---

## Summary of Miss Categories

| Category | Count | Example |
|----------|-------|---------|
| Wildtype alleles (`*1`) | 5 | `UGT1A1*1`, `CYP2D6*1`, `CYP2C9*1` |
| Star alleles in phenotype-only articles | 11 | Article discusses phenotypes but benchmark expects specific alleles |
| Metabolizer status as variant | 2 | `CYP2C19 intermediate metabolizer` |
| HLA variants | 4 | `HLA-B*35:10`, `HLA-DRB1*10:01` |
| rsIDs in alternative notation | 2 | `rs3745274`, `rs28399499` |

**Total misses: 24 variants across 8 articles (of 32 total)**

---

## Detailed Miss Analysis by Article

### PMC4916189
**Missed:** `cyp2b6*9`, `cyp2b6*1`

**Analysis:** The SNP notation expansion successfully captured `rs3745274` and `rs28399499` from the text. However, the benchmark also expects the corresponding star alleles (`*9`, `*1`) which are not explicitly mentioned in the article.

**Note:** The article uses SNP nomenclature (516G>T, 983T>C) rather than star allele notation. V4 successfully expanded these to rsIDs, but the benchmark expects both rsIDs AND star alleles.

**Recommendation:** Consider adding SNP-to-star-allele mapping in addition to SNP-to-rsID:
- `CYP2B6 516G>T` -> `rs3745274` AND `CYP2B6*9`

---

### PMC5561238
**Missed:** `hla-b*35:10`, `hla-drb1*10:01`, `rs3745274`, `rs28399499`, `hla-b*56:06`, `hla-drb1*08:01`

**Analysis:**
- 4 HLA variants missed - likely formatting/OCR issues or non-standard notation
- 2 rsIDs missed - these may appear in alternative notation not captured by current SNP expansion patterns

**Location check needed:** Verify if these variants appear in the article text and in what format.

**Recommendation:**
- Investigate specific HLA formatting in this article
- Check if rs3745274/rs28399499 appear in a notation not covered by current patterns

---

### PMC10946077 & PMC11062152
**Missed:** `ugt1a1*1` (in both articles)

**Analysis:** The wildtype `*1` allele is not explicitly mentioned. Articles discuss UGT1A1*28 and *6 explicitly but reference wildtype as comparison without using `*1` notation.

**Recommendation:** Same as v3 - consider wildtype inference when other star alleles are present.

---

### PMC10880264
**Missed:** `cyp2d6 poor metabolizer`, `cyp2c19 intermediate metabolizer`

**Analysis:** The benchmark expects **metabolizer phenotype status** as the variant, not specific alleles. This is fundamentally different from genetic variant extraction.

**Location in article:**
- Table headers include "PM" (poor metabolizer), "IM" (intermediate metabolizer)
- Text discusses metabolizer status classifications

**Recommendation:** Add metabolizer status extraction pattern:
```python
METABOLIZER_PATTERN = r"(CYP\w+)\s+(poor|intermediate|normal|rapid|ultrarapid)\s+metabolizer"
```

---

### PMC6435416
**Missed:** `cyp2d6*17`, `cyp2d6*35`, `cyp2d6*41`, `cyp2d6*3`, `cyp2d6*4`, `cyp2d6*29`, `cyp2d6*6`, `cyp2d6*9`, `cyp2d6*5`, `cyp2d6*4xn`

**Analysis:** This article discusses CYP2D6 **metabolizer phenotypes** (poor, intermediate, normal, ultrarapid) but does NOT list individual star alleles in the main text. The benchmark annotation includes all alleles that comprise these phenotypes.

**What was found:** `cyp2d6*2xn`, `cyp2d6*2`, `cyp2d6*1xn`, `cyp2d6*1`, `cyp2d6*10` (5 of 15 expected)

**Recommendation:** This is a fundamental mismatch between article content and benchmark expectations. Options:
1. Accept that regex cannot capture alleles not mentioned in text
2. Flag phenotype-only articles for manual review
3. Add metabolizer-to-allele expansion as post-processing

---

### PMC3548984
**Missed:** `cyp2d6*1`

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

---

### PMC10399933
**Missed:** `cyp2c9*1`

**Analysis:** Wildtype `*1` allele not explicitly mentioned.

---

## Impact of V4 Improvements

### SNP Notation Expansion Results

| Article | rsIDs Recovered via SNP Expansion |
|---------|----------------------------------|
| PMC5508045 | `rs9923231` |
| PMC4916189 | `rs28399499`, `rs3745274` |
| PMC4706412 | `rs9923231` |
| PMC2859392 | `rs28371759`, `rs28399499`, `rs3745274` |
| PMC3387531 | `rs28399499`, `rs3745274` |

**Summary:** 5 articles benefited from SNP notation expansion, recovering 9 rsIDs total.

### Recall Improvement: V3 -> V4

| Metric | V3 | V4 | Change |
|--------|-----|-----|--------|
| Average Recall | 87.8% | 91.3% | **+3.5%** |
| Average Precision | 42.8% | 43.4% | +0.6% |

---

## Recommended Improvements for V5

### 1. Metabolizer Status Extraction (NEW)
Add patterns to capture metabolizer phenotypes as variants:

```python
METABOLIZER_PATTERN = r"(CYP\w+)\s+(poor|intermediate|normal|rapid|ultrarapid)\s+metabolizer"
METABOLIZER_ABBREV = r"(CYP\w+)[\s\-]*(PM|IM|NM|RM|UM)\b"
```

**Expected recovery:** 2 variants

### 2. Wildtype Inference (from V3)
When star alleles like `*2`, `*3`, etc. are found for a gene, consider automatically adding `*1` if:
- The article discusses genotype comparisons
- Terms like "wild-type", "reference", or "normal function" appear

```python
WILDTYPE_INDICATORS = ["wild-type", "wildtype", "WT", "reference allele", "normal function"]
```

**Expected recovery:** 5 variants

### 3. SNP-to-Star-Allele Mapping
Extend the SNP expander to also return star allele mappings:

```python
SNP_TO_STAR = {
    ("CYP2B6", "516G>T"): "CYP2B6*9",
    ("CYP2B6", "983T>C"): "CYP2B6*18",
    # ... etc
}
```

**Expected recovery:** 1-2 variants

### 4. HLA Format Investigation
Investigate the specific HLA formats in PMC5561238 that weren't captured:
- `HLA-B*35:10`
- `HLA-DRB1*10:01`
- `HLA-B*56:06`
- `HLA-DRB1*08:01`

---

## Impact Analysis

If all recommended improvements were implemented:

| Improvement | Variants Recovered | Est. Recall Gain |
|-------------|-------------------|------------------|
| Metabolizer status extraction | 2 | +0.6% |
| Wildtype inference | 5 | +1.5% |
| SNP-to-star-allele mapping | 2 | +0.6% |
| HLA format fixes | 4 | +1.2% |

**Estimated new recall: ~95%** (up from 91.3%)

---

## Critical Finding: Missing Supplementary Materials

**The article markdown files do NOT include supplementary material content.**

### Evidence from PMC6435416

The article explicitly states:
> "A complete list of *CYP2D6* diplotypes identified is included in **Supplemental Table S3** (online)."

But the markdown contains only empty placeholders:
```markdown
## Supplementary Material

## Acknowledgements:
...
### Supplementary Materials

### Supplementary Materials
```

The **10 missing variants** in PMC6435416 (`CYP2D6*3`, `*4`, `*4xN`, `*5`, `*6`, `*9`, `*17`, `*29`, `*35`, `*41`) are listed in the **supplementary tables**, not the main article text.

### Impact

This explains why certain articles have significant misses despite the variants being part of the benchmark:
- The benchmark was created from full article content **including supplementary materials**
- The extraction pipeline only has access to the **main article text**

### Verified: PMC6435416 Supplementary Content

Checking the [PMC article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC6435416/), the supplement contains:
- **Supplemental Table S1**: PCR primers
- **Supplemental Table S2**: CYP2D6 star alleles ← **This is where the missing alleles are!**
- **Supplemental Table S3**: Complete list of CYP2D6 diplotypes identified
- **Supplemental Table S4**: All AEs with frequency by metabolizer status
- **Supplemental Table S5**: Variables associated with AEs

File: `NIHMS1518655-supplement-1.pdf` (121.9KB)

---

## BioC API Coverage (Tested)

Tested the [BioC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/FAIR-SMART/) against all 32 benchmark articles:

| Metric | Count | Percentage |
|--------|-------|------------|
| Articles with BioC supplements | 14 | 43.8% |
| Articles without BioC supplements | 18 | 56.2% |

**Articles with BioC-accessible supplements:**
PMC5508045, PMC4916189, PMC12036300, PMC5561238, PMC6465603, PMC12038368, PMC6435416, PMC11971672, PMC3839910, PMC10786722, PMC3584248, PMC4706412, PMC6714829, PMC11603346

**Key finding:** The BioC API provides pre-processed text from supplementary PDFs. For PMC6435416, this includes the complete CYP2D6 star allele table (Table S2) that contains all 10 missing variants.

---

## Approach for Downloading Supplementary Materials

### Available APIs and Methods

| Method | Description | Best For |
|--------|-------------|----------|
| [BioC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/FAIR-SMART/) | RESTful API returning BioC XML/JSON | Pre-processed text, structured data |
| [PMC FTP](https://pmc.ncbi.nlm.nih.gov/tools/ftp/) | Download `.tar.gz` packages | Complete packages with all files |
| [OA Web Service](https://pmc.ncbi.nlm.nih.gov/tools/oa-service/) | Discover downloadable resources | Finding file locations |
| Direct download | From PMC article page | Individual PDFs |

### Typical Supplementary Material Formats

Based on PMC documentation and inspection:

| Format | Frequency | Processing Difficulty | Notes |
|--------|-----------|----------------------|-------|
| **PDF** | ~70% | High | Tables often as images, need OCR or PDF parsing |
| **Word (.docx)** | ~15% | Medium | Can extract text with python-docx |
| **Excel (.xlsx)** | ~10% | Low | Structured data, easy to parse |
| **CSV/TXT** | ~3% | Very Low | Direct text extraction |
| **Images** | ~2% | Very High | Requires OCR/vision models |

### Recommended Approach

#### Option 1: Hybrid Pipeline (Recommended)

```
1. Download Phase
   ├── Use BioC API first (returns pre-processed text when available)
   ├── Fall back to FTP .tar.gz packages
   └── Extract all supplement files

2. Processing Phase (by format)
   ├── PDF → PyMuPDF/pdfplumber for text + tables
   │         └── Vision LLM for complex/image-based tables
   ├── Excel → pandas/openpyxl → markdown tables
   ├── Word → python-docx → markdown
   └── CSV/TXT → direct inclusion

3. Integration Phase
   └── Append processed supplement text to article markdown
```

#### Option 2: LLM-First Approach

Use a vision-capable LLM (Claude, GPT-4V) to:
1. Download raw PDF supplements
2. Send PDF pages directly to vision model
3. Extract structured data (tables, variant lists)

**Pros:** Handles any format, understands context
**Cons:** More expensive, slower, potential hallucination on numbers

#### Option 3: BioC API Only (Simplest)

```python
# Example: Get supplementary materials for PMC6435416
url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/supplmat.cgi/BioC_JSON/PMC6435416/All"
```

**Pros:** Pre-processed, consistent format
**Cons:** May not have all files, limited to text-extractable content

### Implementation Recommendation (Updated After Testing)

**BioC API is the clear winner for quick implementation:**

1. **44% of benchmark articles have BioC supplements** - no additional processing needed
2. **50% of all misses are recoverable** just by including BioC text
3. **PMC6435416 goes from 33% to 100% recall** - the worst-performing article becomes perfect

**Recommended implementation order:**

1. **Phase 1: BioC API integration** (Quick win - ~2.2% recall improvement)
   - Add BioC text fetch to the extraction pipeline
   - Concatenate supplement text with main article text
   - Re-run regex extraction on combined text

2. **Phase 2: PMC FTP for non-BioC articles** (Moderate effort)
   - Download `.tar.gz` packages for the 18 articles without BioC
   - Use pdfplumber for text-based PDF supplements

3. **Phase 3: Vision LLM for stubborn cases** (High effort, diminishing returns)
   - Only needed for image-based tables in PDFs
   - May not be worth it for the remaining ~12 variants

### When to Use an LLM/Agent

| Use Case | LLM Needed? | Why |
|----------|-------------|-----|
| PDF with text-based tables | No | pdfplumber handles this |
| PDF with image-based tables | **Yes** | Vision model required |
| Excel/CSV files | No | Direct pandas parsing |
| Identifying relevant supplements | Maybe | Could filter by filename patterns |
| Normalizing variant formats | Maybe | Regex may suffice for known formats |
| Understanding table structure | **Yes** | Complex nested tables benefit from LLM |

### Actual Impact (Tested)

**BioC API audit of all 8 articles with misses:**

| PMCID | Has BioC? | Misses | Recovered | Still Missing |
|-------|-----------|--------|-----------|---------------|
| PMC6435416 | ✓ Yes | 10 | **10 (100%)** | 0 |
| PMC5561238 | ✓ Yes | 6 | **2 (33%)** | 4 (HLA formats, rsIDs) |
| PMC4916189 | ✓ Yes | 2 | 0 | 2 (CYP2B6 star alleles) |
| PMC10946077 | ✗ No | 1 | - | 1 (UGT1A1*1) |
| PMC10880264 | ✗ No | 2 | - | 2 (metabolizer status) |
| PMC3548984 | ✗ No | 1 | - | 1 (CYP2D6*1) |
| PMC11062152 | ✗ No | 1 | - | 1 (UGT1A1*1) |
| PMC10399933 | ✗ No | 1 | - | 1 (CYP2C9*1) |

**Summary:**
- 3 of 8 articles with misses have BioC supplements
- **12 of 24 variants (50%) recoverable via BioC API**
- Remaining 12 misses: 5 no BioC available, 7 in BioC but not found (different format/location)

**Recall Impact:**

| Metric | Value |
|--------|-------|
| Current V4 recall | 91.3% |
| **With BioC supplements** | **93.5%** |
| Improvement | **+2.2%** |

**Per-article impact:**
- PMC6435416: 33.3% → 100% ✓
- PMC5561238: 86.0% → 90.7% ✓

---

### Quick Start: BioC API Test

```python
import requests

def get_supplement_list(pmcid: str) -> list[str]:
    """Get list of available supplementary files."""
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/supplmat.cgi/BioC_JSON/{pmcid}/LIST"
    response = requests.get(url)
    return response.json()

def get_supplement_text(pmcid: str) -> str:
    """Get all supplementary material text."""
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/supplmat.cgi/BioC_JSON/{pmcid}/All"
    response = requests.get(url)
    return response.text

# Test with PMC6435416
files = get_supplement_list("PMC6435416")
text = get_supplement_text("PMC6435416")
```

**Estimated impact:** This could recover 10+ variants across multiple articles where data is in supplements.

---

## Notes

- V4 successfully improved recall from 87.8% to 91.3% through SNP notation expansion
- Remaining misses fall into predictable categories: wildtype alleles, metabolizer phenotypes, and phenotype-only articles
- The 10 misses in PMC6435416 represent a fundamental mismatch - the article discusses phenotypes but the benchmark expects underlying genotypes
- Precision remains low (~43%) due to over-extraction - a relevance filter could help
- Consider whether metabolizer status and wildtype inference are within scope for a regex-based extractor

---

## Articles with Perfect Recall (24/32)

PMC5508045, PMC12036300, PMC554812, PMC6465603, PMC12038368, PMC12331468, PMC12319246, PMC10275785, PMC11971672, PMC11430164, PMC8790808, PMC3839910, PMC3113609, PMC10786722, PMC384715, PMC3584248, PMC12035587, PMC10993165, PMC4706412, PMC6714829, PMC2859392, PMC11603346, PMC8973308, PMC3387531
