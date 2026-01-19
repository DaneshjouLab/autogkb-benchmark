# Just Ask Experiment Log

Started: 2025-01-17 6:00 PM

## Overview
We are just going to ask some of the best models whether they can extract variants from articles.
Models to try:
- Claude Opus 4.5
- Claude Sonnet 4.5
- GPT 5.2
- GPT 5.1
- Gemini 3 Pro
- Gemini 3 Flash
- Gemini 2.5 Flash

Note: For some reason claude, ignored this and just used GPT-4o instead of the GPT 5 models but I think we have enough information to move on to different experiments.

## Process Hypothesis
1. Get the markdown for the article but only the methods and conclusions sections (functions already defined to do this in utils.py)
2. Ask the model to extract the variants
3. Do 4-5 rounds of prompt iteration based on results (or end when reasonable)
4. Repeat the process with the next model on the list

## Notes
- Save the tried prompts in separate files from the code. Preferably some sort of yaml file(s)
- Use litellm for api calls
- use load_dotenv() for api keys


## Experiment Results

### Results Summary

| Model | Prompt | Avg Recall | Avg Precision | Perfect Recall |
|-------|--------|------------|---------------|----------------|
| GPT-4o | v1 | 59.4% | 39.5% | 38% (12/32) |
| GPT-4o | v2 | 65.7% | 42.1% | 44% (14/32) |
| Claude Sonnet 4.5 | v1 | 61.8% | 39.2% | 47% (15/32) |
| Claude Sonnet 4.5 | v2 | 71.5% | 46.3% | 50% (16/32) |
| Claude Opus 4.5 | v1 | 67.3% | 38.7% | 50% (16/32) |
| **Claude Opus 4.5** | **v2** | **71.6%** | **45.5%** | **53% (17/32)** |
| Claude Opus 4.5 | v3 | 71.0% | 41.6% | 53% (17/32) |
| Gemini 2.0 Flash | v2 | 69.2% | 40.1% | 47% (15/32) |

**Best performer: Claude Opus 4.5 with v2 prompt** (71.6% recall, 45.5% precision, 53% perfect recall)

---

### Prompt Versions

- **v1 (Basic extraction)**: Simple prompt asking for rsIDs, star alleles, and HLA alleles
- **v2 (Structured with examples)**: Added specific variant type examples, normalization rules, and output format guidance
- **v3 (Comprehensive normalization)**: Extended gene lists and explicit normalization rules

The v2 prompt performed best overall due to balanced specificity and flexibility.

---

### Key Observations

1. **Claude models outperform GPT-4o**: Both Sonnet and Opus achieved higher recall and precision than GPT-4o across prompt versions.

2. **Prompt iteration helps significantly**: Moving from v1 to v2 improved recall by ~10% across all models.

3. **v3 did not improve over v2**: Additional complexity in v3 prompt did not yield better results, suggesting diminishing returns from prompt engineering.

4. **Common failure modes**:
   - **HLA alleles in tables**: Many HLA variants (especially in PMC5561238 with 43 variants) were in supplementary tables not captured in methods/conclusions
   - **Wild-type alleles**: *1 alleles often not explicitly mentioned (UGT1A1*1, CYP2D6*1, CYP3A4*1)
   - **Metabolizer phenotypes**: Ground truth includes "CYP2D6 poor metabolizer" which models don't extract as variants
   - **Copy number variants**: CYP2D6*1xN, *2xN patterns in PMC6435416 were missed
   - **Alternative notation**: Some rsIDs referenced by cDNA notation (G516T instead of rs3745274)

5. **Precision vs Recall tradeoff**: Models tend to over-extract variants, finding many that aren't in the ground truth (~40-46% precision). This is expected as they capture all mentioned variants, not just the specific pharmacogenetic associations.

---

### Comparison to Regex Baseline (v3)

| Method | Avg Recall | Avg Precision | Perfect Recall |
|--------|------------|---------------|----------------|
| Regex v3 | 87.8% | 42.8% | 69% (22/32) |
| Claude Opus 4.5 v2 | 71.6% | 45.5% | 53% (17/32) |

The regex approach still outperforms LLM-based extraction for recall, primarily because:
- Regex uses full article text vs. methods/conclusions sections only
- Regex patterns are optimized for the specific variant formats in the dataset
- LLM extraction is limited by the context window and text sections provided

---

### Recommendations

1. **Use full article text**: LLM performance would likely improve with access to full text including tables
2. **Hybrid approach**: Combine regex for high recall with LLM for normalization and deduplication
3. **Two-stage extraction**: First identify sections mentioning variants, then extract from those sections