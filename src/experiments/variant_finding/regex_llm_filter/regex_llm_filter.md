# Regex LLM Filter

This experiment combines the best performing regex-based variant extraction (V5 from regex_variants) with LLM-based filtering to remove false positives. The LLM filters out variants that are only mentioned for context rather than actually being studied in the article.

## Overview

**Approach:**
1. Extract variants using regex patterns (V5 with 93.4% recall, 41.9% precision)
2. Pass extracted variants + article text to an LLM
3. LLM filters out variants not explicitly studied in the research
4. Evaluate filtered results against ground truth

**Goal:** Improve precision while maintaining high recall.

## Implementation

### Files
- `regex_llm_filter.py` - Main experiment script
- `prompts.yaml` - Prompt configurations for different filtering strategies
- `results/` - JSON results for each model/prompt combination

### Usage

```bash
# Basic usage with default settings (Claude Sonnet 4.5, prompt v1)
python regex_llm_filter.py

# Specify model and prompt version
python regex_llm_filter.py --model gpt-4o --prompt v2

# Test on a subset of articles
python regex_llm_filter.py --model claude-sonnet-4-5-20250929 --prompt v1 --max-articles 5

# Try different prompts
python regex_llm_filter.py --model gemini-2.0-flash-001 --prompt v3
```

### Supported Models
- **Claude**: `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`
- **GPT**: `gpt-4o`, `gpt-4o-mini`
- **Gemini**: `gemini-2.0-flash-001`, `gemini-1.5-pro-002`

### Environment Setup
Requires API keys in `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Prompt Versions

### v1: Basic false positive filter
- Simple instructions to identify studied vs. mentioned variants
- Conservative approach: when in doubt, include the variant
- Good baseline for comparison

### v2: Contextual filtering with examples
- Provides explicit examples of include/exclude scenarios
- More detailed guidance on what constitutes "studied"
- Balances precision and recall

### v3: Strict filtering with section awareness
- Emphasizes methods/results sections for inclusion
- Stricter criteria for excluding variants
- May reduce recall but improve precision

### v4: Lenient filtering
- Only excludes clear false positives
- Prioritizes recall over precision
- Good for minimizing false negatives

### v5: Conservative filter with reasoning
- Detailed classification criteria with checkmarks
- Conservative approach favoring recall
- Most comprehensive instructions

## Initial Results

### Test Run (2 articles, v1 prompt)

**Regex V5 Baseline:**
- Average Recall: 85.7%
- Average Precision: 44.2%
- Perfect recall: 1/2 (50%)

**Regex + LLM Filter:**
- Average Recall: 85.7% (+0.0%)
- Average Precision: 59.2% (+15.0%)
- Perfect recall: 1/2 (50%)

**Key Findings:**
- ✓ LLM filtering improved precision by 15% (44.2% → 59.2%)
- ✓ Maintained the same recall (no true positives incorrectly removed)
- ✓ Successfully removed 3 false positives from PMC5508045
- Both GPT-4o and Claude Sonnet 4.5 achieved identical results

### Example: PMC5508045

**Regex Extraction (8 variants):**
- True positives: rs9923231, rs887829, rs2108622, rs1057910
- False positives: CYP2C9*1, CYP2C9*2, CYP2C9*3, rs8175347

**After LLM Filter (5 variants):**
- Kept: rs9923231, rs887829, rs2108622, rs1057910, CYP2C9*3
- Removed: CYP2C9*1, CYP2C9*2, rs8175347
- Result: 100% recall, 80% precision (vs. 50% precision before)

## Next Steps

1. **Full Benchmark Run**: Test on all 32 articles to get comprehensive metrics
2. **Prompt Comparison**: Compare all 5 prompt versions to find optimal strategy
3. **Model Comparison**: Test multiple models (Claude, GPT, Gemini) with best prompt
4. **Error Analysis**: Analyze remaining false positives and false negatives
5. **Hybrid Approaches**: Consider combining multiple prompts or models

## Notes

- Uses `python-dotenv` for loading .env variables ✓
- Uses the highest performing regex setup (V5) ✓
- Easy to try different LLMs via `--model` flag ✓
- Easy to try different prompts via `--prompt` flag ✓
- Accuracy/precision scores computed and displayed ✓
- Results saved as JSON for analysis ✓