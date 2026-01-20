# Visualization Guide

This directory contains comprehensive visualizations comparing all variant extraction methods tested on the AutoGKB benchmark.

## Generated Charts

### 1. `comparison_charts.png` - Comprehensive Multi-Method Comparison

**6-panel visualization comparing all methods:**

- **Panel 1: Recall Comparison** - Shows recall % for all 11 methods tested
  - Best: Regex v5 (93.4%)
  - Worst: Regex v1 baseline (42.0%)

- **Panel 2: Precision Comparison** - Shows precision % for all methods
  - Best: LLM-Only Sonnet 4.5 (46.3%)
  - Range: 31.4% - 46.3%

- **Panel 3: F1 Score** - Harmonic mean of recall and precision
  - Best: Regex v4 (58.9%)
  - Shows overall balanced performance

- **Panel 4: Recall vs Precision Scatter** - Trade-off visualization
  - Upper right is best (high recall + high precision)
  - Shows F1 contour lines
  - Regex methods cluster at high recall, moderate precision
  - LLM methods at moderate recall, moderate-high precision

- **Panel 5: Perfect Recall Articles** - Articles with 100% recall
  - Best: Regex v5 (25/32 articles = 78%)
  - Baseline: v1 (10/32 = 31%)

- **Panel 6: Radar Chart** - Multi-dimensional comparison of key methods
  - Compares v1 (baseline), v5 (best regex), Term Norm Hybrid, Regex+LLM Filter, LLM-Only
  - Shows strengths/weaknesses across all dimensions

**Color coding:**
- 🔵 Blue: Regex methods (v1-v5)
- 🟠 Orange: Hybrid methods (Term Norm, LLM Filter)
- 🟢 Green: LLM-only methods

### 2. `evolution_chart.png` - Regex Evolution Story

**2-panel focused visualization:**

- **Top panel: Recall Progression**
  - Line chart showing improvement v1 → v5 → Term Norm
  - Annotations show incremental improvements (+27.9%, +17.8%, etc.)
  - Total improvement highlighted: +51.3 percentage points
  - Each version labeled with key features

- **Bottom panel: Perfect Recall Articles**
  - Bar chart showing articles with 100% recall
  - Progress from 10/32 (31%) to 25/32 (78%)
  - Color-coded by version

**Key insights visible:**
- Major jumps: v1→v2 (+27.9%), v2→v3 (+17.8%)
- Smaller refinements: v3→v4 (+3.5%), v4→v5 (+2.1%)
- Diminishing returns as recall approaches ceiling
- Term normalization: no additional recall gain

## How to Regenerate Charts

### Prerequisites
```bash
pip install matplotlib seaborn numpy
```

### Generate All Charts
```bash
cd /path/to/autogkb-benchmark

# Generate comprehensive comparison
python src/experiments/variant_finding/create_comparison_charts.py

# Generate evolution chart
python src/experiments/variant_finding/create_evolution_chart.py
```

## Methods Compared

### Regex Methods (Systematic Pattern-Based)
1. **v1** - Baseline: Methods & Conclusions sections only
2. **v2** - Full article text + format normalization
3. **v3** - Context-aware star alleles (gene proximity)
4. **v4** - SNP expansion (516G>T → rs3745274)
5. **v5** - BioC supplement extraction

### Hybrid Methods (Regex + Term Normalization/LLM)
6. **Term Norm Hybrid** - v5 + post-extraction VariantLookup normalization
7. **Regex + Claude Filter** - v5 + Claude Sonnet 4.5 false positive filtering
8. **Regex + GPT-4o Filter** - v5 + GPT-4o false positive filtering

### LLM-Only Methods (Pure Language Model)
9. **LLM-Only (GPT-4o)** - Direct extraction with GPT-4o
10. **LLM-Only (Sonnet 4.5)** - Direct extraction with Claude Sonnet 4.5
11. **LLM-Only (Opus 4.5)** - Direct extraction with Claude Opus 4.5

## Key Findings from Visualizations

### 🏆 Winner: Regex v5
- **Highest recall**: 93.4%
- **Most consistent**: 25/32 articles perfect
- **Fastest**: ~2 minutes
- **Cost-effective**: No API costs

### 📈 Major Breakthroughs

1. **v1 → v2 (+27.9%)**: Expanding from sections to full text
   - Lesson: Scope matters more than pattern sophistication

2. **v2 → v3 (+17.8%)**: Context-aware star alleles
   - Lesson: Standalone patterns need gene context

3. **v3 → v4 (+3.5%)**: SNP expansion
   - Lesson: Pre-extraction text enrichment works

4. **v4 → v5 (+2.1%)**: BioC supplements
   - Lesson: Multiple data sources help

### 🔍 Why Term Normalization Didn't Help (+0.0%)

Post-extraction fuzzy matching:
- ❌ Can't recover missed variants (never extracted)
- ❌ Normalizes typos that match benchmark typos
- ✅ Validates against PharmGKB (data quality value)
- ✅ Provides mapping trail (transparency)

### 🤖 LLM Performance Gap

LLMs lag behind regex by **~22 percentage points** (71% vs 93%):
- Struggle with structured data (tables)
- Miss variants in supplements
- Variable across similar patterns
- Better precision (46% vs 42%) but lower recall

### 📊 Trade-off Visualization

The scatter plot (Panel 4) reveals:
- **No free lunch**: High recall OR high precision, rarely both
- **Regex philosophy**: Maximize recall, accept moderate precision
- **LLM philosophy**: Balanced recall/precision
- **Optimal zone**: Upper right (hard to reach)

## Benchmark Context

**Dataset**: 32 pharmacogenomics articles from PubMed Central
**Total variants**: 322 (avg 10 per article)
**Variant types**: rsIDs, star alleles, HLA alleles, SNP notations
**Evaluation**: Exact string matching against expert-curated gold standard

## Related Documentation

- **METHODS_COMPARISON.md** - Detailed text analysis of all methods
- **regex_variants/misses_analysis_v5.md** - Deep dive into v5 misses
- **regex_term_norm/results_analysis.md** - Term normalization analysis
- **regex_term_norm/SUMMARY.md** - Executive summary of hybrid approach

## Citation

If using these visualizations or insights, please cite:
```
AutoGKB Benchmark - Variant Extraction Methods Comparison
https://github.com/your-repo/autogkb-benchmark
32 pharmacogenomics articles, 322 variants, 11 methods compared
```

---

*Charts generated from experimental results - January 2026*
