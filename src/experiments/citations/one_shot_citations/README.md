# One Shot Citations

A pipeline to find supporting citations for pharmacogenomic association sentences from scientific articles and evaluate their quality using LLM-as-a-judge.

## Overview

This experiment consists of two main components:

1. **Citation Finding** (`one_shot_citations.py`): Takes pharmacogenomic association sentences and finds 3-5 supporting sentences/tables from the source articles
2. **Citation Evaluation** (`citation_judge.py`): Evaluates how well the citations support the claims using an LLM judge

## Quick Start

### Basic Usage

Run with default settings (1 PMCID, gpt-4o model, with evaluation):
```bash
python one_shot_citations.py
```

### Common Commands

Process 3 PMCIDs with gpt-4o-mini:
```bash
python one_shot_citations.py --num-pmcids 3 --model gpt-4o-mini
```

Skip automatic evaluation:
```bash
python one_shot_citations.py --no-eval
```

Use a different judge model:
```bash
python one_shot_citations.py --judge-model claude-sonnet-4-20250514
```

Use the v2 prompt (with explanation context):
```bash
python one_shot_citations.py --prompt v2
```

## Command-Line Arguments

### one_shot_citations.py

- `--model`: Model for citation finding (default: `gpt-4o`)
- `--prompt`: Prompt version from prompts.yaml (default: `v1`)
- `--num-pmcids`: Number of PMCIDs to process (default: `1`)
- `--judge-model`: Model for evaluation (default: `claude-3-haiku-20240307`)
- `--no-eval`: Skip automatic evaluation

### citation_judge.py (standalone)

- `--citations`: Path to citations JSON file (required)
- `--sentence-bench`: Path to sentence_bench.jsonl (default: auto-detected)
- `--judge-model`: Model to use for judging (default: `claude-3-haiku-20240307`)
- `--output`: Path to save evaluation results (required)

## Prompts

Prompts are configured in `prompts.yaml`:

- **v1**: Basic citation finding - finds 3-5 supporting sentences/tables
- **v2**: Citation finding with explanation context - uses explanations to guide search

## Output Files

### Citation Output (`outputs/`)

Format: `citations_{model}_{prompt}_{timestamp}.json`

Structure:
```json
{
  "PMC5508045": {
    "rs9923231": [
      "Citation sentence 1 from article...",
      "Citation sentence 2 from article...",
      "## Table 2: Association between..."
    ],
    "rs1057910": [...]
  }
}
```

### Evaluation Results (`results/`)

Format: `citation_scores_{judge_model}_{timestamp}.json`

Structure:
```json
{
  "overall_avg_score": 82.5,
  "num_pmcids": 1,
  "num_total_variants": 4,
  "per_pmcid": [
    {
      "pmcid": "PMC5508045",
      "num_variants": 4,
      "avg_score": 82.5,
      "scores": {
        "rs9923231": {
          "score": 90.0,
          "justification": "Citations provide strong evidence..."
        }
      }
    }
  ]
}
```

## Evaluation Scoring

Citations are scored 0-100 based on:
- **Relevance**: Do citations directly relate to the claim?
- **Support**: Do citations provide evidence for the specific claim?
- **Completeness**: Do citations include statistical evidence, sample sizes, effect direction?
- **Quality**: Are citations from appropriate sections (Results, Methods, Tables)?

### Score Ranges
- 90-100: Excellent - Strong support with statistical evidence
- 70-89: Good - Reasonable supporting evidence
- 50-69: Fair - Related but lacking key details
- 30-49: Poor - Tangentially related
- 0-29: Very Poor - Irrelevant or contradictory

## Examples

### Example 1: Basic Run
```bash
python one_shot_citations.py --num-pmcids 1
```

Output:
```
✓ rs9923231: 3 citation(s) found
✓ rs1057910: 3 citation(s) found
Overall Average Score: 82.500
```

### Example 2: Larger Batch
```bash
python one_shot_citations.py --num-pmcids 5 --model gpt-4o-mini
```

### Example 3: Standalone Evaluation
```bash
python citation_judge.py \
  --citations outputs/citations_gpt-4o_v1_20260120_104052.json \
  --judge-model claude-3-haiku-20240307 \
  --output results/my_evaluation.json
```

## Dependencies

- litellm
- loguru
- PyYAML
- python-dotenv

## Data Sources

- **Input**: `data/benchmark_v2/sentence_bench.jsonl` - Pharmacogenomic association sentences
- **Articles**: `data/articles/{pmcid}.md` - Article markdown files

## Notes

- Citations can include exact sentences from the article or table headers (e.g., "## Table 2: ...")
- The pipeline processes all variants for a PMCID in a single LLM call for efficiency
- Evaluation is done per-PMCID in batch mode for consistency
- Table references are preserved in the format they appear in markdown (with ## prefix)
