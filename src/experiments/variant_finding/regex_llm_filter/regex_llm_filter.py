"""
Regex + LLM Filter Experiment

This experiment combines regex-based variant extraction (V5) with LLM-based filtering
to remove false positives. The LLM filters out variants that are only mentioned for
context rather than actually being studied in the article.

Usage:
    python regex_llm_filter.py --model claude-3-5-sonnet --prompt v1
    python regex_llm_filter.py --model gpt-4o --prompt v2 --max-articles 5
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Import V5 extraction functions
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.experiments.variant_finding.regex_variants.extract_variants_v5 import (
    extract_all_variants,
    get_combined_text,
    get_snp_expander,
)
from src.experiments.utils import call_llm, get_markdown_text
from src.benchmark_v2.variant_bench import load_variant_bench_data, score_variants

# Load environment variables
load_dotenv()


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def extract_json_array(text: str) -> list[str]:
    """
    Extract JSON array from LLM response.

    Handles various formats:
    - Pure JSON array: ["rs9923231", "CYP2C9*2"]
    - JSON in markdown code block: ```json\n["rs9923231"]\n```
    - JSON with explanation text before/after

    Args:
        text: LLM response text

    Returns:
        List of variant strings
    """
    # Try to find JSON array in the response
    # First try to extract from code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Try to find JSON array anywhere in the text
        json_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # No JSON array found
            return []

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return [str(v).strip() for v in result]
        return []
    except json.JSONDecodeError:
        print(f"    ⚠️  Failed to parse JSON from LLM response")
        return []


def filter_variants_with_llm(
    model: str,
    variants: list[str],
    article_text: str,
    prompt_config: dict,
) -> list[str]:
    """
    Filter variants using an LLM to remove false positives.

    Args:
        model: LLM model to use
        variants: List of variants extracted by regex
        article_text: Full article text
        prompt_config: Prompt configuration dict with 'system' and 'user' keys

    Returns:
        Filtered list of variants that are actually being studied
    """
    if not variants:
        return []

    # Format variants as a list
    variants_list = "\n".join([f"- {v}" for v in variants])

    # Format prompts
    system_prompt = prompt_config["system"]
    user_prompt = prompt_config["user"].format(
        variants_list=variants_list,
        article_text=article_text
    )

    # Call LLM
    response = call_llm(model, system_prompt, user_prompt)

    # Extract filtered variants
    filtered_variants = extract_json_array(response)

    return filtered_variants


def run_experiment(
    model: str,
    prompt_version: str,
    max_articles: int | None = None,
) -> dict:
    """
    Run the regex + LLM filter experiment.

    Args:
        model: LLM model identifier
        prompt_version: Prompt version to use (e.g., "v1", "v2")
        max_articles: Maximum number of articles to process (None = all)

    Returns:
        Results dictionary
    """
    # Initialize SNP expander (needed for V5 extraction)
    print("Initializing SNP expander...")
    expander = get_snp_expander()
    stats = expander.stats()
    print(f"  Loaded {stats['total_mappings']} SNP notation mappings")
    print(f"  Covering {stats['unique_rsids']} unique rsIDs across {len(stats['genes'])} genes\n")

    # Load prompts
    prompts = load_prompts()
    if prompt_version not in prompts:
        available = ", ".join(prompts.keys())
        raise ValueError(f"Prompt version '{prompt_version}' not found. Available: {available}")

    prompt_config = prompts[prompt_version]
    print(f"Using prompt: {prompt_config['name']}")
    print(f"Using model: {model}\n")

    # Load benchmark data
    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())

    if max_articles:
        pmcids = pmcids[:max_articles]

    print(f"Processing {len(pmcids)} articles...\n")

    # Track results
    per_article_results = []
    total_recall_regex = 0
    total_precision_regex = 0
    total_recall_filtered = 0
    total_precision_filtered = 0

    for i, pmcid in enumerate(pmcids, 1):
        print(f"[{i}/{len(pmcids)}] Processing {pmcid}...")

        # Get article text (using combined article + supplement like V5)
        combined_text, supplement_text = get_combined_text(pmcid)

        if not combined_text:
            print(f"  ⚠️  No text found (skipping)\n")
            continue

        # Step 1: Extract variants using V5 regex
        regex_variants = extract_all_variants(combined_text)

        # Step 2: Filter with LLM
        try:
            filtered_variants = filter_variants_with_llm(
                model=model,
                variants=regex_variants,
                article_text=combined_text,
                prompt_config=prompt_config,
            )
        except Exception as e:
            print(f"  ⚠️  LLM call failed: {e}")
            print("  Ending experiment early due to LLM call failure\n")
            break

        # Step 3: Score both regex and filtered results
        true_variants = benchmark_data[pmcid]

        # Score regex extraction
        regex_result = score_variants(regex_variants, true_variants, pmcid)
        regex_precision = (
            len(regex_result.matches) / len(regex_variants) if regex_variants else 1.0
        )

        # Score filtered results
        filtered_result = score_variants(filtered_variants, true_variants, pmcid)
        filtered_precision = (
            len(filtered_result.matches) / len(filtered_variants) if filtered_variants else 1.0
        )

        # Track totals
        total_recall_regex += regex_result.match_rate
        total_precision_regex += regex_precision
        total_recall_filtered += filtered_result.match_rate
        total_precision_filtered += filtered_precision

        # Calculate how many false positives were removed
        removed_variants = set(regex_variants) - set(filtered_variants)
        removed_true_positives = set(regex_result.matches) - set(filtered_result.matches)
        removed_false_positives = removed_variants - removed_true_positives

        # Store results
        per_article_results.append({
            "pmcid": pmcid,
            "true_count": len(true_variants),
            "regex_extracted_count": len(regex_variants),
            "filtered_extracted_count": len(filtered_variants),
            "regex_recall": regex_result.match_rate,
            "regex_precision": regex_precision,
            "filtered_recall": filtered_result.match_rate,
            "filtered_precision": filtered_precision,
            "regex_matches": regex_result.matches,
            "regex_misses": regex_result.misses,
            "regex_extras": regex_result.extras,
            "filtered_matches": filtered_result.matches,
            "filtered_misses": filtered_result.misses,
            "filtered_extras": filtered_result.extras,
            "removed_total": len(removed_variants),
            "removed_false_positives": list(removed_false_positives),
            "removed_true_positives": list(removed_true_positives),
        })

        # Print results
        regex_status = "✓" if regex_result.match_rate == 1.0 else "○" if regex_result.match_rate > 0 else "✗"
        filtered_status = "✓" if filtered_result.match_rate == 1.0 else "○" if filtered_result.match_rate > 0 else "✗"

        print(f"  Regex    {regex_status}: recall={regex_result.match_rate:.0%} precision={regex_precision:.0%} "
              f"(found {len(regex_result.matches)}/{len(true_variants)}, extras={len(regex_result.extras)})")
        print(f"  Filtered {filtered_status}: recall={filtered_result.match_rate:.0%} precision={filtered_precision:.0%} "
              f"(found {len(filtered_result.matches)}/{len(true_variants)}, extras={len(filtered_result.extras)})")

        # Show what was removed
        if removed_false_positives:
            print(f"  ✓ Removed {len(removed_false_positives)} false positives: {list(removed_false_positives)[:5]}")
        if removed_true_positives:
            print(f"  ✗ Incorrectly removed {len(removed_true_positives)} true positives: {list(removed_true_positives)}")

        print()

    # Calculate averages
    n = len(per_article_results)
    if n == 0:
        print("No articles processed successfully.")
        return {}

    avg_recall_regex = total_recall_regex / n
    avg_precision_regex = total_precision_regex / n
    avg_recall_filtered = total_recall_filtered / n
    avg_precision_filtered = total_precision_filtered / n

    # Calculate improvement
    recall_change = avg_recall_filtered - avg_recall_regex
    precision_change = avg_precision_filtered - avg_precision_regex

    # Count perfect recalls
    perfect_recalls_regex = sum(1 for r in per_article_results if r["regex_recall"] == 1.0)
    perfect_recalls_filtered = sum(1 for r in per_article_results if r["filtered_recall"] == 1.0)

    # Build results
    results = {
        "model": model,
        "prompt_version": prompt_version,
        "prompt_name": prompt_config["name"],
        "timestamp": datetime.now().isoformat(),
        "articles_processed": n,
        "regex_results": {
            "avg_recall": avg_recall_regex,
            "avg_precision": avg_precision_regex,
            "perfect_recalls": perfect_recalls_regex,
        },
        "filtered_results": {
            "avg_recall": avg_recall_filtered,
            "avg_precision": avg_precision_filtered,
            "perfect_recalls": perfect_recalls_filtered,
        },
        "improvement": {
            "recall_change": recall_change,
            "precision_change": precision_change,
        },
        "per_article_results": per_article_results,
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {model} - {prompt_version}")
    print(f"{'=' * 70}")
    print(f"Articles processed: {n}")
    print()
    print(f"REGEX EXTRACTION (V5 baseline):")
    print(f"  Average Recall:    {avg_recall_regex:.1%}")
    print(f"  Average Precision: {avg_precision_regex:.1%}")
    print(f"  Perfect recall:    {perfect_recalls_regex}/{n} ({perfect_recalls_regex/n:.0%})")
    print()
    print(f"REGEX + LLM FILTER:")
    print(f"  Average Recall:    {avg_recall_filtered:.1%} ({recall_change:+.1%})")
    print(f"  Average Precision: {avg_precision_filtered:.1%} ({precision_change:+.1%})")
    print(f"  Perfect recall:    {perfect_recalls_filtered}/{n} ({perfect_recalls_filtered/n:.0%})")
    print()

    if precision_change > 0:
        print(f"✓ LLM filtering IMPROVED precision by {precision_change:.1%}")
    else:
        print(f"✗ LLM filtering reduced precision by {precision_change:.1%}")

    if recall_change < 0:
        print(f"⚠️  LLM filtering reduced recall by {abs(recall_change):.1%}")
    elif recall_change > 0:
        print(f"✓ LLM filtering improved recall by {recall_change:.1%}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / f"{model.replace('/', '_')}_{prompt_version}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regex + LLM Filter Experiment for Variant Extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (e.g., claude-3-5-sonnet-20241022, gpt-4o, gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="v1",
        help="Prompt version to use (v1, v2, v3, etc.)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all)",
    )

    args = parser.parse_args()

    run_experiment(
        model=args.model,
        prompt_version=args.prompt,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
