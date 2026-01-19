"""
Just Ask Experiment - LLM-based variant extraction.

Ask various LLMs to extract pharmacogenetic variants from articles and evaluate accuracy.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from litellm import completion
from loguru import logger

load_dotenv()

# Import from sibling modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils import get_methods_and_conclusions_text

# Path to benchmark annotations
BENCHMARK_DIR = Path(__file__).parent.parent.parent.parent / "data" / "benchmark_annotations"
PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def load_prompts() -> dict:
    """Load prompts from yaml file."""
    with open(PROMPTS_FILE) as f:
        return yaml.safe_load(f)


def load_benchmark_variants() -> dict[str, list[str]]:
    """Load variants from benchmark annotation files."""
    pmcid_to_variants: dict[str, list[str]] = {}

    for json_file in BENCHMARK_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        pmcid = json_file.stem
        variants = set()

        for ann in data.get("var_drug_ann") or []:
            if v := ann.get("Variant/Haplotypes"):
                variants.add(v)
        for ann in data.get("var_pheno_ann") or []:
            if v := ann.get("Variant/Haplotypes"):
                variants.add(v)
        for ann in data.get("var_fa_ann") or []:
            if v := ann.get("Variant/Haplotypes"):
                variants.add(v)

        pmcid_to_variants[pmcid] = list(variants)

    return pmcid_to_variants


def extract_json_array(text: str) -> list[str]:
    """Extract JSON array from LLM response."""
    # Try to find JSON array in the response
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response as JSON
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    logger.warning(f"Could not parse JSON from response: {text[:200]}...")
    return []


def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Call LLM using litellm."""
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def score_variants(
    proposed: list[str], true_variants: list[str]
) -> dict:
    """Score proposed variants against ground truth."""
    proposed_set = {v.strip().lower() for v in proposed}

    # Split comma-separated variants in ground truth
    true_set = set()
    for v in true_variants:
        for part in v.split(","):
            true_set.add(part.strip().lower())

    matches = list(proposed_set & true_set)
    extras = list(proposed_set - true_set)
    misses = list(true_set - proposed_set)

    recall = len(matches) / len(true_set) if true_set else 1.0
    precision = len(matches) / len(proposed_set) if proposed_set else (1.0 if not true_set else 0.0)

    return {
        "recall": recall,
        "precision": precision,
        "matches": matches,
        "extras": extras,
        "misses": misses,
        "true_count": len(true_set),
        "proposed_count": len(proposed_set),
    }


def run_experiment(
    model: str,
    prompt_version: str = "v1",
    max_articles: int | None = None,
) -> dict:
    """Run variant extraction experiment on benchmark articles."""
    prompts = load_prompts()
    prompt_config = prompts[prompt_version]
    benchmark_data = load_benchmark_variants()

    pmcids = list(benchmark_data.keys())
    if max_articles:
        pmcids = pmcids[:max_articles]

    print(f"\nRunning experiment: {model} with prompt {prompt_version}")
    print(f"Articles to process: {len(pmcids)}\n")

    results = {
        "model": model,
        "prompt_version": prompt_version,
        "prompt_name": prompt_config["name"],
        "timestamp": datetime.now().isoformat(),
        "per_article_results": [],
    }

    total_recall = 0
    total_precision = 0
    processed = 0

    for pmcid in pmcids:
        # Get article text
        text = get_methods_and_conclusions_text(pmcid)
        if not text:
            print(f"  {pmcid}: No text found (skipping)")
            continue

        # Format prompt
        user_prompt = prompt_config["user"].format(article_text=text)
        system_prompt = prompt_config["system"]

        # Call LLM
        try:
            response = call_llm(model, system_prompt, user_prompt)
            extracted = extract_json_array(response)
        except Exception as e:
            logger.error(f"Error processing {pmcid}: {e}")
            extracted = []

        # Score results
        true_variants = benchmark_data[pmcid]
        scores = score_variants(extracted, true_variants)

        article_result = {
            "pmcid": pmcid,
            **scores,
        }
        results["per_article_results"].append(article_result)

        total_recall += scores["recall"]
        total_precision += scores["precision"]
        processed += 1

        # Print progress
        status = "✓" if scores["recall"] == 1.0 else ("○" if scores["recall"] > 0 else "✗")
        print(
            f"  {status} {pmcid}: recall={scores['recall']:.0%} precision={scores['precision']:.0%} "
            f"(found {len(scores['matches'])}/{scores['true_count']}, extras={len(scores['extras'])})"
        )
        if scores["misses"]:
            print(f"      Missed: {scores['misses']}")

    # Calculate aggregates
    avg_recall = total_recall / processed if processed > 0 else 0
    avg_precision = total_precision / processed if processed > 0 else 0

    results["avg_recall"] = avg_recall
    results["avg_precision"] = avg_precision
    results["articles_processed"] = processed

    # Count perfect recalls
    perfect_recalls = sum(1 for r in results["per_article_results"] if r["recall"] == 1.0)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {model} - {prompt_version}")
    print(f"{'=' * 60}")
    print(f"Articles processed: {processed}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")
    print(f"Perfect recall: {perfect_recalls}/{processed} ({perfect_recalls / processed:.0%})" if processed else "N/A")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_model_name = model.replace("/", "_").replace(":", "_")
    output_path = RESULTS_DIR / f"{safe_model_name}_{prompt_version}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run variant extraction experiment")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Model to use")
    parser.add_argument("--prompt", default="v1", help="Prompt version (v1, v2, v3)")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to process")

    args = parser.parse_args()
    run_experiment(args.model, args.prompt, args.max_articles)
