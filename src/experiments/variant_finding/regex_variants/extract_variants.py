"""
Regex-based variant extraction experiment.

Extract variants from article markdown using regex patterns and evaluate coverage.

The pipeline works as follows:
1. Load the benchmark dataset, which includes ground-truth variants for a list of articles.
2. For each article, read the full text from the corresponding markdown file.
3. Reduce the text to only the "Methods" and "Conclusions" sections, which are most
   likely to contain variant information.
4. Apply a set of regular expressions to the filtered text to extract potential
   variant mentions, including rsIDs, star alleles (e.g., CYP2C9*3), and HLA alleles.
5. Compare the extracted variants against the ground-truth annotations to calculate
   recall (coverage) and precision.
6. Aggregate the results across all articles to compute the average recall and
   precision of the regex-based approach.
7. Save the detailed per-article and summary results to a JSON file for analysis.
"""

import json
import re
from pathlib import Path

from src.experiments.utils import get_methods_and_conclusions_text
from src.benchmark_v2.variant_bench import load_variant_bench_data, score_variants


# Regex patterns for different variant types
VARIANT_PATTERNS = [
    # rsIDs - e.g., rs9923231, rs887829
    r"\brs\d{4,}\b",
    # Star alleles - e.g., CYP2C9*3, CYP2B6*1, CYP2C19*17
    # Gene names: CYP followed by alphanumeric, then *number
    r"\b(CYP\w+)\*(\d+)\b",
    # HLA alleles - e.g., HLA-B*58:01, HLA-DRB1*03:01
    r"\bHLA-[A-Z]+\d*\*\d+:\d+\b",
]


def extract_rsids(text: str) -> list[str]:
    """Extract rsID variants from text."""
    pattern = r"\brs\d{4,}\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches))


def extract_star_alleles(text: str) -> list[str]:
    """Extract star allele variants (e.g., CYP2C9*3) from text."""
    pattern = r"\b(CYP\w+)\*(\d+)\b"
    matches = re.findall(pattern, text)
    # Reconstruct as Gene*Number format
    variants = [f"{gene}*{number}" for gene, number in matches]
    return list(set(variants))


def extract_hla_alleles(text: str) -> list[str]:
    """Extract HLA allele variants from text."""
    pattern = r"\bHLA-[A-Z]+\d*\*\d+:\d+\b"
    matches = re.findall(pattern, text)
    return list(set(matches))


def extract_all_variants(text: str) -> list[str]:
    """Extract all variant types from text."""
    variants = []
    variants.extend(extract_rsids(text))
    variants.extend(extract_star_alleles(text))
    variants.extend(extract_hla_alleles(text))
    return list(set(variants))


def run_experiment():
    """Run the regex variant extraction experiment on all benchmark articles."""
    # Load benchmark data
    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())

    print(f"Running regex extraction on {len(pmcids)} articles...\n")

    results = {
        "run_name": "regex_extraction_v1",
        "patterns_used": [
            "rsIDs: r'\\brs\\d{4,}\\b'",
            "Star alleles: r'\\b(CYP\\w+)\\*(\\d+)\\b'",
            "HLA alleles: r'\\bHLA-[A-Z]+\\d*\\*\\d+:\\d+\\b'",
        ],
    }

    total_match_rate = 0
    total_precision = 0
    per_article_results = []

    for pmcid in pmcids:
        # Get methods and conclusions text
        text = get_methods_and_conclusions_text(pmcid)

        if not text:
            print(f"  {pmcid}: No text found (skipping)")
            continue

        # Extract variants using regex
        extracted_variants = extract_all_variants(text)

        # Score against ground truth
        true_variants = benchmark_data[pmcid]
        result = score_variants(extracted_variants, true_variants, pmcid)

        # Calculate precision
        if len(extracted_variants) > 0:
            precision = len(result.matches) / len(extracted_variants)
        else:
            precision = 1.0 if len(true_variants) == 0 else 0.0

        total_match_rate += result.match_rate
        total_precision += precision

        per_article_results.append(
            {
                "pmcid": pmcid,
                "recall": result.match_rate,
                "precision": precision,
                "true_count": len(true_variants),
                "extracted_count": len(extracted_variants),
                "matches": result.matches,
                "misses": result.misses,
                "extras": result.extras,
            }
        )

        # Print summary
        status = (
            "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        )
        print(
            f"  {status} {pmcid}: recall={result.match_rate:.0%} precision={precision:.0%} "
            f"(found {len(result.matches)}/{len(true_variants)}, extras={len(result.extras)})"
        )

        if result.misses:
            print(f"      Missed: {result.misses}")

    # Calculate aggregates
    n = len(per_article_results)
    avg_recall = total_match_rate / n if n > 0 else 0
    avg_precision = total_precision / n if n > 0 else 0

    results["avg_recall"] = avg_recall
    results["avg_precision"] = avg_precision
    results["articles_scored"] = n
    results["per_article_results"] = per_article_results

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Articles scored: {n}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    # Count perfect recalls
    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls / n:.0%})")

    # Save results
    output_path = Path(__file__).parent / "results_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
