"""
Custom Normalization Experiment

This experiment:
1. Extracts raw variants from benchmark papers using v5 regex extraction
2. Runs our custom normalizer (TermLookup) on the extracted variants
3. Saves mappings with details (original term, normalized term, data source, confidence)
4. Generates summary statistics
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from src.benchmark_v2.variant_bench import load_variant_bench_data
from src.experiments.utils import get_markdown_text
from src.experiments.utils_bioc import fetch_bioc_supplement
from src.term_normalization.term_lookup import TermLookup, TermType

# Import extraction functions from v5
from src.experiments.variant_finding.regex_variants.extract_variants_v5 import (
    extract_rsids,
    extract_star_alleles,
    extract_hla_alleles,
    extract_snp_notations,
)

OUTPUT_DIR = Path(__file__).parent


def extract_all_raw_variants(text: str) -> dict[str, list[str]]:
    """Extract all variant types from text, categorized by type."""
    return {
        "rsids": list(set(extract_rsids(text))),
        "star_alleles": list(set(extract_star_alleles(text))),
        "hla_alleles": list(set(extract_hla_alleles(text))),
        "snp_notations": list(set(extract_snp_notations(text))),  # These are already rsIDs from expansion
    }


def get_combined_text(pmcid: str) -> str:
    """Get combined article + supplement text for extraction."""
    article_text = get_markdown_text(pmcid)
    supplement_text = fetch_bioc_supplement(pmcid, use_cache=True)

    if supplement_text:
        combined_text = (
            article_text + "\n\n--- SUPPLEMENTARY MATERIAL ---\n\n" + supplement_text
        )
    else:
        combined_text = article_text

    return combined_text


def run_experiment():
    """Run the custom normalization experiment."""
    print("=" * 60)
    print("CUSTOM NORMALIZATION EXPERIMENT")
    print("=" * 60)

    # Initialize the term lookup
    print("\nInitializing TermLookup...")
    term_lookup = TermLookup(enable_snp_expansion=True)
    print("  TermLookup initialized successfully")

    # Load benchmark data
    print("\nLoading benchmark data...")
    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())
    print(f"  Loaded {len(pmcids)} articles from benchmark")

    # Step 1: Extract raw variants from all papers
    print("\n" + "=" * 60)
    print("STEP 1: Extracting raw variants from benchmark papers")
    print("=" * 60)

    all_raw_variants = []
    per_paper_variants = {}
    variant_sources = defaultdict(list)  # Track which papers each variant comes from

    for pmcid in pmcids:
        text = get_combined_text(pmcid)
        if not text:
            print(f"  {pmcid}: No text found (skipping)")
            continue

        variants_by_type = extract_all_raw_variants(text)

        # Flatten all variants for this paper
        paper_variants = []
        for variant_type, variants in variants_by_type.items():
            paper_variants.extend(variants)
            for v in variants:
                variant_sources[v].append(pmcid)

        per_paper_variants[pmcid] = {
            "by_type": variants_by_type,
            "all": list(set(paper_variants)),
            "ground_truth": benchmark_data[pmcid],
        }
        all_raw_variants.extend(paper_variants)

        total = len(set(paper_variants))
        print(f"  {pmcid}: {total} variants extracted")

    # Get unique variants
    unique_variants = list(set(all_raw_variants))
    print(f"\n  Total unique variants extracted: {len(unique_variants)}")

    # Save raw variants to file
    raw_variants_path = OUTPUT_DIR / "raw_variants.txt"
    with open(raw_variants_path, "w") as f:
        for v in sorted(unique_variants):
            f.write(f"{v}\n")
    print(f"  Saved to {raw_variants_path}")

    # Step 2: Run custom normalizer on variants
    print("\n" + "=" * 60)
    print("STEP 2: Running custom normalizer on extracted variants")
    print("=" * 60)

    normalization_results = []
    normalized_count = 0
    failed_count = 0

    for i, variant in enumerate(sorted(unique_variants)):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(unique_variants)}...")

        result = term_lookup.search(variant, term_type=TermType.VARIANT, threshold=0.8, top_k=1)

        if result and len(result) > 0:
            normalized_count += 1
            best_match = result[0]
            normalization_results.append({
                "raw_input": variant,
                "normalized_term": best_match.normalized_term,
                "id": best_match.id,
                "url": best_match.url,
                "score": best_match.score,
                "source_papers": variant_sources[variant],
                "status": "normalized",
            })
        else:
            failed_count += 1
            normalization_results.append({
                "raw_input": variant,
                "normalized_term": None,
                "id": None,
                "url": None,
                "score": 0.0,
                "source_papers": variant_sources[variant],
                "status": "not_found",
            })

    print(f"\n  Normalized: {normalized_count}/{len(unique_variants)} ({normalized_count/len(unique_variants)*100:.1f}%)")
    print(f"  Not found: {failed_count}/{len(unique_variants)} ({failed_count/len(unique_variants)*100:.1f}%)")

    # Step 3: Save mappings with details
    print("\n" + "=" * 60)
    print("STEP 3: Saving normalization mappings")
    print("=" * 60)

    mappings_path = OUTPUT_DIR / "normalization_mappings.json"
    mappings_data = {
        "timestamp": datetime.now().isoformat(),
        "total_variants": len(unique_variants),
        "normalized_count": normalized_count,
        "not_found_count": failed_count,
        "normalization_rate": normalized_count / len(unique_variants) if unique_variants else 0,
        "mappings": normalization_results,
    }

    with open(mappings_path, "w") as f:
        json.dump(mappings_data, f, indent=2)
    print(f"  Saved to {mappings_path}")

    # Step 4: Generate summary statistics
    print("\n" + "=" * 60)
    print("STEP 4: Summary Statistics")
    print("=" * 60)

    # Categorize by variant type
    rsid_count = sum(1 for v in unique_variants if v.lower().startswith("rs"))
    star_allele_count = sum(1 for v in unique_variants if "*" in v and not v.startswith("HLA"))
    hla_count = sum(1 for v in unique_variants if v.upper().startswith("HLA"))

    # Score distribution
    high_confidence = sum(1 for r in normalization_results if r["score"] >= 0.95)
    medium_confidence = sum(1 for r in normalization_results if 0.8 <= r["score"] < 0.95)
    low_confidence = sum(1 for r in normalization_results if 0 < r["score"] < 0.8)

    # Normalization rate by type
    rsid_normalized = sum(1 for r in normalization_results
                         if r["raw_input"].lower().startswith("rs") and r["status"] == "normalized")
    star_normalized = sum(1 for r in normalization_results
                         if "*" in r["raw_input"] and not r["raw_input"].startswith("HLA") and r["status"] == "normalized")
    hla_normalized = sum(1 for r in normalization_results
                        if r["raw_input"].upper().startswith("HLA") and r["status"] == "normalized")

    summary = {
        "experiment": "Custom Normalization Evaluation",
        "timestamp": datetime.now().isoformat(),
        "extraction_summary": {
            "total_papers": len(pmcids),
            "total_unique_variants": len(unique_variants),
            "by_type": {
                "rsids": rsid_count,
                "star_alleles": star_allele_count,
                "hla_alleles": hla_count,
            }
        },
        "normalization_summary": {
            "total_normalized": normalized_count,
            "total_not_found": failed_count,
            "normalization_rate": f"{normalized_count/len(unique_variants)*100:.1f}%",
            "by_type": {
                "rsids": f"{rsid_normalized}/{rsid_count} ({rsid_normalized/rsid_count*100:.1f}%)" if rsid_count else "N/A",
                "star_alleles": f"{star_normalized}/{star_allele_count} ({star_normalized/star_allele_count*100:.1f}%)" if star_allele_count else "N/A",
                "hla_alleles": f"{hla_normalized}/{hla_count} ({hla_normalized/hla_count*100:.1f}%)" if hla_count else "N/A",
            },
            "confidence_distribution": {
                "high (>=0.95)": high_confidence,
                "medium (0.8-0.95)": medium_confidence,
                "low (<0.8)": low_confidence,
                "not_found": failed_count,
            }
        },
        "per_paper_summary": [],
    }

    # Per-paper ground truth comparison
    print("\n  Per-paper ground truth comparison:")
    for pmcid, paper_data in per_paper_variants.items():
        extracted = set(v.lower() for v in paper_data["all"])
        ground_truth = set(v.lower() for v in paper_data["ground_truth"])

        matches = extracted & ground_truth
        missed = ground_truth - extracted
        extras = extracted - ground_truth

        recall = len(matches) / len(ground_truth) if ground_truth else 1.0
        precision = len(matches) / len(extracted) if extracted else 1.0

        summary["per_paper_summary"].append({
            "pmcid": pmcid,
            "extracted_count": len(extracted),
            "ground_truth_count": len(ground_truth),
            "matches": len(matches),
            "recall": f"{recall*100:.1f}%",
            "precision": f"{precision*100:.1f}%",
            "missed": list(missed)[:5],  # First 5 missed
        })

        status = "✓" if recall == 1.0 else "○" if recall > 0 else "✗"
        print(f"    {status} {pmcid}: recall={recall:.0%} ({len(matches)}/{len(ground_truth)})")

    # Calculate averages
    avg_recall = sum(float(p["recall"].rstrip("%")) for p in summary["per_paper_summary"]) / len(summary["per_paper_summary"])
    avg_precision = sum(float(p["precision"].rstrip("%")) for p in summary["per_paper_summary"]) / len(summary["per_paper_summary"])

    summary["overall_metrics"] = {
        "average_recall": f"{avg_recall:.1f}%",
        "average_precision": f"{avg_precision:.1f}%",
    }

    # Save summary
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Papers processed: {len(pmcids)}")
    print(f"  Unique variants extracted: {len(unique_variants)}")
    print(f"  Normalization rate: {normalized_count/len(unique_variants)*100:.1f}%")
    print(f"  Average recall vs ground truth: {avg_recall:.1f}%")
    print(f"  Average precision vs ground truth: {avg_precision:.1f}%")
    print("\n  Confidence distribution:")
    print(f"    High (>=0.95): {high_confidence}")
    print(f"    Medium (0.8-0.95): {medium_confidence}")
    print(f"    Low (<0.8): {low_confidence}")
    print(f"    Not found: {failed_count}")

    # Save list of variants not found for analysis
    not_found_path = OUTPUT_DIR / "variants_not_found.txt"
    not_found = [r["raw_input"] for r in normalization_results if r["status"] == "not_found"]
    with open(not_found_path, "w") as f:
        for v in sorted(not_found):
            f.write(f"{v}\n")
    print(f"\n  Variants not found saved to {not_found_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
