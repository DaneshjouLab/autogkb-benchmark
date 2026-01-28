"""
PubTator-based variant extraction experiment.

Extract variants from articles using the PubTator3 API and evaluate coverage.

The pipeline works as follows:
1. Load the benchmark dataset, which includes ground-truth variants for a list of articles.
2. For each article, query the PubTator3 API using the PMID to get annotations.
3. Extract variant/mutation annotations from the BioC JSON response.
4. Compare the extracted variants against the ground-truth annotations to calculate
   recall (coverage) and precision.
5. Aggregate the results across all articles to compute the average recall and
   precision of the PubTator-based approach.
6. Save the detailed per-article and summary results to a JSON file for analysis.

PubTator3 API: https://www.ncbi.nlm.nih.gov/research/pubtator3/
Rate limit: 3 requests per second
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

from src.benchmark_v2.variant_bench import (
    load_variant_bench_data,
    score_variants,
    load_pmcid_title,
)

# PubTator3 API endpoint for exporting annotations
PUBTATOR_API_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

# Rate limit: 3 requests per second, so wait at least 0.35 seconds between requests
REQUEST_DELAY = 0.35


def load_pmid_mapping() -> dict[str, str]:
    """Load the PMCID to PMID mapping from the benchmark data.

    Returns:
        dict mapping PMCID to PMID
    """
    data_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "benchmark_v2"
        / "variant_bench.jsonl"
    )
    pmcid_to_pmid: dict[str, str] = {}

    with open(data_path) as f:
        for line in f:
            record = json.loads(line)
            pmcid_to_pmid[record["pmcid"]] = record["pmid"]

    return pmcid_to_pmid


def fetch_pubtator_annotations(pmid: str, full_text: bool = True) -> dict | None:
    """Fetch annotations from PubTator3 API for a given PMID.

    Args:
        pmid: PubMed ID of the article
        full_text: Whether to request full text annotations (default: True)

    Returns:
        BioC JSON response dict or None if request failed
    """
    params = {"pmids": pmid}
    if full_text:
        params["full"] = "true"

    try:
        response = requests.get(PUBTATOR_API_URL, params=params, timeout=30)
        response.raise_for_status()

        # The API returns a JSON object with the annotations
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch PubTator annotations for PMID {pmid}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse PubTator response for PMID {pmid}: {e}")
        return None


def extract_variants_from_biocjson(biocjson: dict) -> list[str]:
    """Extract variant identifiers from BioC JSON response.

    PubTator annotates variants with type "Variant".
    The rsID can be found in infons.rsid or infons.rsids fields.

    Args:
        biocjson: BioC JSON response from PubTator API

    Returns:
        List of variant identifiers (e.g., rsIDs, star alleles)
    """
    variants = set()

    # BioC JSON structure: PubTator returns either a dict with a document
    # or a list of documents
    documents = []
    if isinstance(biocjson, dict):
        if "PubTator3" in biocjson:
            documents = biocjson.get("PubTator3", [])
        elif "passages" in biocjson:
            documents = [biocjson]
        else:
            documents = [biocjson]
    elif isinstance(biocjson, list):
        documents = biocjson

    for doc in documents:
        # Get passages from the document
        passages = doc.get("passages", [])

        for passage in passages:
            annotations = passage.get("annotations", [])

            for annotation in annotations:
                infons = annotation.get("infons", {})
                ann_type = infons.get("type", "")

                # Check if this is a variant/mutation annotation
                if ann_type.lower() in ["mutation", "variant", "snp", "dnamutation"]:
                    # Primary: Get rsID from rsid field (normalized by PubTator)
                    rsid = infons.get("rsid", "")
                    if rsid:
                        variants.add(rsid)
                        continue

                    # Also check rsids list
                    rsids = infons.get("rsids", [])
                    if rsids:
                        for rs in rsids:
                            if rs:
                                variants.add(rs)
                        continue

                    # Fallback: Get text mention for star alleles, HLA alleles, etc.
                    text_mention = annotation.get("text", "")
                    if text_mention:
                        # Check if it looks like a variant (not just a position like "1639G > A")
                        text_clean = text_mention.strip()
                        # Include star alleles (CYP2C9*3), HLA alleles, etc.
                        if "*" in text_clean or text_clean.lower().startswith("hla-"):
                            variants.add(text_clean)
                        # Include rsIDs that might be in text
                        elif text_clean.lower().startswith("rs"):
                            variants.add(text_clean)

    return list(variants)


def normalize_variant(variant: str) -> str:
    """Normalize variant format for comparison.

    Args:
        variant: Raw variant string

    Returns:
        Normalized variant string
    """
    v = variant.strip()

    # Handle tmVar rs format: "rs#9923231" -> "rs9923231"
    if v.startswith("rs#"):
        v = v.replace("rs#", "rs")

    # Remove leading/trailing whitespace and convert to lowercase for comparison
    return v.lower()


def run_experiment():
    """Run the PubTator variant extraction experiment on all benchmark articles."""
    # Load benchmark data and PMID mapping
    benchmark_data = load_variant_bench_data()
    pmcid_to_pmid = load_pmid_mapping()
    pmcids = list(benchmark_data.keys())

    print(f"Running PubTator extraction on {len(pmcids)} articles...\n")

    results = {
        "run_name": "pubtator_api_v1",
        "method": "PubTator3 API",
        "api_url": PUBTATOR_API_URL,
        "timestamp": datetime.now().isoformat(),
        "full_text": True,
    }

    total_match_rate = 0
    total_precision = 0
    per_article_results = []
    api_failures = []

    for i, pmcid in enumerate(pmcids):
        pmid = pmcid_to_pmid.get(pmcid)
        if not pmid:
            print(f"  {pmcid}: No PMID found (skipping)")
            continue

        # Respect rate limit
        if i > 0:
            time.sleep(REQUEST_DELAY)

        # Fetch annotations from PubTator
        print(f"  [{i + 1}/{len(pmcids)}] Fetching {pmcid} (PMID: {pmid})...", end=" ")
        biocjson = fetch_pubtator_annotations(pmid)

        if biocjson is None:
            print("API request failed")
            api_failures.append(pmcid)
            continue

        # Extract variants from the response
        extracted_variants = extract_variants_from_biocjson(biocjson)

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
                "pmid": pmid,
                "recall": result.match_rate,
                "precision": precision,
                "true_count": len(true_variants),
                "extracted_count": len(extracted_variants),
                "matches": result.matches,
                "misses": result.misses,
                "extras": result.extras,
                "raw_extracted": extracted_variants,
            }
        )

        # Print summary
        status = (
            "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        )
        print(
            f"{status} recall={result.match_rate:.0%} precision={precision:.0%} "
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
    results["api_failures"] = api_failures
    results["per_article_results"] = per_article_results

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Articles scored: {n}")
    print(f"API failures: {len(api_failures)}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    # Count perfect recalls
    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls / n:.0%})")

    # Save results
    output_path = Path(__file__).parent / "results" / "pubtator_api_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_experiment_abstract_only():
    """Run the PubTator variant extraction using abstract-only annotations."""
    # Load benchmark data and PMID mapping
    benchmark_data = load_variant_bench_data()
    pmcid_to_pmid = load_pmid_mapping()
    pmcids = list(benchmark_data.keys())

    print(f"Running PubTator extraction (abstract only) on {len(pmcids)} articles...\n")

    results = {
        "run_name": "pubtator_api_abstract_only",
        "method": "PubTator3 API (Abstract Only)",
        "api_url": PUBTATOR_API_URL,
        "timestamp": datetime.now().isoformat(),
        "full_text": False,
    }

    total_match_rate = 0
    total_precision = 0
    per_article_results = []
    api_failures = []

    for i, pmcid in enumerate(pmcids):
        pmid = pmcid_to_pmid.get(pmcid)
        if not pmid:
            print(f"  {pmcid}: No PMID found (skipping)")
            continue

        # Respect rate limit
        if i > 0:
            time.sleep(REQUEST_DELAY)

        # Fetch annotations from PubTator (abstract only)
        print(f"  [{i + 1}/{len(pmcids)}] Fetching {pmcid} (PMID: {pmid})...", end=" ")
        biocjson = fetch_pubtator_annotations(pmid, full_text=False)

        if biocjson is None:
            print("API request failed")
            api_failures.append(pmcid)
            continue

        # Extract variants from the response
        extracted_variants = extract_variants_from_biocjson(biocjson)

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
                "pmid": pmid,
                "recall": result.match_rate,
                "precision": precision,
                "true_count": len(true_variants),
                "extracted_count": len(extracted_variants),
                "matches": result.matches,
                "misses": result.misses,
                "extras": result.extras,
                "raw_extracted": extracted_variants,
            }
        )

        # Print summary
        status = (
            "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        )
        print(
            f"{status} recall={result.match_rate:.0%} precision={precision:.0%} "
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
    results["api_failures"] = api_failures
    results["per_article_results"] = per_article_results

    print(f"\n{'=' * 60}")
    print("SUMMARY (Abstract Only)")
    print(f"{'=' * 60}")
    print(f"Articles scored: {n}")
    print(f"API failures: {len(api_failures)}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    # Count perfect recalls
    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls / n:.0%})")

    # Save results
    output_path = Path(__file__).parent / "results" / "pubtator_api_abstract_only.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--abstract-only":
        run_experiment_abstract_only()
    else:
        run_experiment()
