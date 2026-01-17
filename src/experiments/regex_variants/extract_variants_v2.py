"""
Regex-based variant extraction experiment - Version 2.

Improvements over v1:
- Uses full article text instead of just methods/conclusions
- Handles HLA alleles without HLA- prefix (e.g., B*5801 -> HLA-B*58:01)
- Handles more gene star allele formats (UGT, NUDT, CYP4F2, etc.)
- Normalizes variants for comparison
"""

import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils import get_markdown_text
from benchmark_v2.variant_bench import load_variant_bench_data, score_variants


def normalize_hla(variant: str) -> str:
    """Normalize HLA allele format to HLA-X*XX:XX format."""
    variant = variant.upper()

    # Already normalized
    if re.match(r'HLA-[A-Z]+\d*\*\d+:\d+', variant):
        return variant

    # Handle formats like B*5801 -> HLA-B*58:01
    # or DRB1*0301 -> HLA-DRB1*03:01
    match = re.match(r'(?:HLA-)?([A-Z]+\d*)\*(\d{2,})(\d{2})?', variant)
    if match:
        gene = match.group(1)
        field1 = match.group(2)
        field2 = match.group(3)

        if len(field1) == 4 and field2 is None:
            # Format: B*5801 -> B*58:01
            field1, field2 = field1[:2], field1[2:]
        elif len(field1) == 2 and field2:
            # Format: already split
            pass
        elif len(field1) > 2 and field2 is None:
            # Try to split: assume first 2 digits are field1
            field2 = field1[2:]
            field1 = field1[:2]

        if field2:
            return f"HLA-{gene}*{field1}:{field2}"
        else:
            return f"HLA-{gene}*{field1}"

    return variant


def normalize_star_allele(gene: str, allele_num: str) -> str:
    """Normalize star allele format."""
    gene = gene.upper()
    # Remove trailing x/X for copy number variants
    allele_num = re.sub(r'[xX].*$', '', allele_num)
    return f"{gene}*{allele_num}"


def extract_rsids(text: str) -> list[str]:
    """Extract rsID variants from text."""
    pattern = r'\brs\d{4,}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.lower() for m in set(matches)]


def extract_star_alleles(text: str) -> list[str]:
    """Extract star allele variants from text.

    Handles genes like: CYP2C9, CYP2D6, CYP3A4, CYP4F2, UGT1A1, NUDT15, etc.
    """
    variants = []

    # Pattern for GENE*NUMBER format
    # Genes: CYP family, UGT family, NUDT, DPYD, TPMT, etc.
    gene_pattern = r'\b(CYP\w+|UGT\w+|NUDT\d+|DPYD|TPMT|NAT\d*|SLCO\w+|ABCB\d+)\*(\d+[xX]?[nN]?)\b'
    matches = re.findall(gene_pattern, text, re.IGNORECASE)
    for gene, allele in matches:
        normalized = normalize_star_allele(gene, allele)
        variants.append(normalized)

    return list(set(variants))


def extract_hla_alleles(text: str) -> list[str]:
    """Extract HLA allele variants from text.

    Handles multiple formats:
    - HLA-B*58:01
    - HLA-B*5801
    - B*58:01
    - B*5801
    """
    variants = []

    # HLA genes: A, B, C, DRB1, DQA1, DQB1, DPA1, DPB1, etc.
    # With HLA- prefix
    pattern1 = r'\bHLA-([A-Z]+\d*)\*(\d{2,}):?(\d{2})?\b'
    matches = re.findall(pattern1, text, re.IGNORECASE)
    for gene, f1, f2 in matches:
        if f2:
            variants.append(f"HLA-{gene.upper()}*{f1}:{f2}")
        elif len(f1) >= 4:
            # Split: 5801 -> 58:01
            variants.append(f"HLA-{gene.upper()}*{f1[:2]}:{f1[2:4]}")
        else:
            variants.append(f"HLA-{gene.upper()}*{f1}")

    # Without HLA- prefix (e.g., B*5801, DRB1*0301)
    # Only match common HLA gene names
    hla_genes = r'(?:A|B|C|Cw|DRB1|DRB3|DRB4|DRB5|DQA1|DQB1|DPA1|DPB1)'
    pattern2 = rf'\b({hla_genes})\*(\d{{2,}})(?::(\d{{2}}))?\b'
    matches = re.findall(pattern2, text, re.IGNORECASE)
    for gene, f1, f2 in matches:
        gene = gene.upper()
        if gene == 'CW':
            gene = 'C'  # Normalize Cw to C
        if f2:
            variants.append(f"HLA-{gene}*{f1}:{f2}")
        elif len(f1) >= 4:
            variants.append(f"HLA-{gene}*{f1[:2]}:{f1[2:4]}")
        else:
            variants.append(f"HLA-{gene}*{f1}")

    return list(set(variants))


def extract_all_variants(text: str) -> list[str]:
    """Extract all variant types from text."""
    variants = []
    variants.extend(extract_rsids(text))
    variants.extend(extract_star_alleles(text))
    variants.extend(extract_hla_alleles(text))
    return list(set(variants))


def run_experiment():
    """Run the regex variant extraction experiment on all benchmark articles."""
    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())

    print(f"Running regex extraction v2 on {len(pmcids)} articles...\n")

    results = {
        "run_name": "regex_extraction_v2",
        "improvements": [
            "Uses full article text",
            "Handles HLA without HLA- prefix",
            "Broader star allele gene patterns (UGT, NUDT, etc)",
            "Format normalization"
        ],
    }

    total_recall = 0
    total_precision = 0
    per_article_results = []

    for pmcid in pmcids:
        # Get FULL article text
        text = get_markdown_text(pmcid)

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

        total_recall += result.match_rate
        total_precision += precision

        per_article_results.append({
            "pmcid": pmcid,
            "recall": result.match_rate,
            "precision": precision,
            "true_count": len(true_variants),
            "extracted_count": len(extracted_variants),
            "matches": result.matches,
            "misses": result.misses,
            "extras": result.extras,
        })

        # Print summary
        status = "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        print(f"  {status} {pmcid}: recall={result.match_rate:.0%} precision={precision:.0%} "
              f"(found {len(result.matches)}/{len(true_variants)}, extras={len(result.extras)})")

        if result.misses:
            print(f"      Missed: {result.misses}")

    # Calculate aggregates
    n = len(per_article_results)
    avg_recall = total_recall / n if n > 0 else 0
    avg_precision = total_precision / n if n > 0 else 0

    results["avg_recall"] = avg_recall
    results["avg_precision"] = avg_precision
    results["articles_scored"] = n
    results["per_article_results"] = per_article_results

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Articles scored: {n}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls/n:.0%})")

    # Save results
    output_path = Path(__file__).parent / "results_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
