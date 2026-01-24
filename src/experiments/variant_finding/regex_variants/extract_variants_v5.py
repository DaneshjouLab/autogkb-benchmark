"""
Regex-based variant extraction experiment - Version 5.

Improvements over v4:
- Integrates BioC API to fetch supplementary material text
- Extracts variants from BOTH main article AND supplement text
- Uses cached BioC responses for efficiency

This addresses the major gap identified in v4 analysis: variants listed in
supplementary tables (like CYP2D6 star alleles in PMC6435416) are now captured.
"""

import json
import re
from pathlib import Path

from src.experiments.utils import get_markdown_text
from src.experiments.utils_bioc import fetch_bioc_supplement, prefetch_bioc_supplements
from src.benchmark_v2.variant_bench import load_variant_bench_data, score_variants
from src.term_normalization.snp_expansion import SNPExpander


# Initialize SNP expander globally
_snp_expander = None


def get_snp_expander() -> SNPExpander:
    """Get or initialize the SNP expander singleton."""
    global _snp_expander
    if _snp_expander is None:
        _snp_expander = SNPExpander()
        _snp_expander.load_or_build()
    return _snp_expander


def normalize_hla(variant: str) -> str:
    """Normalize HLA allele format to HLA-X*XX:XX format."""
    variant = variant.upper()

    # Already normalized
    if re.match(r"HLA-[A-Z]+\d*\*\d+:\d+", variant):
        return variant

    # Handle formats like B*5801 -> HLA-B*58:01
    match = re.match(r"(?:HLA-)?([A-Z]+\d*)\*(\d{2,})(\d{2})?", variant)
    if match:
        gene = match.group(1)
        field1 = match.group(2)
        field2 = match.group(3)

        if len(field1) == 4 and field2 is None:
            field1, field2 = field1[:2], field1[2:]
        elif len(field1) > 2 and field2 is None:
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
    # Remove trailing x/X for copy number variants but keep xN format
    allele_num = re.sub(r"[xX×].*$", "", allele_num)
    return f"{gene}*{allele_num}"


def extract_rsids(text: str) -> list[str]:
    """Extract rsID variants from text."""
    pattern = r"\brs\d{4,}\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.lower() for m in set(matches)]


def extract_snp_notations(text: str) -> list[str]:
    """
    Extract rsIDs from informal SNP notations in text.

    Handles patterns like:
    - CYP2B6 516G>T -> rs3745274
    - CYP2B6-G516T -> rs3745274
    - VKORC1-1639 G>A -> rs9923231

    Returns:
        List of rsIDs derived from SNP notations found in text.
    """
    expander = get_snp_expander()
    rsids = []

    # Pharmacogenes to search for
    pgx_genes = expander.target_genes

    for gene in pgx_genes:
        # Pattern 1: GENE followed by position and substitution
        # Matches: CYP2B6 516G>T, CYP2B6-516G>T, CYP2B6(516G>T)
        pattern1 = rf"\b({gene})[\s\-\(\)]*(-?\d+)\s*([ACGT])\s*>\s*([ACGT])"
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            matched_gene = match.group(1)
            pos = match.group(2)
            ref = match.group(3)
            alt = match.group(4)
            notation = f"{pos}{ref.upper()}>{alt.upper()}"

            mapping = expander.lookup(matched_gene, notation)
            if mapping:
                rsids.append(mapping.rsid.lower())

        # Pattern 2: Reversed notation GENE G516T
        pattern2 = rf"\b({gene})[\s\-\(\)]*([ACGT])(-?\d+)([ACGT])(?![>\d])"
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            matched_gene = match.group(1)
            ref = match.group(2)
            pos = match.group(3)
            alt = match.group(4)
            notation = f"{pos}{ref.upper()}>{alt.upper()}"

            mapping = expander.lookup(matched_gene, notation)
            if mapping:
                rsids.append(mapping.rsid.lower())

    return list(set(rsids))


def extract_star_alleles(text: str) -> list[str]:
    """Extract star allele variants from text.

    Handles:
    - Standard format: CYP2C9*3, UGT1A1*28
    - Space format: CYP2D6 *4, NUDT15 *3
    - Copy number: CYP2D6*1xN, *2xN
    """
    variants = []

    # Gene families that have star allele nomenclature
    pgx_genes = [
        "CYP2D6",
        "CYP2C9",
        "CYP2C19",
        "CYP2B6",
        "CYP3A4",
        "CYP3A5",
        "CYP4F2",
        "CYP2A6",
        "CYP1A2",
        "UGT1A1",
        "UGT2B7",
        "UGT2B15",
        "NUDT15",
        "DPYD",
        "TPMT",
        "NAT1",
        "NAT2",
        "SLCO1B1",
        "SLCO1B3",
        "SLCO2B1",
        "ABCB1",
        "ABCG2",
        "VKORC1",
        "IFNL3",
        "IFNL4",
    ]

    # Create pattern for all genes
    gene_pattern = "|".join(pgx_genes)

    # Pattern 1: GENE*NUMBER format (standard)
    pattern1 = rf"\b({gene_pattern})\*(\d+[xX×]?[nN]?)\b"
    matches = re.findall(pattern1, text, re.IGNORECASE)
    for gene, allele in matches:
        normalized = normalize_star_allele(gene, allele)
        variants.append(normalized)

    # Pattern 2: GENE *NUMBER format (space between gene and asterisk)
    pattern2 = rf"\b({gene_pattern})\s+\*(\d+[xX×]?[nN]?)\b"
    matches = re.findall(pattern2, text, re.IGNORECASE)
    for gene, allele in matches:
        normalized = normalize_star_allele(gene, allele)
        variants.append(normalized)

    # Pattern 3: Standalone star alleles (*3, *4, etc.) - need gene context
    # Look for patterns like "*3, *4, *6" or "*3/*4" near gene names
    standalone_pattern = r"\*(\d{1,2})\b"

    # Find all gene mentions and their positions
    gene_mentions = []
    for gene in pgx_genes:
        for match in re.finditer(rf"\b{gene}\b", text, re.IGNORECASE):
            gene_mentions.append((match.start(), match.end(), gene.upper()))

    # Pattern for diplotypes like *1×N/*2, *1/*10×N (with unicode multiplication sign)
    diplotype_pattern = r"\*(\d{1,2})[×xX]?[nN]?/\*(\d{1,2})[×xX]?[nN]?"
    for match in re.finditer(diplotype_pattern, text):
        allele1 = match.group(1)
        allele2 = match.group(2)
        diplotype_pos = match.start()

        # Find the nearest gene mention within 800 characters before (diplotypes often far from gene name)
        nearest_gene = None
        min_distance = 800

        for gene_start, gene_end, gene_name in gene_mentions:
            if gene_end <= diplotype_pos:
                distance = diplotype_pos - gene_end
                if distance < min_distance:
                    min_distance = distance
                    nearest_gene = gene_name

        if nearest_gene:
            variants.append(f"{nearest_gene}*{allele1}")
            variants.append(f"{nearest_gene}*{allele2}")
            # Also add xN variants if present
            if "×" in match.group(0) or "x" in match.group(0).lower():
                variants.append(f"{nearest_gene}*{allele1}xN")
                variants.append(f"{nearest_gene}*{allele2}xN")

    # Find all standalone star alleles
    for match in re.finditer(standalone_pattern, text):
        allele_num = match.group(1)
        allele_pos = match.start()

        # Find the nearest gene mention within 200 characters before this allele
        nearest_gene = None
        min_distance = 200

        for gene_start, gene_end, gene_name in gene_mentions:
            # Only look at genes that appear before the allele
            if gene_end <= allele_pos:
                distance = allele_pos - gene_end
                if distance < min_distance:
                    min_distance = distance
                    nearest_gene = gene_name

        if nearest_gene:
            normalized = normalize_star_allele(nearest_gene, allele_num)
            variants.append(normalized)

    # Pattern 4: Copy number variants with xN suffix
    xn_pattern = rf"\b({gene_pattern})\*(\d+)[xX×][nN]?\b"
    matches = re.findall(xn_pattern, text, re.IGNORECASE)
    for gene, allele in matches:
        # Also add the base allele
        normalized = normalize_star_allele(gene, allele)
        variants.append(normalized)
        # Add copy number variant format
        variants.append(f"{gene.upper()}*{allele}xN")

    return list(set(variants))


def extract_hla_alleles(text: str) -> list[str]:
    """Extract HLA allele variants from text.

    Handles multiple formats:
    - HLA-B*58:01
    - HLA-B*5801
    - B*58:01
    - B*5801
    - HLA-B*38:(01/02) - parenthetical notation
    - B*39:(01/05/06/09)
    """
    variants = []

    # HLA genes
    hla_genes = r"(?:A|B|C|Cw|DRB1|DRB3|DRB4|DRB5|DQA1|DQB1|DPA1|DPB1)"

    # With HLA- prefix
    pattern1 = r"\bHLA-([A-Z]+\d*)\*(\d{2,}):?(\d{2})?\b"
    matches = re.findall(pattern1, text, re.IGNORECASE)
    for gene, f1, f2 in matches:
        if f2:
            variants.append(f"HLA-{gene.upper()}*{f1}:{f2}")
        elif len(f1) >= 4:
            variants.append(f"HLA-{gene.upper()}*{f1[:2]}:{f1[2:4]}")
        else:
            variants.append(f"HLA-{gene.upper()}*{f1}")

    # Without HLA- prefix
    pattern2 = rf"\b({hla_genes})\*(\d{{2,}})(?::(\d{{2}}))?\b"
    matches = re.findall(pattern2, text, re.IGNORECASE)
    for gene, f1, f2 in matches:
        gene = gene.upper()
        if gene == "CW":
            gene = "C"
        if f2:
            variants.append(f"HLA-{gene}*{f1}:{f2}")
        elif len(f1) >= 4:
            variants.append(f"HLA-{gene}*{f1[:2]}:{f1[2:4]}")
        else:
            variants.append(f"HLA-{gene}*{f1}")

    # Parenthetical notation: HLA-B*38:(01/02) or B*39:(01/05/06/09)
    # Pattern: (HLA-)?(gene)*field1:(allele1/allele2/...)
    paren_pattern = rf"(?:HLA-)?({hla_genes})\*(\d{{2}}):?\(([/\d]+)\)"
    matches = re.findall(paren_pattern, text, re.IGNORECASE)
    for gene, field1, alleles_str in matches:
        gene = gene.upper()
        if gene == "CW":
            gene = "C"
        # Split the alleles: "01/02" -> ["01", "02"]
        allele_nums = alleles_str.split("/")
        for allele_num in allele_nums:
            if allele_num.isdigit():
                variants.append(f"HLA-{gene}*{field1}:{allele_num}")

    return list(set(variants))


def extract_all_variants(text: str) -> list[str]:
    """Extract all variant types from text."""
    variants = []
    variants.extend(extract_rsids(text))
    variants.extend(extract_snp_notations(text))
    variants.extend(extract_star_alleles(text))
    variants.extend(extract_hla_alleles(text))
    return list(set(variants))


def get_combined_text(pmcid: str) -> tuple[str, str | None]:
    """
    Get combined article + supplement text for extraction.

    Returns:
        Tuple of (combined_text, supplement_text_or_none)
    """
    article_text = get_markdown_text(pmcid)
    supplement_text = fetch_bioc_supplement(pmcid, use_cache=True)

    if supplement_text:
        combined_text = (
            article_text + "\n\n--- SUPPLEMENTARY MATERIAL ---\n\n" + supplement_text
        )
    else:
        combined_text = article_text

    return combined_text, supplement_text


def run_experiment():
    """Run the regex variant extraction experiment on all benchmark articles."""
    # Pre-initialize the SNP expander
    print("Initializing SNP expander...")
    expander = get_snp_expander()
    stats = expander.stats()
    print(f"  Loaded {stats['total_mappings']} SNP notation mappings")
    print(
        f"  Covering {stats['unique_rsids']} unique rsIDs across {len(stats['genes'])} genes\n"
    )

    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())

    # Prefetch BioC supplements
    print("Prefetching BioC supplements...")
    bioc_availability = prefetch_bioc_supplements(pmcids, delay=0.2)
    bioc_available_count = sum(1 for v in bioc_availability.values() if v)
    print(f"  {bioc_available_count}/{len(pmcids)} articles have BioC supplements\n")

    print(f"Running regex extraction v5 on {len(pmcids)} articles...\n")

    results = {
        "run_name": "regex_extraction_v5",
        "improvements": [
            "All v4 improvements",
            "BioC API integration for supplementary materials",
            "Extracts variants from article + supplement text",
            f"{bioc_available_count} articles with supplement data",
        ],
        "snp_expansion_stats": stats,
        "bioc_stats": {
            "articles_with_supplements": bioc_available_count,
            "articles_without_supplements": len(pmcids) - bioc_available_count,
        },
    }

    total_recall = 0
    total_precision = 0
    per_article_results = []

    # Track improvements from supplements
    supplement_improvements = []

    for pmcid in pmcids:
        combined_text, supplement_text = get_combined_text(pmcid)

        if not combined_text:
            print(f"  {pmcid}: No text found (skipping)")
            continue

        extracted_variants = extract_all_variants(combined_text)

        # Also extract from article only (for comparison)
        article_only_text = get_markdown_text(pmcid)
        article_only_variants = extract_all_variants(article_only_text)

        # Track which variants came from supplements
        supplement_only_variants = set(extracted_variants) - set(article_only_variants)

        # Track SNP notation expansions
        snp_rsids = extract_snp_notations(combined_text)

        true_variants = benchmark_data[pmcid]
        result = score_variants(extracted_variants, true_variants, pmcid)

        if len(extracted_variants) > 0:
            precision = len(result.matches) / len(extracted_variants)
        else:
            precision = 1.0 if len(true_variants) == 0 else 0.0

        total_recall += result.match_rate
        total_precision += precision

        # Check if supplement helped
        supplement_helped = False
        supplement_recovered = []
        if supplement_text and supplement_only_variants:
            # Check which supplement-only variants are matches
            supplement_recovered = [
                v
                for v in supplement_only_variants
                if v.lower() in [m.lower() for m in result.matches]
            ]
            if supplement_recovered:
                supplement_helped = True
                supplement_improvements.append(
                    {"pmcid": pmcid, "recovered": supplement_recovered}
                )

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
                "snp_expansions": snp_rsids,
                "has_supplement": supplement_text is not None,
                "supplement_recovered": supplement_recovered,
            }
        )

        status = (
            "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        )
        supp_note = (
            f" [+{len(supplement_recovered)} from supp]" if supplement_recovered else ""
        )
        snp_note = f" [+{len(snp_rsids)} SNP exp]" if snp_rsids else ""
        print(
            f"  {status} {pmcid}: recall={result.match_rate:.0%} precision={precision:.0%} "
            f"(found {len(result.matches)}/{len(true_variants)}, extras={len(result.extras)}){supp_note}{snp_note}"
        )

        if result.misses:
            print(f"      Missed: {result.misses}")

    n = len(per_article_results)
    avg_recall = total_recall / n if n > 0 else 0
    avg_precision = total_precision / n if n > 0 else 0

    results["avg_recall"] = avg_recall
    results["avg_precision"] = avg_precision
    results["articles_scored"] = n
    results["per_article_results"] = per_article_results
    results["supplement_improvements"] = {
        "articles_helped": len(supplement_improvements),
        "total_variants_recovered": sum(
            len(s["recovered"]) for s in supplement_improvements
        ),
        "details": supplement_improvements,
    }

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Articles scored: {n}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls / n:.0%})")

    print("\nBioC Supplement Integration:")
    print(f"  Articles with supplements: {bioc_available_count}")
    print(f"  Articles helped by supplements: {len(supplement_improvements)}")
    total_supp_variants = sum(len(s["recovered"]) for s in supplement_improvements)
    print(f"  Total variants recovered from supplements: {total_supp_variants}")

    if supplement_improvements:
        print("\n  Supplement recoveries:")
        for s in supplement_improvements:
            print(f"    {s['pmcid']}: {s['recovered']}")

    output_path = Path(__file__).parent / "results_v5.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
