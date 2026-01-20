"""
Regex + Term Normalization hybrid variant extraction.

This approach combines:
1. V5 regex extraction (with BioC supplements + SNP expansion)
2. Post-extraction normalization using VariantLookup

The goal is to:
- Use fuzzy matching to catch format variations
- Normalize all extracted variants against PharmGKB/ClinPGx databases
- Track normalization mappings to understand transformations
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.experiments.utils import get_markdown_text
from src.experiments.utils_bioc import fetch_bioc_supplement, prefetch_bioc_supplements
from src.benchmark_v2.variant_bench import load_variant_bench_data, score_variants
from src.term_normalization.snp_expansion import SNPExpander
from src.term_normalization import VariantLookup


# Initialize global instances
_snp_expander = None
_variant_lookup = None


def get_snp_expander() -> SNPExpander:
    """Get or initialize the SNP expander singleton."""
    global _snp_expander
    if _snp_expander is None:
        _snp_expander = SNPExpander()
        _snp_expander.load_or_build()
    return _snp_expander


def get_variant_lookup() -> VariantLookup:
    """Get or initialize the VariantLookup singleton."""
    global _variant_lookup
    if _variant_lookup is None:
        _variant_lookup = VariantLookup()
    return _variant_lookup


# ============================================================================
# Extraction Functions (from v5)
# ============================================================================


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


# ============================================================================
# Term Normalization Functions
# ============================================================================


def normalize_variants(
    variants: List[str], threshold: float = 0.8, top_k: int = 3
) -> Dict[str, List[Dict]]:
    """
    Normalize extracted variants using VariantLookup.

    Args:
        variants: List of extracted variant strings
        threshold: Similarity threshold for fuzzy matching (0-1)
        top_k: Number of top matches to return per variant

    Returns:
        Dictionary mapping original_variant -> list of normalized matches
        Each match includes: {id, normalized_term, url, score}
    """
    variant_lookup = get_variant_lookup()
    mappings = {}

    for variant in variants:
        try:
            results = variant_lookup.search(variant, threshold=threshold, top_k=top_k)
            if results:
                mappings[variant] = [
                    {
                        "id": r.id,
                        "normalized_term": r.normalized_term,
                        "url": r.url,
                        "score": r.score,
                    }
                    for r in results
                ]
            else:
                # No match found
                mappings[variant] = []
        except Exception as e:
            # Log error but continue
            print(f"    Warning: Error normalizing {variant}: {e}")
            mappings[variant] = []

    return mappings


def apply_normalization_to_variants(
    extracted_variants: List[str],
    mappings: Dict[str, List[Dict]],
    min_score: float = 0.9,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Apply normalization mappings to get final variant list.

    Strategy:
    - Keep original variant if no mapping found
    - Use normalized term if mapping score >= min_score
    - Track which variants were normalized

    Args:
        extracted_variants: Original extracted variants
        mappings: Normalization mappings from normalize_variants()
        min_score: Minimum score to accept normalized term

    Returns:
        Tuple of (final_variants, normalization_applied)
        - final_variants: List of final variant identifiers
        - normalization_applied: Dict mapping original -> normalized (only for applied normalizations)
    """
    final_variants = []
    normalization_applied = {}

    for variant in extracted_variants:
        if variant in mappings and mappings[variant]:
            # Get best match
            best_match = mappings[variant][0]
            if best_match["score"] >= min_score:
                # Use normalized term
                normalized = best_match["normalized_term"]
                final_variants.append(normalized)
                if normalized.lower() != variant.lower():
                    normalization_applied[variant] = normalized
            else:
                # Score too low, keep original
                final_variants.append(variant)
        else:
            # No mapping, keep original
            final_variants.append(variant)

    return final_variants, normalization_applied


# ============================================================================
# Experiment Runner
# ============================================================================


def run_experiment(
    normalization_threshold: float = 0.8,
    normalization_min_score: float = 0.9,
    normalization_top_k: int = 3,
):
    """
    Run the hybrid regex + term normalization experiment.

    Args:
        normalization_threshold: Fuzzy matching threshold for VariantLookup
        normalization_min_score: Minimum score to apply normalization
        normalization_top_k: Number of candidate matches to retrieve
    """
    # Pre-initialize components
    print("Initializing components...")
    expander = get_snp_expander()
    variant_lookup = get_variant_lookup()
    stats = expander.stats()
    print(f"  SNP Expander: {stats['total_mappings']} mappings")
    print(f"  VariantLookup: Ready\n")

    benchmark_data = load_variant_bench_data()
    pmcids = list(benchmark_data.keys())

    # Prefetch BioC supplements
    print("Prefetching BioC supplements...")
    bioc_availability = prefetch_bioc_supplements(pmcids, delay=0.2)
    bioc_available_count = sum(1 for v in bioc_availability.values() if v)
    print(f"  {bioc_available_count}/{len(pmcids)} articles have BioC supplements\n")

    print(f"Running hybrid extraction on {len(pmcids)} articles...\n")

    results = {
        "run_name": "regex_term_norm_hybrid",
        "description": "Regex extraction (v5) + post-extraction term normalization",
        "normalization_params": {
            "threshold": normalization_threshold,
            "min_score": normalization_min_score,
            "top_k": normalization_top_k,
        },
        "improvements": [
            "All v5 improvements (BioC supplements, SNP expansion)",
            "Post-extraction normalization with VariantLookup",
            "Fuzzy matching for format variations",
            "Normalization mapping tracking",
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

    # Track normalization statistics
    total_normalized_count = 0
    total_variants_extracted = 0
    normalization_helped_articles = []

    for pmcid in pmcids:
        combined_text, supplement_text = get_combined_text(pmcid)

        if not combined_text:
            print(f"  {pmcid}: No text found (skipping)")
            continue

        # Step 1: Extract variants (regex)
        extracted_variants = extract_all_variants(combined_text)
        total_variants_extracted += len(extracted_variants)

        # Step 2: Normalize variants
        normalization_mappings = normalize_variants(
            extracted_variants,
            threshold=normalization_threshold,
            top_k=normalization_top_k,
        )

        # Step 3: Apply normalization
        final_variants, normalization_applied = apply_normalization_to_variants(
            extracted_variants,
            normalization_mappings,
            min_score=normalization_min_score,
        )

        total_normalized_count += len(normalization_applied)

        # Step 4: Score against benchmark
        true_variants = benchmark_data[pmcid]
        result = score_variants(final_variants, true_variants, pmcid)

        if len(final_variants) > 0:
            precision = len(result.matches) / len(final_variants)
        else:
            precision = 1.0 if len(true_variants) == 0 else 0.0

        total_recall += result.match_rate
        total_precision += precision

        # Check if normalization helped
        normalization_helped = False
        if normalization_applied:
            # Compare with non-normalized version
            non_normalized_result = score_variants(
                extracted_variants, true_variants, pmcid
            )
            if len(result.matches) > len(non_normalized_result.matches):
                normalization_helped = True
                normalization_helped_articles.append(
                    {
                        "pmcid": pmcid,
                        "recall_improvement": result.match_rate
                        - non_normalized_result.match_rate,
                        "new_matches": list(
                            set(result.matches) - set(non_normalized_result.matches)
                        ),
                        "normalizations": normalization_applied,
                    }
                )

        per_article_results.append(
            {
                "pmcid": pmcid,
                "recall": result.match_rate,
                "precision": precision,
                "true_count": len(true_variants),
                "extracted_count": len(extracted_variants),
                "final_count": len(final_variants),
                "matches": result.matches,
                "misses": result.misses,
                "extras": result.extras,
                "has_supplement": supplement_text is not None,
                "normalization_mappings": normalization_mappings,
                "normalization_applied": normalization_applied,
                "normalization_helped": normalization_helped,
            }
        )

        status = (
            "✓" if result.match_rate == 1.0 else "○" if result.match_rate > 0 else "✗"
        )
        norm_note = (
            f" [norm: {len(normalization_applied)}]" if normalization_applied else ""
        )
        helped_note = " [+helped]" if normalization_helped else ""
        print(
            f"  {status} {pmcid}: recall={result.match_rate:.0%} precision={precision:.0%} "
            f"(found {len(result.matches)}/{len(true_variants)}, extras={len(result.extras)}){norm_note}{helped_note}"
        )

        if result.misses:
            print(f"      Missed: {result.misses[:5]}")  # Show first 5 misses

    n = len(per_article_results)
    avg_recall = total_recall / n if n > 0 else 0
    avg_precision = total_precision / n if n > 0 else 0

    results["avg_recall"] = avg_recall
    results["avg_precision"] = avg_precision
    results["articles_scored"] = n
    results["per_article_results"] = per_article_results
    results["normalization_stats"] = {
        "total_variants_extracted": total_variants_extracted,
        "total_normalizations_applied": total_normalized_count,
        "normalization_rate": total_normalized_count / total_variants_extracted
        if total_variants_extracted > 0
        else 0,
        "articles_where_normalization_helped": len(normalization_helped_articles),
        "details": normalization_helped_articles,
    }

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Articles scored: {n}")
    print(f"Average Recall: {avg_recall:.1%}")
    print(f"Average Precision: {avg_precision:.1%}")

    perfect_recalls = sum(1 for r in per_article_results if r["recall"] == 1.0)
    print(f"Perfect recall: {perfect_recalls}/{n} articles ({perfect_recalls / n:.0%})")

    print(f"\nNormalization Statistics:")
    print(f"  Total variants extracted: {total_variants_extracted}")
    print(f"  Normalizations applied: {total_normalized_count}")
    norm_rate = (
        total_normalized_count / total_variants_extracted
        if total_variants_extracted > 0
        else 0
    )
    print(f"  Normalization rate: {norm_rate:.1%}")
    print(
        f"  Articles where normalization helped: {len(normalization_helped_articles)}"
    )

    if normalization_helped_articles:
        print(f"\n  Normalization improvements:")
        for item in normalization_helped_articles[:5]:  # Show first 5
            print(
                f"    {item['pmcid']}: +{item['recall_improvement']:.1%} recall, new matches: {item['new_matches']}"
            )
        if len(normalization_helped_articles) > 5:
            print(f"    ... and {len(normalization_helped_articles) - 5} more")

    output_path = Path(__file__).parent / "results_term_norm.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment(
        normalization_threshold=0.8,  # Fuzzy matching threshold
        normalization_min_score=0.9,  # Only apply if score >= 0.9
        normalization_top_k=3,  # Get top 3 candidates
    )
