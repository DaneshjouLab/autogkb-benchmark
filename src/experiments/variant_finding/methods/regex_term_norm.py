"""
Regex extraction + post-extraction term normalization.

Combines v5 regex extraction with fuzzy matching normalization via VariantLookup.
"""

from src.experiments.variant_finding.utils import (
    extract_all_variants,
    get_combined_text,
)
from src.term_normalization import VariantLookup


_variant_lookup = None


def _get_variant_lookup() -> VariantLookup:
    """Get or initialize the VariantLookup singleton."""
    global _variant_lookup
    if _variant_lookup is None:
        _variant_lookup = VariantLookup()
    return _variant_lookup


def regex_term_norm_extract(
    pmcid: str,
    threshold: float = 0.8,
    min_score: float = 0.9,
    top_k: int = 3,
) -> list[str]:
    combined_text, _ = get_combined_text(pmcid)
    if not combined_text:
        return []

    # Step 1: Regex extraction
    extracted = extract_all_variants(combined_text)

    # Step 2: Normalize
    variant_lookup = _get_variant_lookup()
    final_variants = []

    for variant in extracted:
        try:
            results = variant_lookup.search(variant, threshold=threshold, top_k=top_k)
            if results and results[0].score >= min_score:
                final_variants.append(results[0].normalized_term)
            else:
                final_variants.append(variant)
        except Exception:
            final_variants.append(variant)

    return list(set(final_variants))
