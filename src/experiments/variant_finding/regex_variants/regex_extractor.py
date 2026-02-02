"""
Regex-based variant extraction (v5).

Uses pattern matching for rsIDs, star alleles, HLA alleles, and SNP notation
expansion, with BioC supplementary material integration.
"""

from src.experiments.variant_finding.VariantExtractor import VariantExtractor
from src.experiments.variant_finding.utils import (
    extract_all_variants,
    get_combined_text,
    get_snp_expander,
)


class RegexExtractor(VariantExtractor):
    name = "regex_v5"

    def __init__(self):
        # Pre-initialize the SNP expander
        get_snp_expander()

    def get_variants(self, pmcid: str) -> list[str]:
        combined_text, _ = get_combined_text(pmcid)
        if not combined_text:
            return []
        return extract_all_variants(combined_text)
