"""
Base interface and factory for variant extraction methods.
"""

from abc import ABC, abstractmethod


class VariantExtractor(ABC):
    """Base class for all variant extraction methods.

    Subclasses must implement:
        - name (class attribute): identifier for the extractor
        - get_variants(pmcid): extract variants from an article
    """

    name: str = ""

    @abstractmethod
    def get_variants(self, pmcid: str) -> list[str]:
        """Extract variants from an article.

        Args:
            pmcid: PubMed Central ID of the article

        Returns:
            List of variant identifier strings
        """
        ...


def create_variant_extractor(method: str, **kwargs) -> VariantExtractor:
    """Factory function to create a variant extractor by name.

    Args:
        method: One of "just_ask", "regex_v1", "regex_v2", "regex_v3",
                "regex_v4", "regex_v5", "regex_llm_filter",
                "regex_term_norm", "pubtator"
        **kwargs: Passed to the extractor constructor

    Returns:
        A VariantExtractor instance

    Raises:
        ValueError: If the method is not recognized
    """
    # Lazy imports to avoid circular dependencies
    from src.experiments.variant_finding.regex_variants.regex_extractor import (
        RegexExtractor,
    )
    from src.experiments.variant_finding.regex_variants.regex_v1_extractor import (
        RegexV1Extractor,
    )
    from src.experiments.variant_finding.regex_variants.regex_v2_extractor import (
        RegexV2Extractor,
    )
    from src.experiments.variant_finding.regex_variants.regex_v3_extractor import (
        RegexV3Extractor,
    )
    from src.experiments.variant_finding.regex_variants.regex_v4_extractor import (
        RegexV4Extractor,
    )
    from src.experiments.variant_finding.just_ask.just_ask import JustAskExtractor
    from src.experiments.variant_finding.pubtator.pubtator_extractor import (
        PubTatorExtractor,
    )
    from src.experiments.variant_finding.regex_term_norm.regex_term_norm import (
        RegexTermNormExtractor,
    )
    from src.experiments.variant_finding.regex_llm_filter.regex_llm_filter import (
        RegexLLMFilterExtractor,
    )

    extractors: dict[str, type[VariantExtractor]] = {
        "just_ask": JustAskExtractor,
        "regex_v1": RegexV1Extractor,
        "regex_v2": RegexV2Extractor,
        "regex_v3": RegexV3Extractor,
        "regex_v4": RegexV4Extractor,
        "regex_v5": RegexExtractor,
        "regex_llm_filter": RegexLLMFilterExtractor,
        "regex_term_norm": RegexTermNormExtractor,
        "pubtator": PubTatorExtractor,
    }

    if method not in extractors:
        available = ", ".join(extractors.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")

    return extractors[method](**kwargs)
