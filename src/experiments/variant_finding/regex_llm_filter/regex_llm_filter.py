"""
Hybrid regex extraction + LLM filtering.

Uses v5 regex extraction for high-recall candidate generation, then filters
with an LLM to remove false positives.
"""

from pathlib import Path

from src.experiments.variant_finding.VariantExtractor import VariantExtractor
from src.experiments.variant_finding.utils import (
    extract_all_variants,
    extract_json_array,
    get_combined_text,
    get_snp_expander,
    load_prompts,
)
from src.utils import call_llm

PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


class RegexLLMFilterExtractor(VariantExtractor):
    name = "regex_llm_filter"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        prompt_version: str = "v1",
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.prompts = load_prompts(PROMPTS_FILE)
        get_snp_expander()

    def get_variants(self, pmcid: str) -> list[str]:
        combined_text, _ = get_combined_text(pmcid)
        if not combined_text:
            return []

        # Step 1: Regex extraction
        regex_variants = extract_all_variants(combined_text)
        if not regex_variants:
            return []

        # Step 2: LLM filtering
        prompt_config = self.prompts[self.prompt_version]
        variants_list = "\n".join([f"- {v}" for v in regex_variants])

        system_prompt = prompt_config["system"]
        user_prompt = prompt_config["user"].format(
            variants_list=variants_list, article_text=combined_text
        )

        response = call_llm(self.model, system_prompt, user_prompt)
        return extract_json_array(response)
