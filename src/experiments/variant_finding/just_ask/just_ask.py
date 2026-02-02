"""
LLM-based variant extraction.

Asks an LLM to directly extract pharmacogenetic variants from article text.
"""

from pathlib import Path

from src.experiments.variant_finding.VariantExtractor import VariantExtractor
from src.experiments.variant_finding.utils import extract_json_array, load_prompts
from src.utils import call_llm, get_methods_and_conclusions_text

PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


class JustAskExtractor(VariantExtractor):
    name = "just_ask"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", prompt_version: str = "v1"):
        self.model = model
        self.prompt_version = prompt_version
        self.prompts = load_prompts(PROMPTS_FILE)

    def get_variants(self, pmcid: str) -> list[str]:
        text = get_methods_and_conclusions_text(pmcid)
        if not text:
            return []

        prompt_config = self.prompts[self.prompt_version]
        user_prompt = prompt_config["user"].format(article_text=text)
        system_prompt = prompt_config["system"]

        response = call_llm(self.model, system_prompt, user_prompt)
        return extract_json_array(response)
