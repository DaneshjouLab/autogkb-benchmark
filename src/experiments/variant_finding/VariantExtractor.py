# Interface for variant extraction to be implemented by experimental methods
from src.utils import get_pmcid_markdown_text

class VariantExtractor:
    def __init__(self, model: str):
        self.model = model

    def get_pmcid_markdown_text(self, pmcid: str) -> str:
        return get_pmcid_markdown_text(pmcid)
    
    def get_variants(self, pmcid: str) -> list[str]:
        raise NotImplementedError