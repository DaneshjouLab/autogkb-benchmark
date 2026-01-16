from pathlib import Path
from loguru import logger


def get_markdown_text(pmcid: str) -> str:
    """
    Retrieves the markdown text in string format of an article given its PubMed Central ID (PMCID).

    Args:
        pmcid: PMCID of the article

    Returns:
        The text content of the markdown file as a string
    """
    markdown_path = Path("data") / "articles" / f"{pmcid}.md"
    try:
        with open(markdown_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Article {pmcid} not found at {markdown_path}")
        return ""
