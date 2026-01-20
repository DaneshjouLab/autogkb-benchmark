import re
from pathlib import Path

from loguru import logger

# Calculate repository root (utils.py is in src/experiments/)
ROOT = Path(__file__).resolve().parents[2]


def get_markdown_text(pmcid: str) -> str:
    """
    Retrieves the markdown text in string format of an article given its PubMed Central ID (PMCID).

    Args:
        pmcid: PMCID of the article

    Returns:
        The text content of the markdown file as a string
    """
    markdown_path = ROOT / "data" / "articles" / f"{pmcid}.md"
    try:
        with open(markdown_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Article {pmcid} not found at {markdown_path}")
        return ""


def _extract_section(markdown_text: str, section_patterns: list[str]) -> str:
    """
    Extract a section from markdown text based on header patterns.

    Args:
        markdown_text: The full markdown text
        section_patterns: List of regex patterns to match section headers

    Returns:
        The extracted section text, or empty string if not found
    """
    combined_pattern = "|".join(section_patterns)
    # Match section headers (## or ### followed by optional numbering and the section name)
    header_pattern = rf"^(#{{2,3}})\s*(?:\d+\.?\s*)?({combined_pattern})\s*$"

    matches = list(
        re.finditer(header_pattern, markdown_text, re.MULTILINE | re.IGNORECASE)
    )

    if not matches:
        return ""

    sections = []
    for match in matches:
        start = match.end()
        # Find the next section header of same or higher level
        header_level = len(match.group(1))
        next_header_pattern = rf"^#{{1,{header_level}}}\s+\S"
        next_match = re.search(next_header_pattern, markdown_text[start:], re.MULTILINE)

        if next_match:
            end = start + next_match.start()
        else:
            end = len(markdown_text)

        section_text = markdown_text[start:end].strip()
        if section_text:
            sections.append(f"## {match.group(2).title()}\n\n{section_text}")

    return "\n\n".join(sections)


def get_methods_and_conclusions_text(pmcid: str) -> str:
    """
    Retrieves the methods and conclusions sections from an article's markdown.

    Args:
        pmcid: PMCID of the article

    Returns:
        The methods and conclusions sections concatenated as a string
    """
    markdown_text = get_markdown_text(pmcid)
    if not markdown_text:
        return ""

    # Patterns for methods section (various naming conventions)
    methods_patterns = [
        r"materials?\s+and\s+methods?",
        r"methods?",
        r"patients?\s+and\s+methods?",
        r"study\s+design",
        r"experimental\s+procedures?",
    ]

    # Patterns for conclusions section
    conclusions_patterns = [
        r"conclusions?",
        r"discussion",
        r"discussion\s+and\s+conclusions?",
        r"summary",
    ]

    methods_text = _extract_section(markdown_text, methods_patterns)
    conclusions_text = _extract_section(markdown_text, conclusions_patterns)

    result_parts = []
    if methods_text:
        result_parts.append(methods_text)
    if conclusions_text:
        result_parts.append(conclusions_text)

    return "\n\n".join(result_parts)
