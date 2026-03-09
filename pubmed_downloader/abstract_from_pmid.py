"""
PMID --> Abstract (via NCBI E-Fetch API)

Fetches the abstract and basic metadata for articles that are not available
in PubMed Central (no PMCID). Returns a simple markdown-formatted string.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Optional
from loguru import logger


def get_abstract_markdown_from_pmid(pmid: str) -> Optional[str]:
    """
    Fetch the abstract and metadata for a PMID from PubMed and return as markdown.

    Args:
        pmid (str): The PubMed ID to fetch

    Returns:
        Optional[str]: Markdown-formatted abstract with title/authors, or None on failure
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
        "retmode": "xml",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch abstract for PMID {pmid}: {e}")
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML for PMID {pmid}: {e}")
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        logger.error(f"No article found in response for PMID {pmid}")
        return None

    # Extract title
    title_el = article.find(".//ArticleTitle")
    title = _get_element_text(title_el) if title_el is not None else "Unknown Title"

    # Extract authors
    authors = []
    for author in article.findall(".//Author"):
        last = author.findtext("LastName", "")
        fore = author.findtext("ForeName", "")
        if last:
            authors.append(f"{fore} {last}".strip())

    # Extract abstract
    abstract_parts = []
    for abstract_text in article.findall(".//AbstractText"):
        label = abstract_text.get("Label")
        text = _get_element_text(abstract_text)
        if text:
            if label:
                abstract_parts.append(f"**{label}:** {text}")
            else:
                abstract_parts.append(text)

    abstract = (
        "\n\n".join(abstract_parts) if abstract_parts else "No abstract available."
    )

    # Extract journal info
    journal = article.findtext(".//Journal/Title", "")
    year = article.findtext(".//PubDate/Year", "")
    doi_el = article.find(".//ArticleId[@IdType='doi']")
    doi = doi_el.text if doi_el is not None and doi_el.text else None

    # Build markdown
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    if authors:
        lines.append(", ".join(authors))
        lines.append("")
    if journal or year:
        lines.append(f"*{journal}* ({year})".strip())
        lines.append("")
    lines.append(f"PMID: {pmid}")
    if doi:
        lines.append(f"DOI: {doi}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "**Note: This article is not available on PubMed Central (Open Access). Only the abstract is included below.**"
    )
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(abstract)
    lines.append("")

    return "\n".join(lines)


def _get_element_text(element) -> str:
    """Extract all text from an element, including text within child tags (e.g. <i>, <b>)."""
    return "".join(element.itertext()).strip()
