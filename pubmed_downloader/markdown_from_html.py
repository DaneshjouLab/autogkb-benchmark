#!/usr/bin/env python3
"""
PubMed HTML to Markdown Converter

Converts PubMed/PMC HTML articles to structured markdown format.
Based on analysis of PMC article structure and conversion process documented
in pubmed_html_to_markdown_conversion_process.md
"""

import re
import html
from typing import Dict
import os
from bs4 import BeautifulSoup, Tag, NavigableString
from loguru import logger
import tqdm


class PubMedHTMLToMarkdownConverter:
    """Converts PubMed/PMC HTML articles to markdown format."""

    def __init__(self):
        self.soup = None
        self.base_url = "https://pmc.ncbi.nlm.nih.gov"

    def convert_file(self, html_file_path: str) -> str:
        """
        Convert HTML file to markdown string

        Args:
            html_file_path (str): The path to the HTML file to convert
        Returns:
            str: The markdown content
        """
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self.convert_html(html_content)

    def convert_html(self, html_content: str) -> str:
        """
        Convert HTML content to markdown string

        Args:
            html_content (str): The HTML content to convert
        Returns:
            str: The markdown content
        """
        self.soup = BeautifulSoup(html_content, "html.parser")

        # Build markdown document
        markdown_parts = []

        # Extract and format metadata
        metadata = self._extract_metadata()
        markdown_parts.append(self._format_metadata(metadata))

        # Check if this is a scanned document
        is_scanned = self._is_scanned_document()

        if is_scanned:
            markdown_parts.append(self._handle_scanned_document())
        else:
            # Extract main content sections
            markdown_parts.append(self._extract_abstract())
            markdown_parts.append(self._extract_main_content())
            markdown_parts.append(self._extract_references())

        # Join all parts and clean up
        markdown = "\n\n".join(filter(None, markdown_parts))
        return self._clean_markdown(markdown)

    def _extract_pmcid(self) -> str:
        """Extract PMCID from the HTML"""
        # Look for PMCID in canonical URL or meta tags
        canonical = self.soup.find("link", {"rel": "canonical"})
        if canonical and canonical.get("href"):
            match = re.search(r"PMC(\d+)", canonical["href"])
            if match:
                return f"PMC{match.group(1)}"

        # Look in text content
        pmcid_text = self.soup.find(text=re.compile(r"PMCID:\s*PMC\d+"))
        if pmcid_text:
            match = re.search(r"PMC\d+", pmcid_text)
            if match:
                return match.group(0)

        return ""

    def _extract_metadata(self) -> Dict[str, str]:
        """Extract article metadata from HTML head."""
        metadata = {}

        # Extract citation metadata
        meta_mappings = {
            "title": "citation_title",
            "journal": "citation_journal_title",
            "doi": "citation_doi",
            "pmid": "citation_pmid",
            "pdf_url": "citation_pdf_url",
            "publication_date": "citation_publication_date",
            "abstract_url": "citation_abstract_html_url",
            "fulltext_url": "citation_fulltext_html_url",
        }

        for key, meta_name in meta_mappings.items():
            meta_tag = self.soup.find("meta", attrs={"name": meta_name})
            if meta_tag and meta_tag.get("content"):
                metadata[key] = meta_tag["content"].strip()

        # Extract authors
        authors = []
        author_tags = self.soup.find_all("meta", attrs={"name": "citation_author"})
        for tag in author_tags:
            if tag.get("content"):
                authors.append(tag["content"].strip())
        metadata["authors"] = authors

        # Extract title from page title if not found in meta
        if "title" not in metadata:
            title_tag = self.soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()
                # Remove " - PMC" suffix if present
                title = re.sub(r"\s*-\s*PMC\s*$", "", title)
                metadata["title"] = title

        metadata["pmcid"] = self._extract_pmcid()

        return metadata

    def _format_metadata(self, metadata: Dict[str, str]) -> str:
        """Format metadata as markdown header."""
        lines = []

        # Title
        if "title" in metadata:
            lines.append(f"# {metadata['title']}")
            lines.append("")

        lines.append("## Metadata")

        # Authors
        if "authors" in metadata and metadata["authors"]:
            authors_str = ", ".join(metadata["authors"])
            lines.append(f"**Authors:** {authors_str}")

        # Journal
        if "journal" in metadata:
            lines.append(f"**Journal:** {metadata['journal']}")

        # Publication date
        if "publication_date" in metadata:
            lines.append(f"**Date:** {metadata['publication_date']}")

        # DOI
        if "doi" in metadata:
            lines.append(
                f"**DOI:** [{metadata['doi']}](https://doi.org/{metadata['doi']})"
            )

        # PMID
        if "pmid" in metadata:
            lines.append(f"**PMID:** {metadata['pmid']}")

        # PMCID
        if "pmcid" in metadata:
            lines.append(f"**PMCID:** {metadata['pmcid']}")

        # URL
        if "pmcid" in metadata:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{metadata['pmcid']}/"
            lines.append(f"**URL:** {url}")

        # PDF
        if "pdf_url" in metadata:
            lines.append(f"**PDF:** [{metadata['pdf_url']}]({metadata['pdf_url']})")

        return "\n".join(lines)

    def _is_scanned_document(self) -> bool:
        """Check if this is a scanned document (legacy format)."""
        scanned_indicators = [
            self.soup.find("section", class_="scanned-pages"),
            self.soup.find("meta", attrs={"name": "ncbi_type", "content": "scanpage"}),
            self.soup.find("figure", class_="fig-scanned"),
        ]
        return any(indicator is not None for indicator in scanned_indicators)

    def _handle_scanned_document(self) -> str:
        """Handle scanned documents with limited structured content."""
        lines = []
        lines.append(
            "*Note: This is a scanned document with limited structured text. Full content available in PDF.*"
        )
        lines.append("")

        # Try to extract abstract if available
        abstract = self._extract_abstract()
        if abstract.strip():
            lines.append(abstract)

        # Add scanned page images
        scanned_section = self.soup.find("section", class_="scanned-pages")
        if scanned_section:
            lines.append("## Full Text (Scanned Pages)")
            lines.append("")

            figures = scanned_section.find_all("figure", class_="fig-scanned")
            for i, figure in enumerate(figures, 1):
                img = figure.find("img")
                if img and img.get("src"):
                    alt_text = img.get("alt", f"Page {i}")
                    lines.append(f"### Page {i}")
                    lines.append(f"![{alt_text}]({img['src']})")
                    lines.append("")

        return "\n".join(lines)

    def _extract_abstract(self) -> str:
        """Extract abstract section."""
        abstract_section = self.soup.find("section", class_="abstract")
        if not abstract_section:
            return ""

        lines = ["## Abstract", ""]

        # Handle structured abstracts with subsections
        subsections = abstract_section.find_all(["h3", "h4"], class_="pmc_sec_title")
        if subsections:
            current_section = None
            for element in abstract_section.descendants:
                if isinstance(element, Tag):
                    if element.name in ["h3", "h4"] and "pmc_sec_title" in element.get(
                        "class", []
                    ):
                        current_section = element.get_text().strip()
                        # Remove trailing colon if present to avoid double colons
                        current_section = current_section.rstrip(":")
                        lines.append(f"**{current_section}:** ")
                    elif element.name == "p" and current_section:
                        text = self._clean_text(element.get_text())
                        if text:
                            lines.append(text)
                            lines.append("")
        else:
            # Simple abstract without subsections
            paragraphs = abstract_section.find_all("p")
            for p in paragraphs:
                text = self._clean_text(p.get_text())
                if text:
                    lines.append(text)
                    lines.append("")

        return "\n".join(lines)

    def _extract_main_content(self) -> str:
        """Extract main article content sections."""
        main_body = self.soup.find("section", class_="main-article-body")
        if not main_body:
            return ""

        lines = []

        # Find all major sections
        sections = main_body.find_all("section", id=True)

        for section in sections:
            # Skip abstract (already handled), references (handled separately), and keywords (part of abstract)
            if any(
                cls in section.get("class", [])
                for cls in ["abstract", "ref-list", "kwd-group"]
            ):
                continue

            section_content = self._process_section(section)
            if section_content:
                lines.append(section_content)

        return "\n".join(lines)

    def _process_section(self, section: Tag) -> str:
        """Process a single content section."""
        lines = []

        # Extract section title
        title_tag = section.find(["h1", "h2", "h3", "h4"], class_="pmc_sec_title")
        if title_tag:
            title = self._clean_text(title_tag.get_text())
            level = int(title_tag.name[1])  # h2 -> 2, h3 -> 3, etc.
            markdown_level = "#" * level
            lines.append(f"{markdown_level} {title}")
            lines.append("")

        # Process section content
        for element in section.children:
            if isinstance(element, Tag):
                if element.name == "p":
                    text = self._process_paragraph(element)
                    if text:
                        lines.append(text)
                        lines.append("")
                elif element.name == "section" and element.get("class"):
                    # Handle subsections, tables, figures
                    if "tw" in element.get("class", []):  # Table
                        table_md = self._process_table(element)
                        if table_md:
                            lines.append(table_md)
                    elif any(
                        cls in element.get("class", []) for cls in ["fig", "figure"]
                    ):  # Figure
                        figure_md = self._process_figure(element)
                        if figure_md:
                            lines.append(figure_md)
                    else:  # Subsection
                        subsection_md = self._process_section(element)
                        if subsection_md:
                            lines.append(subsection_md)
                elif element.name == "figure":
                    figure_md = self._process_figure(element)
                    if figure_md:
                        lines.append(figure_md)
                elif element.name == "table":
                    # Handle direct table elements
                    table_md = self._convert_table_to_markdown(element)
                    if table_md:
                        lines.append(table_md)

        return "\n".join(lines)

    def _process_paragraph(self, p_tag: Tag) -> str:
        """Process paragraph with inline formatting and citations."""
        text = ""

        def process_element(element):
            """Process a single element and return its markdown representation."""
            if isinstance(element, NavigableString):
                return str(element)
            elif isinstance(element, Tag):
                if element.name == "em" or element.name == "i":
                    return f"*{element.get_text()}*"
                elif element.name == "strong" or element.name == "b":
                    return f"**{element.get_text()}**"
                elif element.name == "sub":
                    return f"_{element.get_text()}_"
                elif element.name == "sup":
                    return f"^{element.get_text()}^"
                elif element.name == "a":
                    # Handle citations and cross-references
                    link_text = element.get_text().strip()
                    href = element.get("href", "")

                    if href.startswith("#"):
                        # Internal reference
                        return f"[{link_text}]({href})"
                    elif href:
                        # External link
                        return f"[{link_text}]({href})"
                    else:
                        return link_text
                else:
                    # For any other tag, recursively process its contents
                    result = ""
                    for child in element.contents:
                        result += process_element(child)
                    return result
            return ""

        # Process all direct contents of the paragraph
        for element in p_tag.contents:
            text += process_element(element)

        return self._clean_text(text)

    def _process_table(self, table_section: Tag) -> str:
        """Process table section."""
        lines = []

        # Extract table title
        title_tag = table_section.find(["h3", "h4"], class_="obj_head")
        title = ""
        if title_tag:
            title = self._clean_text(title_tag.get_text())
            lines.append(f"### {title}")
            lines.append("")

        # Extract and convert table first
        table_tag = table_section.find("table")
        if table_tag:
            table_md = self._convert_table_to_markdown(table_tag)
            if table_md:
                lines.append(table_md)

        # Extract table caption and place below table
        # Look for caption in both 'caption' class and 'tw-foot' class
        caption_div = table_section.find("div", class_="caption")
        if not caption_div:
            caption_div = table_section.find("div", class_="tw-foot")

        if caption_div:
            caption = self._clean_text(caption_div.get_text())
            # Extract table number from title if available
            table_number = ""
            if title:
                # Try to extract table number from title like "Table 1." or "Table 1:"
                match = re.search(r"Table\s+(\d+)", title, re.IGNORECASE)
                if match:
                    table_number = match.group(1)

            # Add blank line before caption to prevent merging with table
            lines.append("")
            if table_number:
                lines.append(f"Table {table_number} Caption: {caption}")
            else:
                lines.append(f"Table Caption: {caption}")
            lines.append("")

        return "\n".join(lines)

    def _convert_table_to_markdown(self, table_tag: Tag) -> str:
        """Convert HTML table to markdown table."""
        rows = []
        max_cols = 0

        # Process header
        thead = table_tag.find("thead")
        header_rows = []
        if thead:
            header_rows = thead.find_all("tr")
            for row in header_rows:
                cells = row.find_all(["th", "td"])
                row_data = []
                for cell in cells:
                    # Handle colspan
                    colspan = int(cell.get("colspan", 1))
                    text = self._clean_text(cell.get_text()).strip()
                    # Escape pipe characters and clean up text
                    text = text.replace("|", "\\|").replace("\n", " ")
                    # Handle empty cells
                    if not text:
                        text = " "
                    row_data.append(text)
                    # Add empty cells for colspan > 1
                    for _ in range(colspan - 1):
                        row_data.append("")
                rows.append(row_data)
                max_cols = max(max_cols, len(row_data))

        # Process body
        tbody = table_tag.find("tbody")
        if tbody:
            body_rows = tbody.find_all("tr")
        else:
            # No explicit tbody, get all tr elements
            body_rows = table_tag.find_all("tr")
            if header_rows:
                # Remove header rows if we already processed them
                body_rows = body_rows[len(header_rows) :]

        for row in body_rows:
            cells = row.find_all(["td", "th"])
            row_data = []
            for cell in cells:
                # Handle colspan (markdown doesn't support rowspan)
                colspan = int(cell.get("colspan", 1))
                text = self._clean_text(cell.get_text()).strip()
                # Escape pipe characters and clean up text
                text = text.replace("|", "\\|").replace("\n", " ")
                # Handle empty cells
                if not text:
                    text = " "
                row_data.append(text)
                # Add empty cells for colspan > 1
                for _ in range(colspan - 1):
                    row_data.append("")
            rows.append(row_data)
            max_cols = max(max_cols, len(row_data))

        if not rows:
            return ""

        # Normalize all rows to have same number of columns
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Convert to markdown
        lines = []

        if rows:
            # Determine if we have proper headers
            has_proper_header = thead and header_rows

            if has_proper_header:
                header = rows[0]
                data_rows = rows[1:]
            else:
                # For tables without proper headers, use first row if it looks like headers
                # Otherwise create generic headers
                first_row = rows[0]
                # Check if first row looks like headers (contains non-numeric text)
                if any(
                    not cell.replace(".", "").replace("-", "").isdigit()
                    and cell.strip()
                    for cell in first_row
                ):
                    header = first_row
                    data_rows = rows[1:]
                else:
                    # Create descriptive headers based on content patterns
                    header = []
                    for i in range(max_cols):
                        if i == 0:
                            header.append("Category")
                        else:
                            header.append(f"Value {i}")
                    data_rows = rows

            # Header row
            lines.append("| " + " | ".join(header) + " |")

            # Separator row - use consistent width for better formatting
            separator_cols = ["---"] * len(header)
            lines.append("| " + " | ".join(separator_cols) + " |")

            # Data rows
            for row in data_rows:
                # Ensure row has same length as header
                while len(row) < len(header):
                    row.append("")
                lines.append("| " + " | ".join(row[: len(header)]) + " |")

        return "\n".join(lines)

    def _process_figure(self, figure_tag: Tag) -> str:
        """Process figure element."""
        lines = []

        # Extract figure title
        title_tag = figure_tag.find(["h3", "h4"], class_="obj_head")
        if title_tag:
            title = self._clean_text(title_tag.get_text())
            lines.append(f"### {title}")
            lines.append("")

        # Extract image
        img_tag = figure_tag.find("img")
        if img_tag and img_tag.get("src"):
            src = img_tag["src"]
            alt = img_tag.get("alt", "Figure")

            # Ensure absolute URL
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = self.base_url + src

            lines.append(f"![{alt}]({src})")
            lines.append("")

            # Add zoom link if available
            zoom_link = figure_tag.find("a", class_="tileshop")
            if zoom_link and zoom_link.get("href"):
                lines.append(f"[View larger image]({zoom_link['href']})")
                lines.append("")

        # Extract figure caption
        caption_tag = figure_tag.find("figcaption")
        if caption_tag:
            caption = self._clean_text(caption_tag.get_text())
            lines.append(caption)
            lines.append("")

        return "\n".join(lines)

    def _extract_references(self) -> str:
        """Extract references section."""
        ref_section = self.soup.find("section", class_="ref-list")
        if not ref_section:
            return ""

        lines = ["## References", ""]

        # Find reference list
        ref_list = ref_section.find(["ul", "ol"], class_="ref-list")
        if ref_list:
            for i, li in enumerate(ref_list.find_all("li"), 1):
                ref_text = self._process_reference(li, i)
                if ref_text:
                    lines.append(ref_text)
                    lines.append("")

        return "\n".join(lines)

    def _process_reference(self, ref_item: Tag, ref_num: int) -> str:
        """Process individual reference."""
        parts = []

        # Add reference number
        parts.append(f"{ref_num}.")

        # Extract citation text
        cite_tag = ref_item.find("cite")
        if cite_tag:
            cite_text = self._clean_text(cite_tag.get_text())
            parts.append(cite_text)
        else:
            # No cite tag, use all text except links
            text_parts = []
            for element in ref_item.children:
                if isinstance(element, NavigableString):
                    text_parts.append(str(element))
                elif isinstance(element, Tag) and element.name != "a":
                    text_parts.append(element.get_text())
            cite_text = self._clean_text("".join(text_parts))
            parts.append(cite_text)

        # Extract links (DOI, PMC, PubMed)
        links = ref_item.find_all("a", href=True)
        link_parts = []
        for link in links:
            href = link["href"]
            text = link.get_text().strip()

            if "doi.org" in href or text.upper() == "DOI":
                link_parts.append(f"[DOI]({href})")
            elif "pmc.ncbi.nlm.nih.gov" in href or text.upper() == "PMC":
                link_parts.append(f"[PMC]({href})")
            elif "pubmed.ncbi.nlm.nih.gov" in href or text.upper() == "PUBMED":
                link_parts.append(f"[PubMed]({href})")
            else:
                link_parts.append(f"[{text}]({href})")

        if link_parts:
            parts.append(" " + " | ".join(link_parts))

        return " ".join(parts)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove extra whitespace
        text = text.strip()

        return text

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up final markdown output."""
        # Remove excessive blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        # Ensure document ends with single newline
        markdown = markdown.strip() + "\n"

        return markdown


def main():
    """Example usage of the converter."""
    import sys

    if len(sys.argv) != 3:
        logger.error(
            "Usage: python html_to_markdown.py <input_html_file> <output_md_file>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    converter = PubMedHTMLToMarkdownConverter()

    try:
        markdown_content = converter.convert_file(input_file)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.debug(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        logger.error(f"Error converting file: {e}")
        sys.exit(1)


def run_local():
    converter = PubMedHTMLToMarkdownConverter()
    input_files = os.listdir("data/html")
    for file in tqdm.tqdm(input_files, desc="Converting HTML to Markdown"):
        converter = PubMedHTMLToMarkdownConverter()
        markdown_content = converter.convert_file(f"data/html/{file}")
        os.makedirs("data/markdown", exist_ok=True)
        with open(
            f"data/markdown/{file.replace('.html', '.md')}", "w", encoding="utf-8"
        ) as f:
            f.write(markdown_content)
    logger.info(f"Converted {len(input_files)} HTML files to Markdown")


def single_file(pmcid: str):
    converter = PubMedHTMLToMarkdownConverter()
    markdown_content = converter.convert_file(f"data/html/{pmcid}.html")
    os.makedirs("data/markdown", exist_ok=True)
    with open(f"data/markdown/{pmcid}.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info(f"Converted {pmcid} to Markdown")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args:
        single_file(args[0])
    else:
        run_local()
