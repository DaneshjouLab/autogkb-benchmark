"""
BioC supplement fetching — re-exported from the pubmed-downloader package.

The canonical implementation lives in PubMedDownloader/pubmed_downloader/utils_bioc.py.
Import from there to get the full API (including format_supplement_as_markdown).
"""

from pubmed_downloader.utils_bioc import (
    fetch_bioc_supplement,
    get_bioc_supplement_cached,
    prefetch_bioc_supplements,
)

__all__ = [
    "fetch_bioc_supplement",
    "get_bioc_supplement_cached",
    "prefetch_bioc_supplements",
]
