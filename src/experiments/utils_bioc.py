"""
BioC API integration for fetching supplementary material text from PMC articles.

The BioC API provides pre-processed text from supplementary PDFs for articles
in the PMC Open Access subset. This is significantly easier than downloading
and parsing PDFs ourselves.

API Documentation: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/FAIR-SMART/
"""

import json
import time
from pathlib import Path

import requests
from loguru import logger

BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/supplmat.cgi"
CACHE_DIR = Path("data/cache/bioc_supplements")


def _get_cache_path(pmcid: str) -> Path:
    """Get the cache file path for a PMCID."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{pmcid}.json"


def fetch_bioc_supplement(pmcid: str, use_cache: bool = True) -> str | None:
    """
    Fetch supplementary material text from the BioC API.

    Args:
        pmcid: PubMed Central ID (e.g., "PMC6435416")
        use_cache: Whether to use cached results (default: True)

    Returns:
        Extracted text from all supplementary materials, or None if not available.
    """
    # Check cache first
    cache_path = _get_cache_path(pmcid)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
                if cached.get("text"):
                    return cached["text"]
                elif cached.get("not_available"):
                    return None
        except (json.JSONDecodeError, KeyError):
            pass  # Cache corrupted, refetch

    # Fetch from BioC API
    url = f"{BIOC_BASE_URL}/BioC_JSON/{pmcid}/All"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            logger.debug(f"BioC API returned {response.status_code} for {pmcid}")
            _save_cache(cache_path, {"not_available": True})
            return None

        content = response.text
        if not content or len(content) < 50:
            logger.debug(f"BioC API returned empty/minimal content for {pmcid}")
            _save_cache(cache_path, {"not_available": True})
            return None

        # Parse BioC JSON and extract text
        data = json.loads(content)
        all_text = _extract_text_from_bioc(data)

        if all_text:
            _save_cache(cache_path, {"text": all_text})
            return all_text
        else:
            _save_cache(cache_path, {"not_available": True})
            return None

    except requests.RequestException as e:
        logger.warning(f"BioC API request failed for {pmcid}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"BioC API returned invalid JSON for {pmcid}: {e}")
        return None


def _extract_text_from_bioc(data: list | dict) -> str:
    """
    Extract all text passages from BioC JSON response.

    BioC JSON structure:
    [
        {
            "source": "BioC",
            "documents": [
                {
                    "id": "filename.pdf",
                    "passages": [
                        {"offset": 0, "text": "...", "annotations": []}
                    ]
                }
            ]
        }
    ]
    """
    all_text = []

    # Handle both list and dict responses
    collections = data if isinstance(data, list) else [data]

    for collection in collections:
        if not isinstance(collection, dict):
            continue

        for doc in collection.get("documents", []):
            if not isinstance(doc, dict):
                continue

            for passage in doc.get("passages", []):
                if not isinstance(passage, dict):
                    continue

                text = passage.get("text", "")
                if text and isinstance(text, str):
                    all_text.append(text)

    return "\n\n".join(all_text)


def _save_cache(cache_path: Path, data: dict) -> None:
    """Save data to cache file."""
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


def get_bioc_supplement_cached(pmcid: str) -> str | None:
    """
    Get supplementary material text, using cache when available.

    This is the recommended function to use in extraction pipelines.
    """
    return fetch_bioc_supplement(pmcid, use_cache=True)


def prefetch_bioc_supplements(pmcids: list[str], delay: float = 0.2) -> dict[str, bool]:
    """
    Prefetch and cache BioC supplements for a list of PMCIDs.

    Args:
        pmcids: List of PMCIDs to prefetch
        delay: Delay between API calls to be nice to the server

    Returns:
        Dict mapping PMCID to whether supplement was available
    """
    results = {}

    for i, pmcid in enumerate(pmcids):
        cache_path = _get_cache_path(pmcid)

        if cache_path.exists():
            # Already cached
            with open(cache_path) as f:
                cached = json.load(f)
                results[pmcid] = bool(cached.get("text"))
            continue

        # Fetch from API
        text = fetch_bioc_supplement(pmcid, use_cache=False)
        results[pmcid] = text is not None

        if (i + 1) % 10 == 0:
            logger.info(f"Prefetched {i + 1}/{len(pmcids)} supplements")

        time.sleep(delay)

    available = sum(1 for v in results.values() if v)
    logger.info(f"Prefetch complete: {available}/{len(pmcids)} have BioC supplements")

    return results


if __name__ == "__main__":
    # Test with a known article
    test_pmcid = "PMC6435416"
    print(f"Testing BioC fetch for {test_pmcid}...")

    text = fetch_bioc_supplement(test_pmcid, use_cache=False)

    if text:
        print(f"✓ Got {len(text)} characters of supplement text")
        print(f"\nFirst 500 chars:\n{text[:500]}")

        # Check for expected content
        if "*17" in text and "*41" in text:
            print("\n✓ Found expected CYP2D6 star alleles in supplement!")
    else:
        print("✗ No supplement available")
