"""
PMCID --> Full Article Text (HTML)
This uses a standard get request with a user agent and accept header to fetch the article text.
"""

import argparse
import requests
from loguru import logger
from typing import List, Optional, Union


def get_html_from_pmcid(pmcid: str) -> Optional[str]:
    """
    Given a PMCID, fetch the full article text from the NCBI website.
    Returns the HTML text of the article in string format from the url
    https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/?report=classic

    Args:
        pmcid (str): The PMCID to fetch

    Returns:
        Optional[str]: The article html text if successful, None if there was an error
    """
    if not isinstance(pmcid, str):
        logger.error("pmcid must be a string")
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/?report=classic"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an exception for 4XX/5XX status codes
        return response.text
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred for PMCID {pmcid}: {str(e)}")
        if response.text:
            logger.error(f"Server response: {response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred for PMCID {pmcid}: {str(e)}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out for PMCID {pmcid}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while fetching PMCID {pmcid}: {str(e)}")
        return None


def main():
    """Entry point for markdown from pmid"""
    parser = argparse.ArgumentParser(
        description="Fetch and save article text from NCBI"
    )
    parser.add_argument("--pmcid", type=str, help="PMCID of the article to fetch")
    parser.add_argument(
        "--save_dir",
        default="data/articles",
        type=str,
        help="Path to save the article text",
    )
    args = parser.parse_args()

    if not args.pmcid:
        parser.error("--pmcid is required")

    text = get_html_from_pmcid(args.pmcid)
    if text is not None:
        with open(f"{args.save_dir}/{args.pmcid}.html", "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
