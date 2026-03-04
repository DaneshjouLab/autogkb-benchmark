"""
Goal: Keep a record of the PMIDs that have been converted to pmcid, html, and markdown
Store the record in a json file (data/records.json)
"""

import os
from typing import List
import re
from loguru import logger
import pandas as pd


def get_scraped_pmids(update: bool = False) -> List[str]:
    """
    Get a list of all the PMIDs in the records.csv file

    Args:
        update (bool): If True, create a new records.csv file
    Returns:
        List[str]: A list of all the PMIDs in the records.csv file
    """
    if update:
        records = create_records()
    else:
        records = pd.read_csv(os.path.join("data", "records.csv"))
    pmid_list = records["pmid"].tolist()
    return pmid_list


def get_scraped_pmcids(update: bool = False) -> List[str]:
    """
    Get a list of all the PMCIDs in the records.csv file

    Args:
        update (bool): If True, create a new records.csv file
    Returns:
        List[str]: A list of all the PMIDs in the records.csv file
    """
    if update:
        records = create_records()
    else:
        records = pd.read_csv(os.path.join("data", "records.csv"))
    pmcid_list = records["pmcid"].tolist()
    return pmcid_list


def parse_markdown_metadata(markdown_text: str) -> dict:
    """
    Extract PMID, PMCID, and URL from a markdown text.

    Args:
        markdown_text (str): The markdown text to parse

    Returns:
        dict: A dictionary containing extracted metadata
    """
    # Dictionary to store extracted metadata
    metadata = {}

    # Regular expressions for extraction
    pmcid_pattern = r"\*\*PMCID:\*\*\s*([^\n]+)"
    pmid_pattern = r"\*\*PMID:\*\*\s*([^\n]+)"
    url_pattern = r"\*\*URL:\*\*\s*([^\n]+)"

    # Extract PMCID
    pmcid_match = re.search(pmcid_pattern, markdown_text)
    if pmcid_match:
        metadata["pmcid"] = pmcid_match.group(1).strip()

    # Extract PMID
    pmid_match = re.search(pmid_pattern, markdown_text)
    if pmid_match:
        metadata["pmid"] = pmid_match.group(1).strip()

    # Extract URL
    url_match = re.search(url_pattern, markdown_text)
    if url_match:
        metadata["url"] = url_match.group(1).strip()

    return metadata


def validate_records(records: pd.DataFrame) -> pd.DataFrame:
    """
    Check if any of the records in the records are missing required fields.

    Args:
        records (pd.DataFrame): DataFrame containing the record map

    Returns:
        pd.DataFrame: DataFrame containing only the records with missing fields
    """
    # Check for missing values in required columns
    missing_mask = records[["pmid", "pmcid", "url"]].isna().any(axis=1)
    missing_records = records[missing_mask]

    if not missing_records.empty:
        logger.warning(f"Found {len(missing_records)} records with missing fields")
        for _, row in missing_records.iterrows():
            missing_fields = [
                col for col in ["pmid", "pmcid", "url"] if pd.isna(row[col])
            ]
            logger.warning(
                f"Record {row['markdown_path']} is missing: {', '.join(missing_fields)}"
            )

    return missing_records


def create_records() -> pd.DataFrame:
    """
    Get a list of all the markdown files in the data/articles directory
    Extract PMID, PMCID, and URL to create a csv table:
    PMID,PMCID,URL,markdown_path

    Returns:
        pd.DataFrame: DataFrame containing the record map
    """
    records = []
    markdown_path = os.path.join("data", "markdown")

    for file in os.listdir(markdown_path):
        if file.endswith(".md"):
            row = {
                "pmid": None,
                "pmcid": None,
                "markdown_path": f"{markdown_path}/{file}",
                "url": None,
            }
            metadata = parse_markdown_metadata(
                open(f"{markdown_path}/{file}", "r").read()
            )
            if metadata["pmid"] is not None:
                row["pmid"] = metadata["pmid"]
            if metadata["pmcid"] is not None:
                row["pmcid"] = metadata["pmcid"]
            if metadata["url"] is not None:
                row["url"] = metadata["url"]
            records.append(row)

    # Create DataFrame from list of records
    records = pd.DataFrame(records)

    missing_records = validate_records(records)
    if len(missing_records) > 0:
        logger.warning(f"Missing records: {missing_records}")
    logger.info("Finished processing records")

    # Save record map to a CSV
    records_path = os.path.join("data", "records.csv")
    records.to_csv(records_path, index=False)

    logger.info(f"Record map saved to {records_path}")
    return records


if __name__ == "__main__":
    create_records()
