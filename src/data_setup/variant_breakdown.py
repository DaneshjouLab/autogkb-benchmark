"""
Goal:
- benchmark jsons --> all variants mentioned
"""

import json
import polars as pl
import pyarrow.parquet as pq
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
from datetime import datetime

class SingleArticleVariants(BaseModel):
    """
    Represents a data model for storing variant information extracted from a single article.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        article_title (str): The title of the article.
        article_path (str): The path to the markdown file of the article.
        variants (list[str]): A list of processed variant strings found in the article.
        raw_variants (list[str]): A list of raw variant strings as extracted directly from annotations.
    """
    pmcid: str
    pmid: str    
    article_title: str
    article_path: str
    variants: list[str]
    raw_variants: list[str]

def get_markdown_from_pmcid(pmcid: str) -> str:
    """
    Retrieves the markdown path in string format of an article given its PubMed Central ID (PMCID).

    Args:
        pmcid (str): The PubMed Central ID of the article.

    Returns:
        str: The markdown path in string format of the article. Returns an empty string
             if the file cannot be found or processed, logging a warning.
    """
    markdown_path = Path("data") / "articles" / f"{pmcid}.md"
    try:
        if not markdown_path.is_file():
            logger.warning(f"Warning: Could not find file {markdown_path}")
            return ""
        return str(markdown_path)
    except Exception as e:
        logger.warning(f"Warning: Could not process file {markdown_path}: {e}")
        return ""

def get_file_variants(file_path: Path | str, deduplicate: bool = True, ungroup: bool = True) -> SingleArticleVariants:
    """
    Extracts all mentioned variants from a single JSON article file.

    This function reads a JSON file, extracts variant/haplotype information from
    'var_drug_ann', 'var_pheno_ann', and 'var_fa_ann' sections. It can also
    deduplicate and ungroup variants based on the provided flags.

    Args:
        file_path (Path | str): The path to the JSON file containing article annotations.
        deduplicate (bool, optional): If True, removes duplicate variants. Defaults to True.
        ungroup (bool, optional): If True, splits comma-separated grouped variants into
                                   individual variants. Defaults to True.

    Returns:
        SingleArticleVariants: An object containing the article's metadata and
                               the extracted variants. Returns an empty
                               SingleArticleVariants object if the file cannot
                               be processed, logging a warning.
    """
    # from json file, extract all the variants from variant/haplotypes
    # Convert file to path
    if isinstance(file_path, str):
        file_path = Path(file_path)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Warning: Could not process file {file_path}: {e}")
        return SingleArticleVariants(pmcid="", pmid="", article_title="", article_path="", variants=[], raw_variants=[])
    variants: list[str] = []
    pmcid = data['pmcid']
    pmid = data['pmid']
    article_title = data['title']
    article_path = get_markdown_from_pmcid(pmcid)
    for item in data['var_drug_ann']:
        variants.append(item['Variant/Haplotypes'])
    for item in data['var_pheno_ann']:
        variants.append(item['Variant/Haplotypes'])
    for item in data['var_fa_ann']:
        variants.append(item['Variant/Haplotypes'])

    if deduplicate:
        variants = list(set(variants))
    if ungroup:
        variants = [variant.split(',') for variant in variants]
        variants = [variant for sublist in variants for variant in sublist]
    return SingleArticleVariants(pmcid=pmcid, pmid=pmid, article_title=article_title, article_path=article_path, variants=variants, raw_variants=variants)

def get_dir_variants(dir_path: str, deduplicate: bool = True, ungroup: bool = True) -> list[SingleArticleVariants]:
    """
    Processes all JSON article files within a specified directory to extract variants.

    This function iterates through all JSON files in the given directory,
    applies `get_file_variants` to each, and aggregates the results.

    Args:
        dir_path (str): The path to the directory containing JSON annotation files.
        deduplicate (bool, optional): If True, variants extracted from each file
                                     will be deduplicated. Defaults to True.
        ungroup (bool, optional): If True, splits comma-separated grouped variants
                                  into individual variants for each file. Defaults to True.

    Returns:
        list[SingleArticleVariants]: A list of `SingleArticleVariants` objects, each
                                     representing the variants found in one article file.

    Raises:
        ValueError: If the provided `dir_path` does not exist or is not a directory.
    """
    # Validate path
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    variant_list: list[SingleArticleVariants] = []

    # Loop through json files in the directory
    for file in directory.glob("*.json"):
        try:
            article_variants = get_file_variants(file, deduplicate=deduplicate, ungroup=ungroup)
            variant_list.append(article_variants)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")
    return variant_list

def get_benchmark_variants(save_parquet: bool = True) -> list[SingleArticleVariants]:
    """
    Retrieves and processes variants from the benchmark annotation directory.

    This function specifically targets the 'data/benchmark_annotations' directory,
    deduplicating and ungrouping variants by default.

    Args:
        save_parquet (bool, optional): If True, saves the results to a parquet file
                                       at 'data/benchmark_variants.parquet'. Defaults to True.

    Returns:
        list[SingleArticleVariants]: A list of `SingleArticleVariants` objects
                                     representing the processed benchmark variants.
    """
    benchmark_dir = "data/benchmark_annotations"
    logger.info(f"Loading variants from benchmark dir {benchmark_dir}")
    benchmark_variants = get_dir_variants(benchmark_dir, deduplicate=True, ungroup=True)

    if save_parquet:
        # Convert list of SingleArticleVariants to a polars DataFrame
        df = pl.DataFrame([variant.model_dump() for variant in benchmark_variants])
        output_path = Path("data") / "benchmark_variants.parquet"
        # Use pyarrow for writing parquet (more reliable)
        pq.write_table(df.to_arrow(), output_path)
        logger.info(f"Saved {len(benchmark_variants)} articles to {output_path}")

    return benchmark_variants


if __name__ == "__main__":
    benchmark_variants = get_benchmark_variants(save_parquet=True)
    print(f"Found {len(benchmark_variants)} articles with variants")

    # Verify the parquet file is loadable
    table = pq.read_table("data/benchmark_variants.parquet")
    df = pl.from_arrow(table)
    print(f"Loaded parquet with {len(df)} rows and columns: {df.columns}")
    print(df)






