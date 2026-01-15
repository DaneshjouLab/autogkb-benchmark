"""
Goal:
- benchmark jsons --> all variants mentioned
"""

import json
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from loguru import logger

class SingleArticleVariants(BaseModel):
    """
    Represents a data model for storing variant information extracted from a single article.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        article_title (str): The title of the article.
        article_text (str): The full markdown text content of the article.
        variants (list[str]): A list of processed variant strings found in the article.
        raw_variants (list[str]): A list of raw variant strings as extracted directly from annotations.
    """
    pmcid: str
    pmid: str    
    article_title: str
    article_text: str
    variants: list[str]
    raw_variants: list[str]

def get_markdown_from_pmcid(pmcid: str) -> str:
    """
    Retrieves the markdown content of an article given its PubMed Central ID (PMCID).

    Args:
        pmcid (str): The PubMed Central ID of the article.

    Returns:
        str: The markdown text content of the article. Returns an empty string
             if the file cannot be found or processed, logging a warning.
    """
    markdown_path = Path("data") / "articles" / f"{pmcid}.md"
    try:
        with open(markdown_path, 'r') as f:
            markdown_text = f.read()
    except Exception as e:
        logger.warning(f"Warning: Could not process file {markdown_path}: {e}")
        return ""
    return markdown_text

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
        return SingleArticleVariants(pmcid="", pmid="", article_title="", article_text="", variants=[], raw_variants=[])
    variants: list[str] = []
    pmcid = data['pmcid']
    pmid = data['pmid']
    article_title = data['title']
    article_text = get_markdown_from_pmcid(pmcid)
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
    return SingleArticleVariants(pmcid=pmcid, pmid=pmid, article_title=article_title, article_text=article_text, variants=variants, raw_variants=variants)

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

def get_benchmark_variants() -> list[SingleArticleVariants]:
    """
    Retrieves and processes variants from the benchmark annotation directory.

    This function specifically targets the 'data/benchmark_annotations' directory,
    deduplicating and ungrouping variants by default.

    Returns:
        list[SingleArticleVariants]: A list of `SingleArticleVariants` objects
                                     representing the processed benchmark variants.
    """
    benchmark_dir = "data/benchmark_annotations"
    benchmark_variants = get_dir_variants(benchmark_dir, deduplicate=True, ungroup=True)
    return benchmark_variants

if __name__ == "__main__":

    benchmark_variants = get_benchmark_variants()
    print(f"Found {len(benchmark_variants)} benchmark variants")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path("scratch") / f"benchmark_variants_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([variant.model_dump() for variant in benchmark_variants], f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")
    # pmcid4916189 is bad, prints to text like "Once-Daily Efavirenz 400 and 600\u00a0mg in Treatment-Na\u00efve HIV-Infected Patients"






