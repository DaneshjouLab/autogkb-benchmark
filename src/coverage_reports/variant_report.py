"""
Generate a summary of the variants in the benchmark vs the proposed
"""

import json
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger

class SingleArticleVariants(BaseModel):
    title: str
    pmcid: str
    variants: list[str]

def variants_from_file(file_path: str) -> SingleArticleVariants:
    """Gets all the (unique) mentioned variants in a file"""
    # from json file, extract all the variants from variant/haplotypes
    with open(file_path, 'r') as f:
        data = json.load(f)
    variants: list[str] = []
    pmcid = data['pmcid']
    article_title = data['title']
    for item in data['var_drug_ann']:
        variants.append(item['Variant/Haplotypes'])
    for item in data['var_pheno_ann']:
        variants.append(item['Variant/Haplotypes'])
    for item in data['var_fa_ann']:
        variants.append(item['Variant/Haplotypes'])

    return SingleArticleVariants(title=article_title, pmcid=pmcid, variants=variants)

class SingleArticleVariantReport(BaseModel):
    title: str
    pmcid: str
    benchmark_variants: list[str]
    proposed_variants: list[str]

class VariantCoverageReport(BaseModel):
    run_name: str
    timestamp: str
    benchmark_variants: list[str]
    proposed_variants: list[str]
    article_variant_breakdown: list[SingleArticleVariantReport]

def get_variants_from_dir(dir_path: str) -> tuple[list[SingleArticleVariants], list[str]]:
    """Loops through variants_from_file for a whole directory
    Create a dictionary of pmcid --> [variants]
    """
    # Validate path
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    variant_list: list[SingleArticleVariants] = []
    all_variants: list[str] = []
    for file in directory.glob("*.json"):
        try:
            article_variants = variants_from_file(str(file))
            all_variants.extend(article_variants.variants)
            variant_list.append(article_variants)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")
    all_variants = list(dict.fromkeys(all_variants))
    return variant_list, all_variants

def get_article_reports(benchmark_dir: str, proposed_dir: str) -> tuple[list[SingleArticleVariantReport], list[str], list[str]]:
    benchmark_variants, benchmark_all_variants = get_variants_from_dir(benchmark_dir)
    proposed_variants, proposed_all_variants = get_variants_from_dir(proposed_dir)
    proposed_by_pmcid = {article.pmcid: article.variants for article in proposed_variants}
    article_reports: list[SingleArticleVariantReport] = []
    for article in benchmark_variants:
        article_report = SingleArticleVariantReport(
            title=article.title,
            pmcid=article.pmcid,
            benchmark_variants=article.variants,
            proposed_variants=proposed_by_pmcid.get(article.pmcid, []),
        )
        article_reports.append(article_report)
    
    # Deduplicate all variant lists
    benchmark_all_variants = list(dict.fromkeys(benchmark_all_variants))
    proposed_all_variants = list(dict.fromkeys(proposed_all_variants))
    
    logger.info(f"Found {len(article_reports)} articles")
    logger.info(f"Found {len(benchmark_all_variants)} benchmark variants")
    logger.info(f"Found {len(proposed_all_variants)} proposed variants")
    return article_reports, benchmark_all_variants, proposed_all_variants

def proposed_variant_summary(proposed_annotations_dir: str, benchmark_dir: str = "data/benchmark_annotations", run_name: str = "") -> dict[str]:
    """Generates report for all variants in data/coverage_reports/variant_reports/<time-stamp>_variants_<run_name>.json"""
    article_reports, benchmark_variants, proposed_variants = get_article_reports(benchmark_dir, proposed_annotations_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_name = timestamp
    if not run_name:
        run_name = proposed_annotations_dir
    output_file_name += f"_variant_report.json"
    output_path = Path("data") / "coverage_reports" / "variant_reports" / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = VariantCoverageReport(
        run_name=run_name,
        timestamp=timestamp,
        benchmark_variants=benchmark_variants,
        proposed_variants=proposed_variants,
        article_variant_breakdown=article_reports
    )
    with open(output_path, "w") as f:
        try :
            json.dump(summary.model_dump(), f, indent=2)
            logger.info(f"Wrote variant summary to {output_path}")
        except Exception as e:
            logger.error(f"Error writing to file {output_path}: {e}")

if __name__ == "__main__":
    proposed_annotations_dir = "data/proposed_annotations"
    proposed_variant_summary(proposed_annotations_dir)



