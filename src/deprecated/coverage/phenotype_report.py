"""
Generate a summary of the phenotypes in the benchmark vs the proposed
"""

import json
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from loguru import logger


class SingleArticlePhenotypes(BaseModel):
    title: str
    pmcid: str
    phenotypes: list[str]


def null_to_empty(value: str | None) -> str:
    """Converts null string values to empty strings"""
    if value is None or value == "null" or value == "":
        return ""
    return value


def phenotypes_from_file(file_path: str) -> SingleArticlePhenotypes:
    """Gets all the (unique) mentioned phenotypes in a file"""
    # from json file, extract all the phenotypes from all annotation types
    with open(file_path, "r") as f:
        data = json.load(f)
    phenotypes: list[str] = []
    pmcid = data["pmcid"]
    article_title = data["title"]

    # Extract from var_drug_ann - uses "Population Phenotypes or diseases" field
    for item in data["var_drug_ann"]:
        phenotypes.append(null_to_empty(item.get("Population Phenotypes or diseases")))

    # Extract from var_pheno_ann - uses "Phenotype" field
    for item in data["var_pheno_ann"]:
        phenotypes.append(null_to_empty(item.get("Phenotype")))

    # Extract from var_fa_ann - check both fields
    for item in data["var_fa_ann"]:
        phenotypes.append(null_to_empty(item.get("Phenotype")))
        phenotypes.append(null_to_empty(item.get("Population Phenotypes or diseases")))

    return SingleArticlePhenotypes(
        title=article_title, pmcid=pmcid, phenotypes=phenotypes
    )


class SingleArticlePhenotypeReport(BaseModel):
    title: str
    pmcid: str
    benchmark_phenotypes: list[str]
    proposed_phenotypes: list[str]


class PhenotypeCoverageReport(BaseModel):
    run_name: str
    timestamp: str
    benchmark_phenotypes: list[str]
    proposed_phenotypes: list[str]
    article_phenotype_breakdown: list[SingleArticlePhenotypeReport]


def get_phenotypes_from_dir(
    dir_path: str,
) -> tuple[list[SingleArticlePhenotypes], list[str]]:
    """Loops through phenotypes_from_file for a whole directory
    Create a dictionary of pmcid --> [phenotypes]
    """
    # Validate path
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    phenotype_list: list[SingleArticlePhenotypes] = []
    all_phenotypes: list[str] = []
    for file in directory.glob("*.json"):
        try:
            article_phenotypes = phenotypes_from_file(str(file))
            all_phenotypes.extend(article_phenotypes.phenotypes)
            phenotype_list.append(article_phenotypes)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")
    return phenotype_list, all_phenotypes


def get_article_reports(
    benchmark_dir: str, proposed_dir: str
) -> tuple[list[SingleArticlePhenotypeReport], list[str], list[str]]:
    benchmark_phenotypes, benchmark_all_phenotypes = get_phenotypes_from_dir(
        benchmark_dir
    )
    proposed_phenotypes, proposed_all_phenotypes = get_phenotypes_from_dir(proposed_dir)
    proposed_by_pmcid = {
        article.pmcid: article.phenotypes for article in proposed_phenotypes
    }
    article_reports: list[SingleArticlePhenotypeReport] = []
    for article in benchmark_phenotypes:
        article_report = SingleArticlePhenotypeReport(
            title=article.title,
            pmcid=article.pmcid,
            benchmark_phenotypes=article.phenotypes,
            proposed_phenotypes=proposed_by_pmcid.get(article.pmcid, []),
        )
        article_reports.append(article_report)
    logger.info(f"Found {len(article_reports)} articles")
    logger.info(f"Found {len(benchmark_all_phenotypes)} benchmark phenotypes")
    logger.info(f"Found {len(proposed_all_phenotypes)} proposed phenotypes")

    # Remove empty strings
    benchmark_all_phenotypes = [
        phenotype for phenotype in benchmark_all_phenotypes if phenotype
    ]
    proposed_all_phenotypes = [
        phenotype for phenotype in proposed_all_phenotypes if phenotype
    ]

    # Deduplicate all phenotype lists
    benchmark_all_phenotypes = list(dict.fromkeys(benchmark_all_phenotypes))
    proposed_all_phenotypes = list(dict.fromkeys(proposed_all_phenotypes))

    return article_reports, benchmark_all_phenotypes, proposed_all_phenotypes


def proposed_phenotype_summary(
    proposed_annotations_dir: str,
    benchmark_dir: str = "data/benchmark_annotations",
    run_name: str = "",
) -> dict[str]:
    """Generates report for all phenotypes in data/coverage/<time-stamp>_phenotypes_<run_name>.json"""
    article_reports, benchmark_phenotypes, proposed_phenotypes = get_article_reports(
        benchmark_dir, proposed_annotations_dir
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_name = timestamp
    if not run_name:
        run_name = proposed_annotations_dir
    output_file_name += f"_phenotype_report.json"
    output_path = Path("data") / "coverage" / "phenotype_reports" / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = PhenotypeCoverageReport(
        run_name=run_name,
        timestamp=timestamp,
        benchmark_phenotypes=benchmark_phenotypes,
        proposed_phenotypes=proposed_phenotypes,
        article_phenotype_breakdown=article_reports,
    )
    with open(output_path, "w") as f:
        try:
            json.dump(summary.model_dump(), f, indent=2)
            logger.info(f"Wrote phenotype summary to {output_path}")
        except Exception as e:
            logger.error(f"Error writing to file {output_path}: {e}")


if __name__ == "__main__":
    proposed_annotations_dir = "data/proposed_annotations"
    proposed_phenotype_summary(proposed_annotations_dir)
