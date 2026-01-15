"""
Generate a summary of the drugs in the benchmark vs the proposed
"""

import json
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger

class SingleArticleDrugs(BaseModel):
    title: str
    pmcid: str
    drugs: list[str]

def null_to_empty(value: str | None) -> str:
    """Converts null string values to empty strings"""
    if value is None or value == "null" or value == "":
        return ""
    return value


def drugs_from_file(file_path: str) -> SingleArticleDrugs:
    """Gets all the (unique) mentioned drugs in a file"""
    # from json file, extract all the drugs from variant/drug annotations
    with open(file_path, 'r') as f:
        data = json.load(f)
    drugs: list[str] = []
    pmcid = data['pmcid']
    article_title = data['title']
    for item in data['var_drug_ann']:
        drugs.append(null_to_empty(item['Drug(s)']))
    for item in data['var_pheno_ann']:
        drugs.append(null_to_empty(item['Drug(s)']))
    for item in data['var_fa_ann']:
        drugs.append(null_to_empty(item['Drug(s)']))
    return SingleArticleDrugs(title=article_title, pmcid=pmcid, drugs=drugs)

class SingleArticleDrugReport(BaseModel):
    title: str
    pmcid: str
    benchmark_drugs: list[str]
    proposed_drugs: list[str]

class DrugCoverageReport(BaseModel):
    run_name: str
    timestamp: str
    benchmark_drugs: list[str]
    proposed_drugs: list[str]
    article_drug_breakdown: list[SingleArticleDrugReport]

def get_drugs_from_dir(dir_path: str) -> tuple[list[SingleArticleDrugs], list[str]]:
    """Loops through drugs_from_file for a whole directory
    Create a dictionary of pmcid --> [drugs]
    """
    # Validate path
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    drug_list: list[SingleArticleDrugs] = []
    all_drugs: list[str] = []
    for file in directory.glob("*.json"):
        try:
            article_drugs = drugs_from_file(str(file))
            all_drugs.extend(article_drugs.drugs)
            drug_list.append(article_drugs)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")
    return drug_list, all_drugs

def get_article_reports(benchmark_dir: str, proposed_dir: str) -> tuple[list[SingleArticleDrugReport], list[str], list[str]]:
    benchmark_drugs, benchmark_all_drugs = get_drugs_from_dir(benchmark_dir)
    proposed_drugs, proposed_all_drugs = get_drugs_from_dir(proposed_dir)
    proposed_by_pmcid = {article.pmcid: article.drugs for article in proposed_drugs}
    article_reports: list[SingleArticleDrugReport] = []
    for article in benchmark_drugs:
        article_report = SingleArticleDrugReport(
            title=article.title,
            pmcid=article.pmcid,
            benchmark_drugs=article.drugs,
            proposed_drugs=proposed_by_pmcid.get(article.pmcid, []),
        )
        article_reports.append(article_report)
    logger.info(f"Found {len(article_reports)} articles")
    logger.info(f"Found {len(benchmark_all_drugs)} benchmark drugs")
    logger.info(f"Found {len(proposed_all_drugs)} proposed drugs")

    # Remove empty strings
    benchmark_all_drugs = [drug for drug in benchmark_all_drugs if drug]
    proposed_all_drugs = [drug for drug in proposed_all_drugs if drug]

    # Deduplicate all drug lists
    benchmark_all_drugs = list(dict.fromkeys(benchmark_all_drugs))
    proposed_all_drugs = list(dict.fromkeys(proposed_all_drugs))

    return article_reports, benchmark_all_drugs, proposed_all_drugs

def proposed_drug_summary(proposed_annotations_dir: str, benchmark_dir: str = "data/benchmark_annotations", run_name: str = "") -> dict[str]:
    """Generates report for all drugs in data/coverage/<time-stamp>_drugs_<run_name>.json"""
    article_reports, benchmark_drugs, proposed_drugs = get_article_reports(benchmark_dir, proposed_annotations_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_name = timestamp
    if not run_name:
        run_name = proposed_annotations_dir
    output_file_name += f"_drug_report.json"
    output_path = Path("data") / "coverage" / "drug_reports" / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = DrugCoverageReport(
        run_name=run_name,
        timestamp=timestamp,
        benchmark_drugs=benchmark_drugs,
        proposed_drugs=proposed_drugs,
        article_drug_breakdown=article_reports
    )
    with open(output_path, "w") as f:
        try :
            json.dump(summary.model_dump(), f, indent=2)
            logger.info(f"Wrote drug summary to {output_path}")
        except Exception as e:
            logger.error(f"Error writing to file {output_path}: {e}")

if __name__ == "__main__":
    proposed_annotations_dir = "data/proposed_annotations"
    proposed_drug_summary(proposed_annotations_dir)



