"""
Goal:
- Create a dataset of (pmcid, variant, summary sentence)
- Should follow the same variants that are created in the variant bench table
- Summary sentence should be the sentence that contains the variant from the benchmark annotations
- Should be a jsonl file stored in data/benchmark_v2/sentence_bench.jsonl
"""

import json
from pydantic import BaseModel
from pathlib import Path
from loguru import logger


class SingleSentenceEntry(BaseModel):
    """
    Represents a single sentence entry with its associated variant and article info.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        pmid (str): The PubMed ID of the article.
        variant (str): The variant/haplotype associated with this sentence.
        sentence (str): The summary sentence from the annotation.
        annotation_type (str): The type of annotation (var_drug_ann, var_pheno_ann, or var_fa_ann).
    """

    pmcid: str
    pmid: str
    variant: str
    sentence: str
    annotation_type: str


def get_file_sentences(file_path: Path | str) -> list[SingleSentenceEntry]:
    """
    Extracts all sentence entries from a single JSON article file.

    This function reads a JSON file and extracts (variant, sentence) pairs from
    'var_drug_ann', 'var_pheno_ann', and 'var_fa_ann' sections.

    Args:
        file_path (Path | str): The path to the JSON file containing article annotations.

    Returns:
        list[SingleSentenceEntry]: A list of sentence entries extracted from the file.
                                   Returns an empty list if the file cannot be processed.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Warning: Could not process file {file_path}: {e}")
        return []

    entries: list[SingleSentenceEntry] = []
    pmcid = data["pmcid"]
    pmid = data["pmid"]

    annotation_types = ["var_drug_ann", "var_pheno_ann", "var_fa_ann"]

    for ann_type in annotation_types:
        for item in data.get(ann_type, []):
            variant = item.get("Variant/Haplotypes", "")
            sentence = item.get("Sentence", "")

            if variant and sentence:
                # Handle grouped variants (comma-separated)
                individual_variants = [v.strip() for v in variant.split(",")]
                for individual_variant in individual_variants:
                    entries.append(
                        SingleSentenceEntry(
                            pmcid=pmcid,
                            pmid=pmid,
                            variant=individual_variant,
                            sentence=sentence,
                            annotation_type=ann_type,
                        )
                    )

    return entries


def get_dir_sentences(dir_path: str) -> list[SingleSentenceEntry]:
    """
    Processes all JSON article files within a specified directory to extract sentences.

    Args:
        dir_path (str): The path to the directory containing JSON annotation files.

    Returns:
        list[SingleSentenceEntry]: A list of all sentence entries from all files.

    Raises:
        ValueError: If the provided `dir_path` does not exist or is not a directory.
    """
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")

    all_entries: list[SingleSentenceEntry] = []

    for file in directory.glob("*.json"):
        try:
            file_entries = get_file_sentences(file)
            all_entries.extend(file_entries)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {file}: {e}")

    return all_entries


def get_benchmark_sentences(save_jsonl: bool = True) -> list[SingleSentenceEntry]:
    """
    Retrieves and processes sentences from the benchmark annotation directory.

    This function specifically targets the 'data/benchmark_annotations' directory.

    Args:
        save_jsonl (bool, optional): If True, saves the results to a JSONL file
                                     at 'data/benchmark_v2/sentence_bench.jsonl'. Defaults to True.

    Returns:
        list[SingleSentenceEntry]: A list of `SingleSentenceEntry` objects
                                   representing the benchmark sentences.
    """
    benchmark_dir = "data/benchmark_annotations"
    logger.info(f"Loading sentences from benchmark dir {benchmark_dir}")
    benchmark_sentences = get_dir_sentences(benchmark_dir)

    if save_jsonl:
        output_path = Path("data") / "benchmark_v2" / "sentence_bench.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for entry in benchmark_sentences:
                f.write(json.dumps(entry.model_dump()) + "\n")
        logger.info(f"Saved {len(benchmark_sentences)} sentence entries to {output_path}")

    return benchmark_sentences


if __name__ == "__main__":
    benchmark_sentences = get_benchmark_sentences(save_jsonl=True)
    print(f"Found {len(benchmark_sentences)} sentence entries")

    # Verify the JSONL file is loadable
    with open("data/benchmark_v2/sentence_bench.jsonl", "r") as f:
        loaded_sentences = [json.loads(line) for line in f]
    print(f"Loaded JSONL with {len(loaded_sentences)} rows")
    print(f"First entry: {loaded_sentences[0]}")
