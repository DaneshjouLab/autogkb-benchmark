"""
Pharmacogenomics Knowledge Extraction Pipeline

Delegates to experiment factory classes for each stage:
- Variant extraction via VariantExtractor
- Sentence generation via SentenceGenerator
- Citation finding via CitationFinder
- Summary generation via SummaryGenerator

Output Structure:
  outputs/<run_name>/
    config.yaml          — copy of resolved config
    metadata.yaml        — timestamp, pmcids, stages, git sha
    variants.json        — {pmcid: [variant_list]}
    sentences/           — {pmcid}.json per article
    citations/           — {pmcid}.json per article
    summaries/           — {pmcid}.json per article
    outputs/             — {pmcid}.json — full combined result per article
    eval_results/        — populated by eval pipeline

Example Commands:

1. Run pipeline for a single PMCID:
   python -m src.generation_pipeline.pipeline --num-pmcids 1

2. Run pipeline for all PMCIDs in benchmark:
   python -m src.generation_pipeline.pipeline

3. Run with custom config file:
   python -m src.generation_pipeline.pipeline --config configs/my_config.yaml

4. Run specific stages only:
   python -m src.generation_pipeline.pipeline --stages variants,sentences

5. Run specific PMCIDs:
   python -m src.generation_pipeline.pipeline --pmcids PMC123456 PMC789012

6. Run and evaluate:
   python -m src.generation_pipeline.pipeline --num-pmcids 1 --eval
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Paths
PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.variant_finding.variant_extractor import VariantExtractor
from src.modules.sentence_generation.sentence_generator import SentenceGenerator
from src.modules.sentence_generation.models import GeneratedSentence
from src.modules.citations.citation_finder import CitationFinder
from src.modules.citations.models import Citation
from src.modules.summary.summary_generator import SummaryGenerator
from src.modules.summary.models import ArticleSummary

CONFIGS_DIR = PIPELINE_DIR / "configs"
CONFIG_FILE = CONFIGS_DIR / "base_config.yaml"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
VARIANT_BENCH_PATH = ROOT / "data" / "benchmark_v2" / "variant_bench.jsonl"


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """Load pipeline configuration from YAML file."""
    logger.debug(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config.get('config', {}).get('name', 'unknown')}")
    return config


def get_pmcids_from_benchmark(num_pmcids: int | None = None) -> list[str]:
    """Get list of PMCIDs from the variant benchmark file."""
    logger.debug(f"Loading PMCIDs from {VARIANT_BENCH_PATH}")
    pmcids = []
    with open(VARIANT_BENCH_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pmcids.append(rec["pmcid"])
            if num_pmcids and len(pmcids) >= num_pmcids:
                break
    logger.info(f"Loaded {len(pmcids)} PMCID(s)")
    return pmcids


def _git_sha() -> str:
    """Get the current git SHA, or 'unknown' if not in a git repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# =============================================================================
# FACTORY INITIALIZATION
# =============================================================================


def _build_extractor(config: dict) -> VariantExtractor:
    cfg = config["variant_extraction"]
    return VariantExtractor(method=cfg["method"])


def _build_sentence_generator(config: dict) -> SentenceGenerator:
    cfg = config["sentence_generation"]
    kwargs = {}
    if "model" in cfg:
        kwargs["model"] = cfg["model"]
    if "prompt_version" in cfg:
        kwargs["prompt_version"] = cfg["prompt_version"]
    return SentenceGenerator(method=cfg["method"], **kwargs)


def _build_citation_finder(config: dict) -> CitationFinder:
    cfg = config["citation_finding"]
    kwargs = {}
    if "model" in cfg:
        kwargs["model"] = cfg["model"]
    if "prompt_version" in cfg:
        kwargs["prompt_version"] = cfg["prompt_version"]
    return CitationFinder(method=cfg["method"], **kwargs)


def _build_summary_generator(config: dict) -> SummaryGenerator:
    cfg = config["summary_generation"]
    kwargs = {}
    if "model" in cfg:
        kwargs["model"] = cfg["model"]
    if "prompt_version" in cfg:
        kwargs["prompt_version"] = cfg["prompt_version"]
    return SummaryGenerator(method=cfg["method"], **kwargs)


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def process_pmcid(
    pmcid: str,
    stages: set[str],
    extractor: VariantExtractor | None,
    generator: SentenceGenerator | None,
    finder: CitationFinder | None,
    summarizer: SummaryGenerator | None,
) -> dict:
    """Process a single PMCID through the pipeline stages.

    Returns a dict with keys: pmcid, variants, sentences, citations, summary.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing PMCID: {pmcid}")
    logger.info(f"{'=' * 60}")

    result: dict = {"pmcid": pmcid}

    # Stage 1: Variant Extraction
    variants: list[str] = []
    if "variants" in stages and extractor:
        variants = extractor.get_variants(pmcid)
        result["variants"] = variants
        logger.info(f"  Extracted {len(variants)} variant(s)")

    # Stage 2: Sentence Generation
    sentences: dict[str, list[GeneratedSentence]] = {}
    if "sentences" in stages and generator and variants:
        sentences = generator.generate(pmcid, variants)
        result["sentences"] = {
            v: [s.model_dump() for s in sents] for v, sents in sentences.items()
        }
        total = sum(len(sents) for sents in sentences.values())
        logger.info(f"  Generated {total} sentence(s) for {len(sentences)} variant(s)")

    # Stage 3: Citation Finding
    citations: list[Citation] = []
    if "citations" in stages and finder and sentences:
        associations = [
            {
                "variant": v,
                "sentence": s.sentence,
                "explanation": s.explanation,
            }
            for v, sents in sentences.items()
            for s in sents
        ]
        if associations:
            citations = finder.find_citations(pmcid, associations)
            result["citations"] = [c.model_dump() for c in citations]
            logger.info(f"  Found citations for {len(citations)} association(s)")

    # Stage 4: Summary Generation
    if "summary" in stages and summarizer:
        variants_data = [
            {
                "variant": v,
                "sentences": [s.sentence for s in sents],
            }
            for v, sents in sentences.items()
        ]
        citations_data = (
            {pmcid: [c.model_dump() for c in citations]} if citations else None
        )
        summary: ArticleSummary = summarizer.generate(
            pmcid, variants_data, citations_data
        )
        result["summary"] = summary.model_dump()
        logger.info(f"  Generated summary ({len(summary.summary)} chars)")

    return result


def _save_result(result: dict, run_dir: Path, all_variants: dict) -> None:
    """Save per-PMCID results into the run directory structure."""
    pmcid = result["pmcid"]

    # Save per-stage files
    if "sentences" in result:
        d = run_dir / "sentences"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w") as f:
            json.dump(result["sentences"], f, indent=2)

    if "citations" in result:
        d = run_dir / "citations"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w") as f:
            json.dump(result["citations"], f, indent=2)

    if "summary" in result:
        d = run_dir / "summaries"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w") as f:
            json.dump(result["summary"], f, indent=2)

    # Save full combined output
    d = run_dir / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{pmcid}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Accumulate variants for aggregated file
    if "variants" in result:
        all_variants[pmcid] = result["variants"]


def run_pipeline(
    pmcids: list[str],
    config: dict,
    stages: set[str],
    run_dir: Path,
) -> Path:
    """Run the full pipeline on multiple PMCIDs.

    Returns the run directory path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build factory instances
    extractor = _build_extractor(config) if "variants" in stages else None
    generator = _build_sentence_generator(config) if "sentences" in stages else None
    finder = _build_citation_finder(config) if "citations" in stages else None
    summarizer = _build_summary_generator(config) if "summary" in stages else None

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_name": config.get("config", {}).get("name", "unknown"),
        "pmcids": pmcids,
        "stages": sorted(stages),
        "git_sha": _git_sha(),
    }
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # Process each PMCID
    all_variants: dict[str, list[str]] = {}
    for i, pmcid in enumerate(pmcids, 1):
        logger.info(f"\n[{i}/{len(pmcids)}] Processing {pmcid}")
        try:
            result = process_pmcid(
                pmcid, stages, extractor, generator, finder, summarizer
            )
            _save_result(result, run_dir, all_variants)
        except Exception as e:
            logger.error(f"Failed to process {pmcid}: {e}")

    # Save aggregated variants file
    if all_variants:
        variants_data = {
            "extractor": config["variant_extraction"]["method"],
            "run_name": config.get("config", {}).get("name", "unknown"),
            "timestamp": metadata["timestamp"],
            "variants": all_variants,
        }
        with open(run_dir / "variants.json", "w") as f:
            json.dump(variants_data, f, indent=2)

    # Create eval_results directory
    (run_dir / "eval_results").mkdir(exist_ok=True)

    logger.success(f"Pipeline complete! Results saved to: {run_dir}")
    return run_dir


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run the pharmacogenomics knowledge extraction pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help=f"Path to config YAML file (default: {CONFIG_FILE})",
    )
    parser.add_argument(
        "--num-pmcids",
        type=int,
        default=None,
        help="Number of PMCIDs to process (default: all)",
    )
    parser.add_argument(
        "--pmcids",
        nargs="+",
        default=None,
        help="Specific PMCIDs to process (overrides --num-pmcids)",
    )
    parser.add_argument(
        "--stages",
        default="variants,sentences,citations,summary",
        help="Comma-separated list of stages to run (default: all)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after pipeline completes",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get PMCIDs
    if args.pmcids:
        pmcids = args.pmcids
    else:
        pmcids = get_pmcids_from_benchmark(args.num_pmcids)

    # Parse stages
    stages = set(s.strip() for s in args.stages.split(","))
    valid_stages = {"variants", "sentences", "citations", "summary"}
    invalid_stages = stages - valid_stages
    if invalid_stages:
        logger.error(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")
        sys.exit(1)

    config_info = config.get("config", {})
    logger.info("Pipeline Configuration:")
    logger.info(f"  Config: {config_info.get('name', 'unknown')}")
    logger.info(f"  PMCIDs to process: {len(pmcids)}")
    logger.info(f"  Stages: {sorted(stages)}")
    logger.info(
        f"  Variant extraction: {config['variant_extraction']['method']}"
    )
    if "sentences" in stages:
        logger.info(f"  Sentence model: {config['sentence_generation']['model']}")
    if "citations" in stages:
        logger.info(f"  Citation model: {config['citation_finding']['model']}")
    if "summary" in stages:
        logger.info(f"  Summary model: {config['summary_generation']['model']}")

    # Create output directory
    config_name = config_info.get("name", "pipeline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"{config_name}_{timestamp}"
    logger.info(f"  Output directory: {run_dir}")

    # Copy config into run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    # Run pipeline
    run_dir = run_pipeline(pmcids, config, stages, run_dir)

    # Optional evaluation
    if args.eval:
        from src.eval_pipeline.eval_pipeline import evaluate_run

        logger.info("\nRunning evaluation...")
        evaluate_run(run_dir, config, stages)


if __name__ == "__main__":
    main()
