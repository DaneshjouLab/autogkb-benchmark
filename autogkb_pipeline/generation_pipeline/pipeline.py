"""
Pharmacogenomics Knowledge Extraction Pipeline

Delegates to experiment factory classes for each stage:
- Variant extraction via VariantExtractor
- Term normalization via TermNormalizer
- Sentence generation via SentenceGenerator
- Citation finding via CitationFinder
- Summary generation via SummaryGenerator

Output Structure:
  outputs/<run_name>/
    config.yaml              — copy of resolved config
    metadata.yaml            — timestamp, pmcids, stages, git sha
    variants.json            — {pmcid: [variant_list]} (raw extracted)
    normalized_variants.json — {pmcid: [variant_list]} (after normalization)
    normalized_variants/     — {pmcid}.json per article (mappings)
    sentences/               — {pmcid}.json per article
    citations/               — {pmcid}.json per article
    summaries/               — {pmcid}.json per article
    outputs/                 — {pmcid}.json — full combined result per article
    eval_results/            — populated by eval pipeline

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

import time

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Paths
PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autogkb_pipeline.modules.variant_finding.variant_extractor import VariantExtractor
from autogkb_pipeline.modules.term_normalization.term_normalizer import TermNormalizer
from autogkb_pipeline.modules.term_normalization.models import NormalizationResult
from autogkb_pipeline.modules.sentence_generation.sentence_generator import SentenceGenerator
from autogkb_pipeline.modules.sentence_generation.models import GeneratedSentence
from autogkb_pipeline.modules.citations.citation_finder import CitationFinder
from autogkb_pipeline.modules.citations.models import Citation
from autogkb_pipeline.modules.summary.summary_generator import SummaryGenerator
from autogkb_pipeline.modules.summary.models import ArticleSummary

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


def _build_term_normalizer(config: dict) -> TermNormalizer:
    cfg = config.get("term_normalization", {})
    method = cfg.get("method", "pharmgkb_fuzzy")
    kwargs = {}
    if "threshold" in cfg:
        kwargs["threshold"] = cfg["threshold"]
    if "min_score" in cfg:
        kwargs["min_score"] = cfg["min_score"]
    if "top_k" in cfg:
        kwargs["top_k"] = cfg["top_k"]
    return TermNormalizer(method=method, **kwargs)


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
    normalizer: TermNormalizer | None,
    generator: SentenceGenerator | None,
    finder: CitationFinder | None,
    summarizer: SummaryGenerator | None,
    preloaded_variants: dict[str, list[str]] | None = None,
) -> dict:
    """Process a single PMCID through the pipeline stages.

    Args:
        preloaded_variants: Optional {pmcid: [variant_list]} loaded from a
            previous run's variants.json via --variants-file.

    Returns a dict with keys: pmcid, variants, normalized_variants, sentences,
    citations, summary.
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

    # Load variants from file if extraction didn't run
    if not variants and preloaded_variants and pmcid in preloaded_variants:
        variants = preloaded_variants[pmcid]
        result["variants"] = variants
        logger.info(f"  Loaded {len(variants)} variant(s) from variants file")

    # Stage 1.5: Term Normalization
    if "term_normalization" in stages and normalizer and variants:
        norm_result: NormalizationResult = normalizer.normalize(pmcid, variants)
        result["normalized_variants"] = {
            m.original: {"normalized": m.normalized, "score": m.score}
            for m in norm_result.mappings
        }
        # Downstream stages use normalized variants
        variants = norm_result.normalized_variants
        logger.info(
            f"  Normalized variants: {len(norm_result.normalized_variants)} "
            f"({sum(m.changed for m in norm_result.mappings)} changed)"
        )

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


def _save_result(
    result: dict,
    run_dir: Path,
    all_variants: dict,
    all_normalized_variants: dict,
) -> None:
    """Save per-PMCID results into the run directory structure."""
    pmcid = result["pmcid"]

    # Save normalized variants per-article (mappings)
    if "normalized_variants" in result:
        d = run_dir / "normalized_variants"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w", encoding="utf-8") as f:
            json.dump(result["normalized_variants"], f, indent=2, ensure_ascii=False)

    # Save per-stage files
    if "sentences" in result:
        d = run_dir / "sentences"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w", encoding="utf-8") as f:
            json.dump(result["sentences"], f, indent=2, ensure_ascii=False)

    if "citations" in result:
        d = run_dir / "citations"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w", encoding="utf-8") as f:
            json.dump(result["citations"], f, indent=2, ensure_ascii=False)

    if "summary" in result:
        d = run_dir / "summaries"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{pmcid}.json", "w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2, ensure_ascii=False)

    # Save full combined output
    d = run_dir / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{pmcid}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Accumulate raw variants
    if "variants" in result:
        all_variants[pmcid] = result["variants"]

    # Accumulate normalized variants separately
    if "normalized_variants" in result:
        all_normalized_variants[pmcid] = result["normalized_variants"]


def run_pipeline(
    pmcids: list[str],
    config: dict,
    stages: set[str],
    run_dir: Path,
    variants_file: Path | None = None,
) -> Path:
    """Run the full pipeline on multiple PMCIDs.

    Args:
        variants_file: Optional path to a variants.json from a previous run.
            Used to feed pre-extracted variants into term_normalization or
            downstream stages without re-running extraction.

    Returns the run directory path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    # Load pre-extracted variants if provided
    preloaded_variants: dict[str, list[str]] | None = None
    if variants_file:
        with open(variants_file) as f:
            data = json.load(f)
        preloaded_variants = data["variants"]
        logger.info(
            f"Loaded variants for {len(preloaded_variants)} PMCID(s) "
            f"from {variants_file}"
        )

    # Build factory instances
    extractor = _build_extractor(config) if "variants" in stages else None
    normalizer = (
        _build_term_normalizer(config)
        if "term_normalization" in stages
        and config.get("term_normalization", {}).get("enabled", True)
        else None
    )
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
    if variants_file:
        metadata["variants_file"] = str(variants_file)
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # Process each PMCID
    all_variants: dict[str, list[str]] = {}
    all_normalized_variants: dict[str, list[str]] = {}
    for i, pmcid in enumerate(pmcids, 1):
        logger.info(f"\n[{i}/{len(pmcids)}] Processing {pmcid}")
        try:
            result = process_pmcid(
                pmcid,
                stages,
                extractor,
                normalizer,
                generator,
                finder,
                summarizer,
                preloaded_variants,
            )
            _save_result(result, run_dir, all_variants, all_normalized_variants)
        except Exception as e:
            logger.error(f"Failed to process {pmcid}: {e}")

    # Save aggregated raw variants file
    if all_variants:
        variants_data = {
            "extractor": config["variant_extraction"]["method"],
            "run_name": config.get("config", {}).get("name", "unknown"),
            "timestamp": metadata["timestamp"],
            "variants": all_variants,
        }
        with open(run_dir / "variants.json", "w", encoding="utf-8") as f:
            json.dump(variants_data, f, indent=2, ensure_ascii=False)

    # Save aggregated normalized variants file
    if all_normalized_variants:
        norm_cfg = config.get("term_normalization", {})
        norm_data = {
            "normalizer": norm_cfg.get("method", "pharmgkb_fuzzy"),
            "run_name": config.get("config", {}).get("name", "unknown"),
            "timestamp": metadata["timestamp"],
            "normalized_variants": all_normalized_variants,
        }
        with open(run_dir / "normalized_variants.json", "w", encoding="utf-8") as f:
            json.dump(norm_data, f, indent=2, ensure_ascii=False)

    # Create eval_results directory
    (run_dir / "eval_results").mkdir(exist_ok=True)

    # Update metadata with total elapsed time
    elapsed_seconds = time.monotonic() - start_time
    metadata["elapsed_seconds"] = round(elapsed_seconds, 2)
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.success(
        f"Pipeline complete in {elapsed_seconds:.1f}s! Results saved to: {run_dir}"
    )
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
        default="variants,term_normalization,sentences,citations,summary",
        help="Comma-separated list of stages to run: variants, term_normalization, sentences, citations, summary (default: all)",
    )
    parser.add_argument(
        "--variants-file",
        type=Path,
        default=None,
        help="Path to a variants.json from a previous run to use as input for term_normalization or downstream stages",
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
    valid_stages = {
        "variants",
        "term_normalization",
        "sentences",
        "citations",
        "summary",
    }
    invalid_stages = stages - valid_stages
    if invalid_stages:
        logger.error(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")
        sys.exit(1)

    config_info = config.get("config", {})
    logger.info("Pipeline Configuration:")
    logger.info(f"  Config: {config_info.get('name', 'unknown')}")
    logger.info(f"  PMCIDs to process: {len(pmcids)}")
    logger.info(f"  Stages: {sorted(stages)}")
    logger.info(f"  Variant extraction: {config['variant_extraction']['method']}")
    if args.variants_file:
        logger.info(f"  Variants file: {args.variants_file}")
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
    run_dir = run_pipeline(pmcids, config, stages, run_dir, args.variants_file)

    # Optional evaluation
    if args.eval:
        from autogkb_pipeline.eval_pipeline.eval_pipeline import evaluate_run

        logger.info("\nRunning evaluation...")
        evaluate_run(run_dir, config, stages)


if __name__ == "__main__":
    main()
