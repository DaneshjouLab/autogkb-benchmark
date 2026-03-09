"""
Pharmacogenomics Knowledge Extraction Pipeline

Delegates to experiment factory classes for each stage:
- Variant extraction via VariantExtractor
- Term normalization via TermNormalizer
- Sentence generation via SentenceGenerator
- Citation finding via CitationFinder
- Summary generation via SummaryGenerator

Output: appends GenerationRecord JSON lines to data/generations.jsonl

Example Commands:

1. Run pipeline from PMID (auto-converts to PMCID):
   python -m generation --pmid 32948745

2. Run pipeline for specific PMCIDs:
   python -m generation --pmcids PMC10275785

3. Run specific stages only:
   python -m generation --pmcids PMC10275785 --stages variants

4. Run and evaluate:
   python -m generation --pmcids PMC10275785 --eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from shared.utils import ROOT, DATA_DIR, get_markdown_text
from shared.data_setup.pmcid_converter import PMIDConverter
from shared.data_setup.pmc_title_fetcher import get_title_from_pmcid
from shared.data_setup.download_article import download_article

from generation.models import GenerationRecord, GenerationMetadata, GenerationStatus
from generation.modules.variant_finding.variant_extractor import VariantExtractor
from generation.modules.term_normalization.term_normalizer import TermNormalizer
from generation.modules.term_normalization.models import NormalizationResult
from generation.modules.sentence_generation.sentence_generator import SentenceGenerator
from generation.modules.sentence_generation.models import GeneratedSentence
from generation.modules.citations.citation_finder import CitationFinder
from generation.modules.citations.models import Citation
from generation.modules.summary.summary_generator import SummaryGenerator
from generation.modules.summary.models import ArticleSummary

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PIPELINE_DIR / "configs"
CONFIG_FILE = CONFIGS_DIR / "base_config.yaml"
VARIANT_BENCH_PATH = DATA_DIR / "benchmark_v2" / "variant_bench.jsonl"
PMCID_MAPPING_PATH = DATA_DIR / "pmcid_mapping.json"
GENERATIONS_JSONL = DATA_DIR / "generations.jsonl"
GENERATIONS_DIR = DATA_DIR / "generations"


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
# PMID RESOLUTION
# =============================================================================


def _load_pmcid_mapping() -> dict[str, str | None]:
    """Load cached PMID->PMCID mappings from disk."""
    if PMCID_MAPPING_PATH.exists():
        with open(PMCID_MAPPING_PATH) as f:
            return json.load(f)
    return {}


def _save_pmcid_mapping(mapping: dict[str, str | None]) -> None:
    """Persist PMID->PMCID mappings to disk."""
    PMCID_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PMCID_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)


def resolve_pmids(pmids: list[str]) -> list[tuple[str, str]]:
    """Convert PMIDs to PMCIDs, using cache where possible.

    Returns list of (pmid, pmcid) tuples for successfully resolved PMIDs.
    """
    mapping = _load_pmcid_mapping()

    # Find PMIDs not yet cached (or cached as None = not-found)
    unknown = [p for p in pmids if p not in mapping]

    if unknown:
        logger.info(f"Converting {len(unknown)} new PMID(s) via NCBI API...")
        converter = PMIDConverter()
        new_mappings = converter.convert(unknown, show_progress=True)
        for pmid in unknown:
            mapping[pmid] = new_mappings.get(pmid)
        _save_pmcid_mapping(mapping)

    # Build results, skipping PMIDs with no PMCID
    results: list[tuple[str, str]] = []
    for pmid in pmids:
        pmcid = mapping.get(pmid)
        if not pmcid:
            logger.error(f"No PMCID found for PMID {pmid}, skipping")
            continue
        # Auto-download article markdown if missing
        md_path = DATA_DIR / "articles" / f"{pmcid}.md"
        if not md_path.exists():
            logger.info(f"Article not found locally, downloading {pmcid}...")
            try:
                download_article(pmcid)
            except Exception as e:
                logger.warning(f"Could not download {pmcid}: {e}")
        results.append((pmid, pmcid))

    return results


def _build_reverse_pmcid_map() -> dict[str, str]:
    """Build PMCID->PMID reverse lookup from the mapping file."""
    mapping = _load_pmcid_mapping()
    return {v: k for k, v in mapping.items() if v is not None}


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
) -> dict | None:
    """Process a single PMCID through the pipeline stages.

    Returns a dict with keys: pmcid, variants, normalized_variants, sentences,
    citations, summary.  Returns ``None`` if variant extraction ran but found
    nothing (not a PGx article).
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

    # Bail out early if variant extraction found nothing — likely not a PGx article
    if "variants" in stages and extractor and not variants:
        logger.warning(
            f"  No variants found for {pmcid} — skipping (likely not a PGx article)"
        )
        return None

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


def _save_generation_md(record: GenerationRecord) -> None:
    """Save the article text as a markdown file in data/generations/."""
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    pmid = record.pmid or "unknown"
    filename = f"PMID{pmid}_{record.id}.md"
    (GENERATIONS_DIR / filename).write_text(record.text_content, encoding="utf-8")


def _append_jsonl(record: GenerationRecord) -> None:
    """Append a GenerationRecord as one JSON line to data/generations.jsonl."""
    GENERATIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(GENERATIONS_JSONL, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    _save_generation_md(record)


def _update_jsonl(record_id: str, updates: dict) -> None:
    """Update an existing record in generations.jsonl by ID."""
    if not GENERATIONS_JSONL.exists():
        return
    lines = GENERATIONS_JSONL.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("id") == record_id:
            data.update(updates)
        new_lines.append(json.dumps(data, ensure_ascii=False))
    GENERATIONS_JSONL.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def run_pipeline(
    pmcid_pairs: list[tuple[str | None, str]],
    config: dict,
    stages: set[str],
    variants_file: Path | None = None,
) -> None:
    """Run the full pipeline on multiple PMCIDs and append JSONL output.

    Args:
        pmcid_pairs: List of (pmid_or_none, pmcid) tuples.
        config: Pipeline configuration dict.
        stages: Set of stage names to run.
        variants_file: Optional path to a variants.json from a previous run.
    """
    start_time = time.monotonic()
    git_sha = _git_sha()

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

    config_info = config.get("config", {})

    # Process each PMCID
    for i, (pmid, pmcid) in enumerate(pmcid_pairs, 1):
        logger.info(f"\n[{i}/{len(pmcid_pairs)}] Processing {pmcid}")

        # Auto-download article if missing
        md_path = DATA_DIR / "articles" / f"{pmcid}.md"
        if not md_path.exists():
            try:
                download_article(pmcid)
            except Exception as e:
                logger.warning(f"Could not download article for {pmcid}: {e}")

        # Get article title and text early for the in_progress record
        try:
            title = get_title_from_pmcid(pmcid, DATA_DIR)
        except Exception:
            title = None
        text_content = get_markdown_text(pmcid)

        # Write initial in_progress record
        record = GenerationRecord(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            text_content=text_content,
            annotations={},
            annotation_citations=[],
            status=GenerationStatus.in_progress,
            generation_metadata=GenerationMetadata(
                config_name=config_info.get("name", "unknown"),
                variant_extraction_method=config["variant_extraction"]["method"],
                sentence_generation_method=config.get("sentence_generation", {}).get(
                    "method"
                ),
                sentence_model=config.get("sentence_generation", {}).get("model"),
                citation_model=config.get("citation_finding", {}).get("model"),
                summary_model=config.get("summary_generation", {}).get("model"),
                elapsed_seconds=0.0,
                git_sha=git_sha,
                stages_run=sorted(stages),
            ),
        )
        record_id = record.id
        _append_jsonl(record)
        logger.info(f"  Wrote in_progress record {record_id}")

        pmcid_start = time.monotonic()
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

            if result is None:
                # Skipped (e.g. no variants found)
                _update_jsonl(
                    record_id,
                    {
                        "status": GenerationStatus.completed.value,
                        "generation_metadata": {
                            **record.generation_metadata.model_dump(),
                            "elapsed_seconds": round(time.monotonic() - pmcid_start, 2),
                        },
                    },
                )
                continue

            # Build annotations dict: {variant_name: [{sentence, explanation}]}
            annotations = {}
            if "sentences" in result:
                for variant, sent_list in result["sentences"].items():
                    annotations[variant] = [
                        {
                            "sentence": s["sentence"],
                            "explanation": s.get("explanation", ""),
                        }
                        for s in sent_list
                    ]

            # Build annotation_citations list
            annotation_citations = result.get("citations", [])

            elapsed = round(time.monotonic() - pmcid_start, 2)

            _update_jsonl(
                record_id,
                {
                    "annotations": annotations,
                    "annotation_citations": annotation_citations,
                    "annotation_data": result,
                    "status": GenerationStatus.completed.value,
                    "generation_metadata": {
                        **record.generation_metadata.model_dump(),
                        "elapsed_seconds": elapsed,
                    },
                },
            )
            logger.info(f"  Updated record {record_id} to completed")

        except Exception as e:
            elapsed = round(time.monotonic() - pmcid_start, 2)
            _update_jsonl(
                record_id,
                {
                    "status": GenerationStatus.error.value,
                    "error": str(e),
                    "generation_metadata": {
                        **record.generation_metadata.model_dump(),
                        "elapsed_seconds": elapsed,
                    },
                },
            )
            logger.error(f"Failed to process {pmcid}: {e}")

    total_elapsed = time.monotonic() - start_time
    logger.success(
        f"Pipeline complete in {total_elapsed:.1f}s! "
        f"Results appended to {GENERATIONS_JSONL}"
    )


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
        "--pmid",
        nargs="+",
        default=None,
        help="PMID(s) to process (auto-converts to PMCIDs)",
    )
    parser.add_argument(
        "--num-pmcids",
        type=int,
        default=None,
        help="Number of PMCIDs to process from benchmark (default: all)",
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
        help="Comma-separated list of stages to run (default: all)",
    )
    parser.add_argument(
        "--variants-file",
        type=Path,
        default=None,
        help="Path to a variants.json from a previous run",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Build (pmid, pmcid) pairs from the chosen input mode
    if args.pmid:
        # PMID-first: resolve to PMCIDs
        pairs = resolve_pmids(args.pmid)
    elif args.pmcids:
        # Direct PMCIDs — try reverse lookup for PMID
        reverse_map = _build_reverse_pmcid_map()
        pairs = [(reverse_map.get(p), p) for p in args.pmcids]
    else:
        # Benchmark PMCIDs
        pmcids = get_pmcids_from_benchmark(args.num_pmcids)
        reverse_map = _build_reverse_pmcid_map()
        pairs = [(reverse_map.get(p), p) for p in pmcids]

    if not pairs:
        logger.error("No PMCIDs to process.")
        sys.exit(1)

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
    logger.info(f"  Articles to process: {len(pairs)}")
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
    logger.info(f"  Output: {GENERATIONS_JSONL}")

    # Run pipeline
    run_pipeline(pairs, config, stages, args.variants_file)


if __name__ == "__main__":
    main()
