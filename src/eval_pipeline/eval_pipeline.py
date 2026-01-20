"""
Evaluation Pipeline for Pharmacogenomics Knowledge Extraction

This pipeline evaluates the outputs from the generation pipeline against ground truth data.
It produces comprehensive reports for:
- Variant extraction accuracy (exact matching against ground truth)
- Sentence generation quality (LLM-judged similarity to ground truth)

Citation evaluation is currently disabled.

Example Commands:

1. Evaluate a pipeline output file:
   python eval_pipeline.py --input ../pipeline/outputs/base_config.json

2. Evaluate with custom config:
   python eval_pipeline.py --input ../pipeline/outputs/base_config.json --config configs/custom.yaml

3. Evaluate specific stages only:
   python eval_pipeline.py --input ../pipeline/outputs/base_config.json --stages variants,sentences

4. Use a different judge model:
   python eval_pipeline.py --input ../pipeline/outputs/base_config.json --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment (API keys, etc.)
load_dotenv()

# Paths
EVAL_PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = EVAL_PIPELINE_DIR.parents[1]

# Add repository root to Python path to enable imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import benchmark evaluation functions
from src.benchmark_v2.variant_bench import (
    score_variants_by_pmcid,
    load_variant_bench_data,
    VariantBenchResult,
)
from src.benchmark_v2.sentence_bench import (
    load_sentence_bench_data,
    score_variant_sentences,
    VariantSentenceScore,
)

# Import LLM utilities for summary generation
from litellm import completion

# Default paths within eval_pipeline folder
CONFIGS_DIR = EVAL_PIPELINE_DIR / "configs"
CONFIG_FILE = CONFIGS_DIR / "default_config.yaml"
OUTPUTS_DIR = EVAL_PIPELINE_DIR / "outputs"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VariantEvaluationResult:
    """Result of variant evaluation for a single PMCID."""

    pmcid: str
    match_rate: float  # Recall: matches / ground_truth
    precision: float  # matches / proposed
    recall: float  # matches / ground_truth (same as match_rate)
    f1_score: float  # 2 * (precision * recall) / (precision + recall)
    matches: list[str]
    misses: list[str]
    extras: list[str]
    num_proposed: int
    num_ground_truth: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceEvaluationResult:
    """Result of sentence evaluation for a single PMCID."""

    pmcid: str
    avg_score: float
    num_variants_scored: int
    num_variants_not_in_ground_truth: int
    per_variant: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PMCIDEvaluationResult:
    """Complete evaluation result for a single PMCID."""

    pmcid: str
    variant_evaluation: VariantEvaluationResult | None = None
    sentence_evaluation: SentenceEvaluationResult | None = None

    def to_dict(self) -> dict:
        result = {"pmcid": self.pmcid}
        if self.variant_evaluation:
            result["variant_evaluation"] = self.variant_evaluation.to_dict()
        if self.sentence_evaluation:
            result["sentence_evaluation"] = self.sentence_evaluation.to_dict()
        return result


@dataclass
class VariantMetrics:
    """Aggregate variant evaluation metrics."""

    precision: float
    recall: float
    f1_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result including metadata and all PMCID results."""

    metadata: dict
    overall_variant_metrics: VariantMetrics | None
    overall_sentence_score: float | None
    num_pmcids: int
    per_pmcid: list[PMCIDEvaluationResult]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "overall_variant_metrics": self.overall_variant_metrics.to_dict() if self.overall_variant_metrics else None,
            "overall_sentence_score": self.overall_sentence_score,
            "num_pmcids": self.num_pmcids,
            "per_pmcid": [r.to_dict() for r in self.per_pmcid],
            "summary": self.summary,
        }


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """Load evaluation configuration from YAML file."""
    logger.debug(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded eval config: {config.get('config', {}).get('name', 'unknown')}")
    return config


def load_pipeline_output(input_path: Path) -> dict:
    """Load pipeline output JSON file."""
    logger.debug(f"Loading pipeline output from {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    logger.info(f"Loaded pipeline output with {len(data.get('results', []))} PMCID(s)")
    return data


# =============================================================================
# VARIANT EVALUATION
# =============================================================================


def calculate_f1_metrics(
    num_matches: int,
    num_proposed: int,
    num_ground_truth: int,
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score.

    Args:
        num_matches: Number of correct matches (true positives)
        num_proposed: Total number of proposed items
        num_ground_truth: Total number of ground truth items

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Precision: what fraction of proposed are correct
    precision = num_matches / num_proposed if num_proposed > 0 else 0.0

    # Recall: what fraction of ground truth were found
    recall = num_matches / num_ground_truth if num_ground_truth > 0 else 0.0

    # F1: harmonic mean of precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score


def evaluate_variants_for_pmcid(
    pmcid: str,
    proposed_variants: list[str],
) -> VariantEvaluationResult:
    """Evaluate proposed variants against ground truth for a single PMCID.

    Args:
        pmcid: PMCID to evaluate
        proposed_variants: List of variant identifiers proposed by the pipeline

    Returns:
        VariantEvaluationResult with match statistics including F1 metrics
    """
    logger.info(f"Evaluating variants for {pmcid}")

    try:
        result: VariantBenchResult = score_variants_by_pmcid(proposed_variants, pmcid)

        num_matches = len(result.matches)
        num_proposed = len(proposed_variants)
        num_ground_truth = len(result.matches) + len(result.misses)

        precision, recall, f1_score = calculate_f1_metrics(
            num_matches, num_proposed, num_ground_truth
        )

        eval_result = VariantEvaluationResult(
            pmcid=result.pmcid,
            match_rate=result.match_rate,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1_score=round(f1_score, 3),
            matches=result.matches,
            misses=result.misses,
            extras=result.extras,
            num_proposed=num_proposed,
            num_ground_truth=num_ground_truth,
        )

        logger.info(
            f"  Variants - P: {precision:.1%}, R: {recall:.1%}, F1: {f1_score:.3f} "
            f"({num_matches}/{num_ground_truth} found, {len(result.extras)} extra)"
        )
        if result.misses:
            logger.warning(f"  Missed variants: {result.misses}")
        if result.extras:
            logger.debug(f"  Extra variants: {result.extras}")

        return eval_result

    except KeyError as e:
        logger.warning(f"  PMCID {pmcid} not found in variant benchmark: {e}")
        return VariantEvaluationResult(
            pmcid=pmcid,
            match_rate=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            matches=[],
            misses=[],
            extras=proposed_variants,
            num_proposed=len(proposed_variants),
            num_ground_truth=0,
        )


# =============================================================================
# SENTENCE EVALUATION
# =============================================================================


def evaluate_sentences_for_pmcid(
    pmcid: str,
    associations: list[dict],
    judge_model: str,
) -> SentenceEvaluationResult:
    """Evaluate generated sentences against ground truth for a single PMCID.

    Args:
        pmcid: PMCID to evaluate
        associations: List of association dicts from pipeline output
        judge_model: LLM model to use for judging

    Returns:
        SentenceEvaluationResult with detailed scores
    """
    logger.info(f"Evaluating sentences for {pmcid}")

    # Load ground truth data
    ground_truth_data = load_sentence_bench_data()

    if pmcid not in ground_truth_data:
        logger.warning(f"  PMCID {pmcid} not found in sentence benchmark")
        return SentenceEvaluationResult(
            pmcid=pmcid,
            avg_score=0.0,
            num_variants_scored=0,
            num_variants_not_in_ground_truth=len(associations),
            per_variant=[],
        )

    ground_truth_variants = ground_truth_data[pmcid]

    # Build generated sentences map: variant -> [sentences]
    generated_map: dict[str, list[str]] = {}
    for assoc in associations:
        variant_id = assoc.get("variant_id", "")
        sentence = assoc.get("sentence", "")
        if variant_id and sentence:
            if variant_id not in generated_map:
                generated_map[variant_id] = []
            generated_map[variant_id].append(sentence)

    # Score each variant
    per_variant_scores = []
    total_score = 0.0
    num_variants_scored = 0

    # Score variants that are in ground truth
    for variant, gt_sentences in ground_truth_variants.items():
        gen_sentences = generated_map.get(variant, [])

        if gen_sentences:
            variant_score: VariantSentenceScore = score_variant_sentences(
                variant=variant,
                ground_truth=gt_sentences,
                generated=gen_sentences,
                model=judge_model,
            )

            per_variant_scores.append({
                "variant": variant,
                "ground_truth": variant_score.ground_truth,
                "generated": variant_score.generated,
                "score": variant_score.score,
                "critique": variant_score.critique,
            })

            if variant_score.score is not None:
                total_score += variant_score.score
                num_variants_scored += 1
                logger.debug(f"  {variant}: {variant_score.score:.2f}")
        else:
            # No generated sentences for this ground truth variant
            per_variant_scores.append({
                "variant": variant,
                "ground_truth": gt_sentences,
                "generated": [],
                "score": 0.0,
                "critique": "No generated sentences for this variant",
            })
            total_score += 0.0
            num_variants_scored += 1
            logger.warning(f"  {variant}: No generated sentences")

    # Track extra variants not in ground truth
    generated_extras = [v for v in generated_map.keys() if v not in ground_truth_variants]
    for variant in generated_extras:
        per_variant_scores.append({
            "variant": variant,
            "ground_truth": None,
            "generated": generated_map[variant],
            "score": None,
            "critique": "Variant not found in ground truth - not scored",
        })

    # Calculate average score
    avg_score = total_score / num_variants_scored if num_variants_scored > 0 else 0.0

    logger.info(
        f"  Sentence avg score: {avg_score:.3f} "
        f"({num_variants_scored} variants scored, {len(generated_extras)} extras)"
    )

    return SentenceEvaluationResult(
        pmcid=pmcid,
        avg_score=round(avg_score, 3),
        num_variants_scored=num_variants_scored,
        num_variants_not_in_ground_truth=len(generated_extras),
        per_variant=per_variant_scores,
    )


# =============================================================================
# SUMMARY GENERATION
# =============================================================================


def generate_evaluation_summary(
    eval_result: "EvaluationResult",
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate an LLM summary of the evaluation findings.

    Args:
        eval_result: The complete evaluation result
        model: LLM model to use for summary generation

    Returns:
        Summary text describing the evaluation findings
    """
    logger.info("Generating evaluation summary...")

    # Build context for the summary
    variant_metrics = eval_result.overall_variant_metrics
    sentence_score = eval_result.overall_sentence_score

    # Build per-PMCID details
    pmcid_details = []
    for pmcid_result in eval_result.per_pmcid:
        detail = f"PMCID: {pmcid_result.pmcid}"

        if pmcid_result.variant_evaluation:
            ve = pmcid_result.variant_evaluation
            detail += f"\n  Variants: P={ve.precision:.1%}, R={ve.recall:.1%}, F1={ve.f1_score:.3f}"
            detail += f"\n    Matches: {ve.matches}"
            if ve.misses:
                detail += f"\n    Misses: {ve.misses}"
            if ve.extras:
                detail += f"\n    Extras (not in ground truth): {ve.extras}"

        if pmcid_result.sentence_evaluation:
            se = pmcid_result.sentence_evaluation
            detail += f"\n  Sentences: avg_score={se.avg_score:.3f} ({se.num_variants_scored} scored)"
            for var_score in se.per_variant:
                if var_score["score"] is not None:
                    detail += f"\n    {var_score['variant']}: {var_score['score']:.2f} - {var_score['critique'][:100]}..."

        pmcid_details.append(detail)

    prompt = f"""You are summarizing the evaluation results of a pharmacogenomics knowledge extraction pipeline.
The pipeline extracts genetic variants and association sentences from scientific articles.

## Overall Metrics
- Number of PMCIDs evaluated: {eval_result.num_pmcids}
- Variant Extraction: {"Precision=" + f"{variant_metrics.precision:.1%}, Recall={variant_metrics.recall:.1%}, F1={variant_metrics.f1_score:.3f}" if variant_metrics else "Not evaluated"}
- Sentence Generation: {"Average score=" + f"{sentence_score:.3f}" if sentence_score else "Not evaluated"}

## Per-PMCID Details
{chr(10).join(pmcid_details)}

Write a concise summary (2-4 paragraphs) of these evaluation results. Include:
1. Overall performance assessment for both variant extraction and sentence generation
2. Key strengths and weaknesses observed
3. Specific examples of what worked well or poorly (mention specific variants or issues)
4. Recommendations for improvement if applicable

Be specific and quantitative where possible. Focus on actionable insights."""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        summary = response.choices[0].message.content
        logger.info(f"Generated evaluation summary ({len(summary)} chars)")
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"[Error generating summary: {e}]"


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def evaluate_pmcid(
    pmcid_result: dict,
    config: dict,
    stages: set[str],
) -> PMCIDEvaluationResult:
    """Evaluate a single PMCID result from the pipeline output.

    Args:
        pmcid_result: Dict with pipeline output for one PMCID
        config: Evaluation configuration
        stages: Set of stages to evaluate

    Returns:
        PMCIDEvaluationResult with all evaluations
    """
    pmcid = pmcid_result.get("pmcid", "")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating PMCID: {pmcid}")
    logger.info(f"{'=' * 60}")

    result = PMCIDEvaluationResult(pmcid=pmcid)

    # Variant Evaluation
    if "variants" in stages and config.get("variant_evaluation", {}).get("enabled", True):
        proposed_variants = pmcid_result.get("variants", [])
        result.variant_evaluation = evaluate_variants_for_pmcid(pmcid, proposed_variants)

    # Sentence Evaluation
    if "sentences" in stages and config.get("sentence_evaluation", {}).get("enabled", True):
        associations = pmcid_result.get("associations", [])
        judge_model = config.get("sentence_evaluation", {}).get(
            "judge_model", "claude-sonnet-4-20250514"
        )
        result.sentence_evaluation = evaluate_sentences_for_pmcid(
            pmcid, associations, judge_model
        )

    return result


def evaluate_pipeline_output(
    input_path: Path,
    config: dict,
    stages: set[str],
) -> EvaluationResult:
    """Evaluate a complete pipeline output file.

    Args:
        input_path: Path to pipeline output JSON file
        config: Evaluation configuration
        stages: Set of stages to evaluate

    Returns:
        EvaluationResult with all evaluations
    """
    # Load pipeline output
    pipeline_data = load_pipeline_output(input_path)
    pipeline_metadata = pipeline_data.get("metadata", {})
    results = pipeline_data.get("results", [])

    timestamp = datetime.now().isoformat()

    # Build evaluation metadata
    config_info = config.get("config", {})
    metadata = {
        "timestamp": timestamp,
        "eval_config_name": config_info.get("name", "unknown"),
        "eval_config_description": config_info.get("description", ""),
        "stages_evaluated": list(stages),
        "source_pipeline_config": pipeline_metadata.get("config_name", "unknown"),
        "source_file": str(input_path),
        "judge_model": config.get("sentence_evaluation", {}).get(
            "judge_model", "claude-sonnet-4-20250514"
        ),
    }

    # Evaluate each PMCID
    per_pmcid_results: list[PMCIDEvaluationResult] = []
    for pmcid_result in results:
        try:
            eval_result = evaluate_pmcid(pmcid_result, config, stages)
            per_pmcid_results.append(eval_result)
        except Exception as e:
            logger.error(f"Failed to evaluate {pmcid_result.get('pmcid', 'unknown')}: {e}")
            per_pmcid_results.append(
                PMCIDEvaluationResult(pmcid=pmcid_result.get("pmcid", "unknown"))
            )

    # Calculate overall metrics
    overall_variant_metrics = None
    overall_sentence_score = None

    if "variants" in stages:
        # Aggregate counts across all PMCIDs for micro-averaged F1
        total_matches = 0
        total_proposed = 0
        total_ground_truth = 0

        for r in per_pmcid_results:
            if r.variant_evaluation is not None:
                total_matches += len(r.variant_evaluation.matches)
                total_proposed += r.variant_evaluation.num_proposed
                total_ground_truth += r.variant_evaluation.num_ground_truth

        if total_proposed > 0 or total_ground_truth > 0:
            precision, recall, f1_score = calculate_f1_metrics(
                total_matches, total_proposed, total_ground_truth
            )
            overall_variant_metrics = VariantMetrics(
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1_score=round(f1_score, 3),
            )

    if "sentences" in stages:
        sentence_scores = [
            r.sentence_evaluation.avg_score
            for r in per_pmcid_results
            if r.sentence_evaluation is not None
        ]
        if sentence_scores:
            overall_sentence_score = round(
                sum(sentence_scores) / len(sentence_scores), 3
            )

    # Create the result object first (without summary)
    eval_result = EvaluationResult(
        metadata=metadata,
        overall_variant_metrics=overall_variant_metrics,
        overall_sentence_score=overall_sentence_score,
        num_pmcids=len(per_pmcid_results),
        per_pmcid=per_pmcid_results,
        summary="",
    )

    # Generate LLM summary
    summary_model = config.get("sentence_evaluation", {}).get(
        "judge_model", "claude-sonnet-4-20250514"
    )
    eval_result.summary = generate_evaluation_summary(eval_result, summary_model)

    return eval_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline outputs against ground truth."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to pipeline output JSON file to evaluate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help=f"Path to eval config YAML file (default: {CONFIG_FILE})",
    )
    parser.add_argument(
        "--stages",
        default="variants,sentences",
        help="Comma-separated list of stages to evaluate (default: variants,sentences)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model for sentence evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (default: auto-generated)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override judge model if specified
    if args.judge_model:
        if "sentence_evaluation" not in config:
            config["sentence_evaluation"] = {}
        config["sentence_evaluation"]["judge_model"] = args.judge_model

    # Parse stages
    stages = set(s.strip() for s in args.stages.split(","))
    valid_stages = {"variants", "sentences", "citations", "summary"}
    invalid_stages = stages - valid_stages
    if invalid_stages:
        logger.error(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")
        sys.exit(1)

    # Log configuration
    config_info = config.get("config", {})
    logger.info(f"Evaluation Configuration:")
    logger.info(f"  Config: {config_info.get('name', 'unknown')}")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Stages: {sorted(stages)}")
    if "sentences" in stages:
        judge_model = config.get("sentence_evaluation", {}).get(
            "judge_model", "claude-sonnet-4-20250514"
        )
        logger.info(f"  Judge model: {judge_model}")

    # Run evaluation
    eval_result = evaluate_pipeline_output(args.input, config, stages)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        # Extract pipeline config name from input filename
        pipeline_name = args.input.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"eval_{pipeline_name}_{timestamp}.json"

    # Save results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.success(f"\nEvaluation complete!")
    logger.success(f"Results saved to: {out_path}")

    # Print summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  PMCIDs evaluated: {eval_result.num_pmcids}")

    if eval_result.overall_variant_metrics is not None:
        vm = eval_result.overall_variant_metrics
        logger.info(
            f"  Variant extraction - P: {vm.precision:.1%}, R: {vm.recall:.1%}, F1: {vm.f1_score:.3f}"
        )

    if eval_result.overall_sentence_score is not None:
        logger.info(
            f"  Sentence generation - Avg score: {eval_result.overall_sentence_score:.3f}"
        )

    # Print per-PMCID breakdown
    logger.info(f"\nPer-PMCID Results:")
    for pmcid_result in eval_result.per_pmcid:
        pmcid = pmcid_result.pmcid
        parts = [f"  {pmcid}:"]

        if pmcid_result.variant_evaluation:
            ve = pmcid_result.variant_evaluation
            parts.append(f"P={ve.precision:.1%} R={ve.recall:.1%} F1={ve.f1_score:.3f}")

        if pmcid_result.sentence_evaluation:
            se = pmcid_result.sentence_evaluation
            parts.append(f"sentences={se.avg_score:.3f}")

        logger.info(" ".join(parts))

    # Print LLM summary
    if eval_result.summary:
        logger.info(f"\n{'=' * 60}")
        logger.info("LLM Evaluation Summary:")
        logger.info(f"{'=' * 60}")
        logger.info(eval_result.summary)


if __name__ == "__main__":
    main()
