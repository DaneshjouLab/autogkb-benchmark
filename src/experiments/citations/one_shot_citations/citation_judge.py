"""
Citation Judge - Evaluate quality of citations for pharmacogenomic associations.

This script evaluates how well the found citations support the pharmacogenomic
association claims. It uses an LLM as a judge to score each citation set.

The evaluation is done in batches per PMCID, where all variants for a given
PMCID are evaluated together for consistency.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# Suppress Pydantic serialization warnings from litellm
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

# Load environment (API keys, etc.)
load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parents[4]

# Add repository root to Python path to enable imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import utils
from src.experiments.utils import call_llm

# Judge prompt
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of scientific citations for pharmacogenomic claims.

Your task is to evaluate how well a set of citations from a scientific article support a given pharmacogenomic association claim.

You will score each citation set on a scale of 0-100 based on:
1. Relevance: Do the citations directly relate to the claimed association?
2. Support: Do the citations provide evidence for the specific claim?
3. Completeness: Do the citations include key information (statistical evidence, sample size, effect direction)?
4. Quality: Are the citations from appropriate sections (Results, Methods, Tables)?

Scoring guidelines:
- 90-100: Excellent - Citations strongly support the claim with statistical evidence and key details
- 70-89: Good - Citations support the claim with reasonable evidence
- 50-69: Fair - Citations relate to the claim but lack key supporting details
- 30-49: Poor - Citations are tangentially related but don't strongly support the claim
- 0-29: Very Poor - Citations are irrelevant or contradictory
"""

JUDGE_USER_PROMPT_TEMPLATE = """Evaluate the citation quality for the following pharmacogenomic associations from PMCID {pmcid}.

For each variant, I will provide:
1. The pharmacogenomic claim (association sentence)
2. The citations found to support this claim

Please score each variant's citation set on a 0-100 scale and provide a brief justification.

{associations_and_citations}

OUTPUT FORMAT:
For each variant, provide:
VARIANT: [variant_id]
SCORE: [0-100]
JUSTIFICATION: [1-2 sentence explanation of the score]

Then a blank line before the next variant.

Example:
VARIANT: rs9923231
SCORE: 85
JUSTIFICATION: Citations provide strong statistical evidence (p-values) and effect sizes. Table reference is appropriate. Missing explicit sample size but overall well-supported.

VARIANT: rs1057910
SCORE: 72
JUSTIFICATION: Citations support the general association but lack specific statistical significance values. Effect direction is clear.
"""


def load_citations(citations_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load citation data from JSON file.

    Args:
        citations_path: Path to citations JSON file

    Returns:
        Dictionary with structure {pmcid: {variant: [citations]}}
    """
    logger.debug(f"Loading citations from {citations_path}")
    with open(citations_path) as f:
        data = json.load(f)
    logger.info(f"Loaded citations for {len(data)} PMCID(s)")
    return data


def load_sentence_bench(sentence_bench_path: Path) -> dict[str, dict[str, dict]]:
    """Load sentence benchmark data grouped by PMCID and variant.

    Args:
        sentence_bench_path: Path to sentence_bench.jsonl

    Returns:
        Dictionary with structure {pmcid: {variant: {sentence, explanation}}}
    """
    logger.debug(f"Loading sentence benchmark from {sentence_bench_path}")
    pmcid_data: dict[str, dict[str, dict]] = {}

    with open(sentence_bench_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pmcid = rec["pmcid"]
            variant = rec["variant"]

            if pmcid not in pmcid_data:
                pmcid_data[pmcid] = {}

            # Handle both formats
            sentences = rec.get("sentences", [])
            if sentences and isinstance(sentences[0], dict):
                sentence = sentences[0]["sentence"]
                explanation = sentences[0].get("explanation", "")
            else:
                sentence = sentences[0] if sentences else ""
                explanation = ""

            pmcid_data[pmcid][variant] = {
                "sentence": sentence,
                "explanation": explanation
            }

    logger.info(f"Loaded sentence data for {len(pmcid_data)} PMCID(s)")
    return pmcid_data


def parse_judge_output(output: str) -> dict[str, dict[str, Any]]:
    """Parse judge LLM output into variant scores.

    Expected format:
        VARIANT: rs9923231
        SCORE: 85
        JUSTIFICATION: Citations provide strong evidence...

        VARIANT: rs1057910
        SCORE: 72
        JUSTIFICATION: Citations support the general association...
    """
    result: dict[str, dict[str, Any]] = {}

    # Split by VARIANT: markers
    variant_blocks = re.split(r'\n\s*VARIANT:\s*', output)

    for block in variant_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        if not lines:
            continue

        variant_id = lines[0].strip()
        # Remove "VARIANT:" prefix if present (happens for first variant in output)
        if variant_id.upper().startswith('VARIANT:'):
            variant_id = variant_id[8:].strip()  # Remove "VARIANT:" and any whitespace

        score = None
        justification = ""

        for line in lines[1:]:
            line = line.strip()
            if line.upper().startswith('SCORE:'):
                score_text = line.split(':', 1)[1].strip()
                try:
                    score = float(score_text)
                except ValueError:
                    logger.warning(f"Could not parse score: {score_text}")
            elif line.upper().startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
            elif justification:
                # Continue multi-line justification
                justification += " " + line

        if variant_id and score is not None:
            result[variant_id] = {
                "score": score,
                "justification": justification.strip()
            }
            logger.debug(f"Parsed score {score} for {variant_id}")

    if not result:
        logger.warning("Failed to parse any scores from judge output")
        logger.debug(f"Output was: {output[:500]}...")

    return result


def evaluate_pmcid(
    pmcid: str,
    citations: dict[str, list[str]],
    sentence_data: dict[str, dict],
    judge_model: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate citations for a single PMCID.

    Args:
        pmcid: PMCID identifier
        citations: Dictionary mapping variant -> list of citations
        sentence_data: Dictionary mapping variant -> {sentence, explanation}
        judge_model: Model name for judge LLM

    Returns:
        Dictionary mapping variant -> {score, justification}
    """
    logger.info(f"Evaluating citations for PMCID: {pmcid}")

    # Format associations and citations for the prompt
    associations_text_parts = []
    for variant, cites in citations.items():
        sent_info = sentence_data.get(variant, {})
        sentence = sent_info.get("sentence", "")

        cite_text = "\n   ".join([f"{i+1}. {c}" for i, c in enumerate(cites)])

        associations_text_parts.append(
            f"VARIANT: {variant}\n"
            f"CLAIM: {sentence}\n"
            f"CITATIONS:\n   {cite_text if cites else '(No citations found)'}"
        )

    associations_and_citations = "\n\n".join(associations_text_parts)

    # Create prompt
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        pmcid=pmcid,
        associations_and_citations=associations_and_citations
    )

    # Call judge LLM
    try:
        logger.debug(f"Calling judge LLM for {len(citations)} variant(s)")
        output = call_llm(judge_model, JUDGE_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"Error calling judge for {pmcid}: {e}")
        # Return empty scores
        return {variant: {"score": 0, "justification": "Error during evaluation"} for variant in citations}

    # Parse scores
    scores = parse_judge_output(output)

    # Ensure all variants have scores
    for variant in citations:
        if variant not in scores:
            scores[variant] = {
                "score": 0,
                "justification": "No score provided by judge"
            }
            logger.warning(f"Missing score for {variant}, defaulting to 0")

    return scores


def evaluate_citations(
    citations_path: Path,
    sentence_bench_path: Path,
    judge_model: str,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate all citations and save results.

    Args:
        citations_path: Path to citations JSON file
        sentence_bench_path: Path to sentence_bench.jsonl
        judge_model: Model name for judge LLM
        output_path: Path to save evaluation results

    Returns:
        Dictionary with evaluation summary
    """
    # Load data
    citations_data = load_citations(citations_path)
    sentence_data = load_sentence_bench(sentence_bench_path)

    # Evaluate each PMCID
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    pmcid_summaries = []
    all_scores = []

    for pmcid, pmcid_citations in citations_data.items():
        # Get sentence data for this PMCID
        pmcid_sentences = sentence_data.get(pmcid, {})

        if not pmcid_sentences:
            logger.warning(f"No sentence data found for {pmcid}, skipping evaluation")
            continue

        # Evaluate this PMCID
        scores = evaluate_pmcid(pmcid, pmcid_citations, pmcid_sentences, judge_model)

        all_results[pmcid] = scores

        # Calculate average for this PMCID
        variant_scores = [s["score"] for s in scores.values()]
        avg_score = sum(variant_scores) / len(variant_scores) if variant_scores else 0
        all_scores.extend(variant_scores)

        pmcid_summaries.append({
            "pmcid": pmcid,
            "num_variants": len(scores),
            "avg_score": avg_score,
            "scores": scores
        })

        logger.info(f"✓ {pmcid}: avg score = {avg_score:.2f}")

    # Calculate overall average
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    # Create summary result
    result = {
        "overall_avg_score": overall_avg,
        "num_pmcids": len(pmcid_summaries),
        "num_total_variants": len(all_scores),
        "per_pmcid": pmcid_summaries,
        "details": all_results
    }

    # Save results
    logger.debug(f"Saving evaluation results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.success(f"Saved evaluation results to {output_path}")

    return result


def main():
    """Main entry point for standalone evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate citation quality for pharmacogenomic associations"
    )
    parser.add_argument(
        "--citations",
        required=True,
        type=Path,
        help="Path to citations JSON file",
    )
    parser.add_argument(
        "--sentence-bench",
        type=Path,
        default=ROOT / "data" / "benchmark_v2" / "sentence_bench.jsonl",
        help="Path to sentence_bench.jsonl (default: data/benchmark_v2/sentence_bench.jsonl)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-3-haiku-20240307",
        help="Model to use for judging (default: claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save evaluation results JSON",
    )
    args = parser.parse_args()

    result = evaluate_citations(
        citations_path=args.citations,
        sentence_bench_path=args.sentence_bench,
        judge_model=args.judge_model,
        output_path=args.output,
    )

    logger.info("Evaluation Summary")
    logger.info(f"Overall Average Score: {result['overall_avg_score']:.3f}")
    logger.info(f"Number of PMCIDs: {result['num_pmcids']}")
    logger.info(f"Total Variants Evaluated: {result['num_total_variants']}")


if __name__ == "__main__":
    main()
