"""
One Shot Citations - Find supporting citations for pharmacogenomic association sentences.

(See one_shot_citations.md for more details)

This script takes generated pharmacogenomic association sentences and finds
supporting citations (2-5 sentences) from the source articles that support each claim.

Example Commands:

1. Run with default model (gpt-4o) and prompt (v1) for one PMCID:
   python one_shot_citations.py

2. Specify a different model and prompt, and process 3 PMCIDs:
   python one_shot_citations.py --model gpt-4o-mini --prompt v2 --num-pmcids 3

3. Run without automatic evaluation:
   python one_shot_citations.py --no-eval

4. Specify a different judge model for evaluation:
   python one_shot_citations.py --model gpt-4o --prompt v1 --judge-model claude-3-haiku-20240307

Prompts:
- v1: Basic citation finding - finds 3-5 supporting sentences
- v2: Citation finding with explanation context - uses explanation to guide search
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import yaml
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
from src.experiments.utils import call_llm, get_markdown_text

SENTENCE_BENCH_PATH = ROOT / "data" / "benchmark_v2" / "sentence_bench.jsonl"
PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
RESULTS_DIR = Path(__file__).parent / "results"


def load_prompts() -> dict:
    """Load prompt configurations from prompts.yaml."""
    logger.debug(f"Loading prompts from {PROMPTS_FILE}")
    with open(PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
    logger.info(f"Loaded {len(prompts)} prompt(s)")
    return prompts


def load_sentence_data(num_pmcids: int | None = None) -> dict[str, list[dict]]:
    """Load sentence data from sentence_bench.jsonl grouped by PMCID.

    Args:
        num_pmcids: Optional limit on number of PMCIDs to load

    Returns:
        Dictionary mapping pmcid -> list of {variant, sentences, explanation} dicts
    """
    logger.debug(f"Loading sentence data from {SENTENCE_BENCH_PATH}")
    pmcid_data: dict[str, list[dict]] = {}

    with open(SENTENCE_BENCH_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pmcid = rec["pmcid"]

            if pmcid not in pmcid_data:
                pmcid_data[pmcid] = []

            # Handle both formats: sentences can be list of strings or list of dicts
            sentences = rec.get("sentences", [])
            if sentences and isinstance(sentences[0], dict):
                # Format: [{"sentence": "...", "explanation": "..."}]
                entry = {
                    "variant": rec["variant"],
                    "sentence": sentences[0]["sentence"],
                    "explanation": sentences[0].get("explanation", "")
                }
            else:
                # Format: ["sentence string"]
                entry = {
                    "variant": rec["variant"],
                    "sentence": sentences[0] if sentences else "",
                    "explanation": ""
                }

            pmcid_data[pmcid].append(entry)

    # Limit to num_pmcids if specified
    if num_pmcids is not None:
        pmcid_data = dict(list(pmcid_data.items())[:num_pmcids])

    logger.info(f"Loaded data for {len(pmcid_data)} PMCID(s)")
    return pmcid_data


def parse_citation_output(output: str) -> dict[str, list[str]]:
    """Parse LLM output into a dict mapping variant -> list of citations.

    Expected format:
        VARIANT: rs9923231
        CITATIONS:
        1. First citation sentence
        2. Second citation sentence
        3. Third citation sentence

        VARIANT: rs1057910
        CITATIONS:
        1. First citation sentence
        2. Second citation sentence
    """
    result: dict[str, list[str]] = {}

    # Split by VARIANT: markers
    variant_blocks = re.split(r'\n\s*VARIANT:\s*', output)

    for block in variant_blocks:
        if not block.strip():
            continue

        # First line should be variant ID, rest is citations
        lines = block.strip().split('\n')
        if not lines:
            continue

        variant_id = lines[0].strip()
        # Remove "VARIANT:" prefix if present (happens for first variant in output)
        if variant_id.upper().startswith('VARIANT:'):
            variant_id = variant_id[8:].strip()  # Remove "VARIANT:" and any whitespace

        # Find CITATIONS: section
        citations = []
        in_citations = False

        for line in lines[1:]:
            line = line.strip()
            if line.upper().startswith('CITATIONS:'):
                in_citations = True
                continue

            if in_citations and line:
                # Remove leading numbers like "1. " or "1) "
                citation = re.sub(r'^\d+[\.)]\s*', '', line)
                if citation:
                    citations.append(citation)

        if variant_id and citations:
            result[variant_id] = citations
            logger.debug(f"Parsed {len(citations)} citation(s) for {variant_id}")

    if not result:
        logger.warning("Failed to parse any citations from output")
        logger.debug(f"Output was: {output[:500]}...")

    return result


def process_pmcid(
    pmcid: str,
    associations: list[dict],
    model: str,
    prompt_cfg: dict,
    prompt_name: str,
) -> dict[str, dict[str, list[str]]]:
    """Process a single PMCID: find citations for all associations.

    Args:
        pmcid: PMCID identifier
        associations: List of {variant, sentence, explanation} dicts
        model: Model name for LLM
        prompt_cfg: Prompt configuration from prompts.yaml
        prompt_name: Name of the prompt being used

    Returns:
        Dictionary with structure {pmcid: {variant: [citations]}}
    """
    logger.info(
        f"Processing PMCID: {pmcid} with {len(associations)} association(s)"
    )

    # Get article text
    article_text = get_markdown_text(pmcid)
    if not article_text:
        logger.warning(
            f"No article text found for {pmcid}. Citations may be empty."
        )

    # Format associations for the prompt
    if prompt_name == "v2":
        # Include explanations
        associations_text = "\n".join([
            f"- Variant: {a['variant']}\n  Sentence: {a['sentence']}\n  Explanation: {a['explanation']}"
            for a in associations
        ])
    else:
        # Just sentences
        associations_text = "\n".join([
            f"- Variant: {a['variant']}\n  Sentence: {a['sentence']}"
            for a in associations
        ])

    # Create the prompt
    user_prompt = prompt_cfg["user"].format(
        associations=associations_text,
        article_text=article_text
    )
    system_prompt = prompt_cfg["system"]

    # Call LLM
    try:
        logger.debug(f"Making LLM call for {len(associations)} association(s)")
        output = call_llm(model, system_prompt, user_prompt)
    except Exception as e:
        output = ""
        logger.error(f"Error generating citations for {pmcid}: {e}")

    # Parse the output
    variant_citations = parse_citation_output(output)

    # Build result structure
    result: dict[str, dict[str, list[str]]] = {pmcid: {}}

    # Ensure all requested variants are in the result
    for assoc in associations:
        variant = assoc["variant"]
        if variant in variant_citations:
            result[pmcid][variant] = variant_citations[variant]
            logger.info(
                f"✓ {variant}: {len(variant_citations[variant])} citation(s) found"
            )
        else:
            # Variant not found in parsed output
            result[pmcid][variant] = []
            logger.warning(f"✗ {variant}: no citations found in output")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find supporting citations for pharmacogenomic association sentences."
        )
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name for litellm (default: gpt-4o)",
    )
    parser.add_argument(
        "--prompt",
        default="v1",
        help="Prompt key from prompts.yaml (e.g., v1, v2)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after generation",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-3-haiku-20240307",
        help="Model to use for evaluation judging (default: claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--num-pmcids",
        type=int,
        default=1,
        help="Number of PMCIDs to process (default: 1)",
    )
    args = parser.parse_args()

    prompts = load_prompts()
    if args.prompt not in prompts:
        raise KeyError(f"Prompt '{args.prompt}' not found in {PROMPTS_FILE}")
    prompt_cfg = prompts[args.prompt]

    # Load sentence data
    pmcid_data = load_sentence_data(args.num_pmcids)

    logger.info(f"Prompt: {args.prompt} ({prompt_cfg.get('name', '')})")
    logger.info(f"Citation Model: {args.model}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"Processing {len(pmcid_data)} PMCID(s)")

    # Generate timestamp once for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect results from all PMCIDs
    all_results: dict[str, dict[str, list[str]]] = {}

    # Process each PMCID
    for pmcid, associations in pmcid_data.items():
        pmcid_result = process_pmcid(
            pmcid=pmcid,
            associations=associations,
            model=args.model,
            prompt_cfg=prompt_cfg,
            prompt_name=args.prompt,
        )
        # Merge results into the combined dictionary
        all_results.update(pmcid_result)

    # Save all results to a single output file
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = OUTPUTS_DIR / f"citations_{safe_model}_{args.prompt}_{timestamp}.json"

    logger.debug(f"Saving citations to {out_path}")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.success(f"Saved citations for {len(all_results)} PMCID(s) to {out_path}")

    # Evaluate citations if requested
    if not args.no_eval:
        logger.info("Running citation quality evaluation")
        try:
            from src.experiments.citations.one_shot_citations.citation_judge import (
                evaluate_citations
            )

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_judge_model = args.judge_model.replace("/", "_").replace(":", "_")
            eval_path = (
                RESULTS_DIR / f"citation_scores_{safe_judge_model}_{timestamp}.json"
            )

            logger.debug(f"Running evaluation with judge model: {args.judge_model}")
            eval_result = evaluate_citations(
                citations_path=out_path,
                sentence_bench_path=SENTENCE_BENCH_PATH,
                judge_model=args.judge_model,
                output_path=eval_path,
            )

            logger.info("Evaluation Summary")
            logger.info(f"Overall Average Score: {eval_result['overall_avg_score']:.3f}")
            logger.info(f"Number of PMCIDs: {len(eval_result['per_pmcid'])}")
            logger.info("Per-PMCID Scores:")
            for pmcid_result in eval_result['per_pmcid']:
                logger.info(
                    f"  {pmcid_result['pmcid']}: {pmcid_result['avg_score']:.3f} "
                    f"({pmcid_result['num_variants']} variants)"
                )

        except ImportError:
            logger.warning("citation_judge module not found, skipping evaluation")
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            logger.info(
                "Citations were saved successfully, but evaluation could not be completed."
            )
    else:
        logger.info("Skipping evaluation (--no-eval flag set)")

    logger.success(f"Completed processing {len(pmcid_data)} PMCID(s)")


if __name__ == "__main__":
    main()
