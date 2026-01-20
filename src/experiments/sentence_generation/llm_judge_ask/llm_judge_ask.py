"""
LLM Judge Ask - Generate sentences for a single article's variants and save output.

(See llm_judge_ask.md for more details)

This script automates the process of generating sentences using an LLM for given
PMCID variants, saving the output, and optionally evaluating the generated
sentences against ground truth.

Example Commands:

1. Run with default model (claude-sonnet-4-20250514) and prompt (v3) for one PMCID:
   python llm_judge_ask.py

2. Specify a different model and prompt, and process 5 PMCIDs:
   python llm_judge_ask.py --model gpt-4o-mini --prompt v2 --num-pmcids 5

3. Run without automatic evaluation:
   python llm_judge_ask.py --no-eval

4. Specify a different judge model for evaluation:
   python llm_judge_ask.py --model claude-sonnet-4-20250514 --prompt v1 --judge-model claude-haiku-4-5-20250106

We reuse prompts from the raw_sentence_ask experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from litellm import completion
from loguru import logger

# Load environment (API keys, etc.)
load_dotenv()

# Import sentence bench for evaluation
try:
    from src.benchmark_v2.sentence_bench import score_and_save
except ImportError:
    score_and_save = None

# Paths
ROOT = Path(__file__).resolve().parents[4]
VARIANT_BENCH_PATH = ROOT / "data" / "benchmark_v2" / "variant_bench.jsonl"
PROMPTS_FILE = "prompts.yaml"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
RESULTS_DIR = Path(__file__).parent / "results"


def load_prompts() -> dict:
    """Load prompt configurations from prompts.yaml."""
    logger.debug(f"Loading prompts from {PROMPTS_FILE}")
    with open(PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
    logger.info(f"Loaded {len(prompts)} prompt(s)")
    return prompts


def get_n_pmcids_and_variants(n: int) -> list[tuple[str, list[str]]]:
    """Return the first N PMCIDs and their variant lists from variant_bench.jsonl."""
    logger.debug(f"Loading {n} PMCID(s) from {VARIANT_BENCH_PATH}")
    results = []
    with open(VARIANT_BENCH_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            results.append((rec["pmcid"], rec["variants"]))
    logger.info(f"Loaded {len(results)} PMCID(s) with variants")
    return results


def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Call LLM via litellm.completion and return content string."""
    logger.debug(f"Calling LLM with model: {model}")
    # Some models disallow explicit temperature=0; set only when supported
    no_temp_models = model.startswith("o1") or model.startswith("o3") or model.startswith("gpt-5")

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not no_temp_models:
        kwargs["temperature"] = 0

    resp = completion(**kwargs)
    logger.debug(f"LLM response received ({len(resp.choices[0].message.content)} chars)")
    return resp.choices[0].message.content


def split_sentences(text: str) -> list[str]:
    """Split model output into a list of sentences.

    Handles either newline-separated or standard sentence punctuation.
    """
    # If output has newlines, treat each non-empty line as a sentence
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    # Otherwise, split by sentence-ending punctuation.
    # Keep the delimiter by using a regex split with capture then rejoin.
    parts = re.split(r"([.!?])\s+", text.strip())
    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append((parts[i] + parts[i + 1]).strip())
    # If there is a trailing fragment without punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return [s for s in sentences if s]


def process_pmcid(
    pmcid: str,
    variants: list[str],
    model: str,
    prompt_cfg: dict,
    prompt_name: str,
    judge_model: str,
    no_eval: bool,
) -> None:
    """Process a single PMCID: generate sentences and optionally evaluate."""
    logger.info(f"Processing PMCID: {pmcid} with {len(variants)} variant(s)")

    # Get article text; reuse utils for markdown content
    try:
        from src.experiments.utils import (
            get_methods_and_conclusions_text,
            get_markdown_text,
        )
    except Exception:
        get_methods_and_conclusions_text = None
        get_markdown_text = None

    article_text = ""
    if get_methods_and_conclusions_text is not None:
        article_text = get_methods_and_conclusions_text(pmcid)
    if not article_text and get_markdown_text is not None:
        article_text = get_markdown_text(pmcid)

    if not article_text:
        logger.warning(
            f"No article text found for {pmcid}. The model may return generic sentences."
        )

    logger.info(f"Variants: {', '.join(variants)}")

    result: dict[str, dict[str, list[str]]] = {pmcid: {}}

    for variant in variants:
        logger.debug(f"Processing variant: {variant}")
        user_prompt = prompt_cfg["user"].format(variant=variant, article_text=article_text)
        system_prompt = prompt_cfg["system"]

        try:
            output = call_llm(model, system_prompt, user_prompt)
        except Exception as e:
            output = ""
            logger.error(f"Error generating for {pmcid}/{variant}: {e}")

        sentences = split_sentences(output) if output else []
        result[pmcid][variant] = sentences

        preview = sentences[0] if sentences else "<no output>"
        logger.info(f"✓ {variant}: {preview[:90]}{'...' if len(preview) > 90 else ''}")

    # Save output file
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"{safe_model}_{prompt_name}_{timestamp}.json"

    logger.debug(f"Saving generated sentences to {out_path}")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.success(f"Saved generated sentences to {out_path}")

    # Evaluate generated sentences against ground truth
    if not no_eval and score_and_save is not None:
        logger.info("Evaluating sentences against ground truth")
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_judge_model = judge_model.replace("/", "_").replace(":", "_")
            eval_path = RESULTS_DIR / f"sentence_scores_llm_{safe_judge_model}_{timestamp}.json"

            logger.debug(f"Running evaluation with judge model: {judge_model}")
            eval_result = score_and_save(
                generated_sentences_path=out_path,
                pmcid=pmcid,
                method="llm",
                model=judge_model,
                output_path=eval_path,
            )

            logger.info(f"Evaluation Summary for {pmcid}")
            logger.info(f"Average Score: {eval_result.avg_score:.3f}")
            logger.info(f"Number of Variants: {eval_result.num_variants}")
            logger.info("Per-Variant Scores:")
            for variant_result in eval_result.per_variant:
                logger.info(f"  {variant_result['variant']}: {variant_result['score']:.3f} (Jaccard: {variant_result['best_similarity']:.3f})")

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            logger.info("Generated sentences were saved successfully, but evaluation could not be completed.")
    elif no_eval:
        logger.info("Skipping evaluation (--no-eval flag set)")
    elif score_and_save is None:
        logger.info("Skipping evaluation (sentence_bench module not available)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate association sentences for PMCID variants and save JSON."
        )
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model name for litellm (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--prompt",
        default="v3",
        help="Prompt key from prompts.yaml (e.g., v1, v2, v3)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after generation",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20250106",
        help="Model to use for evaluation judging (default: claude-haiku-4-5-20250106)",
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

    # Get PMCIDs to process
    pmcids_and_variants = get_n_pmcids_and_variants(args.num_pmcids)

    logger.info(f"Prompt: {args.prompt} ({prompt_cfg.get('name', '')})")
    logger.info(f"Generation Model: {args.model}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"Processing {len(pmcids_and_variants)} PMCID(s)")

    # Process each PMCID
    for pmcid, variants in pmcids_and_variants:
        process_pmcid(
            pmcid=pmcid,
            variants=variants,
            model=args.model,
            prompt_cfg=prompt_cfg,
            prompt_name=args.prompt,
            judge_model=args.judge_model,
            no_eval=args.no_eval,
        )

    logger.success(f"Completed processing {len(pmcids_and_variants)} PMCID(s)")


if __name__ == "__main__":
    main()

