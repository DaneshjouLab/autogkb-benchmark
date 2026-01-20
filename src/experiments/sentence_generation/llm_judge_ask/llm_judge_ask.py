"""
LLM Judge Ask - Generate sentences for a single article's variants and save output.

Per llm_judge_ask.md:
- Load a prompt by name from prompts.yaml
- Use the first PMCID + its variants from variant_bench.jsonl
- Generate sentences with an LLM and save to a timestamped JSON under ./outputs

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

# Load environment (API keys, etc.)
load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parents[4]
VARIANT_BENCH_PATH = ROOT / "data" / "benchmark_v2" / "variant_bench.jsonl"
PROMPTS_FILE = "prompts.yaml"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def load_prompts() -> dict:
    with open(PROMPTS_FILE) as f:
        return yaml.safe_load(f)


def get_first_pmcid_and_variants() -> tuple[str, list[str]]:
    """Return the first PMCID and its variant list from variant_bench.jsonl."""
    with open(VARIANT_BENCH_PATH) as f:
        first_line = f.readline()
        if not first_line:
            raise RuntimeError("variant_bench.jsonl is empty")
        rec = json.loads(first_line)
        return rec["pmcid"], rec["variants"]


def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Call LLM via litellm.completion and return content string."""
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate association sentences for the first PMCID's variants and save JSON."
        )
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model name for litellm (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--prompt",
        default="v1",
        help="Prompt key from prompts.yaml (e.g., v1, v2, v3)",
    )
    args = parser.parse_args()

    prompts = load_prompts()
    if args.prompt not in prompts:
        raise KeyError(f"Prompt '{args.prompt}' not found in {PROMPTS_FILE}")
    prompt_cfg = prompts[args.prompt]

    pmcid, variants = get_first_pmcid_and_variants()

    # Get article text; reuse utils for markdown content
    # Prefer minimal dependency: use methods+conclusions if available, else full markdown
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
        print(
            f"Warning: No article text found for {pmcid}. The model may return generic sentences."
        )

    print(f"PMCID: {pmcid}")
    print(f"Variants: {', '.join(variants)}")
    print(f"Prompt: {args.prompt} ({prompt_cfg.get('name', '')})")
    print(f"Model: {args.model}")

    result: dict[str, dict[str, list[str]]] = {pmcid: {}}

    for variant in variants:
        user_prompt = prompt_cfg["user"].format(variant=variant, article_text=article_text)
        system_prompt = prompt_cfg["system"]

        try:
            output = call_llm(args.model, system_prompt, user_prompt)
        except Exception as e:
            output = f""  # Leave empty on error
            print(f"Error generating for {pmcid}/{variant}: {e}")

        sentences = split_sentences(output) if output else []
        result[pmcid][variant] = sentences

        # Brief console feedback
        preview = sentences[0] if sentences else "<no output>"
        print(f"  ✓ {variant}: {preview[:90]}{'...' if len(preview) > 90 else ''}")

    # Save output file
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"{safe_model}_{args.prompt}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

