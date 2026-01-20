"""
Goal:
Given a set of ground truth sentences and a set of generated sentences, use an LLM judge to evaluate the similarity between them
and score the results. It goes variant-by-variant and produces a 0-1 score per variant (evaluating all the sentences for that variant
against all of the ground truths for that variant).
It should also produce a summary of the evaluation.
We do not care about exact wording, only whether the associations are generally correct (variant, direction, effect, phenotype, drug, comparison).

Notes:
- Use litellm for API calls
- Use load_dotenv() for API keys
- Ground truth sentences are stored in sentence_bench.jsonl
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

# Load environment variables for API keys
load_dotenv()


@dataclass
class VariantSentenceScore:
    """Score for a single variant's sentences."""

    variant: str
    ground_truth: list[str]
    generated: list[str]
    score: float
    best_similarity: float
    best_gt: str
    best_gen: str


@dataclass
class SentenceBenchResult:
    """Overall result for sentence benchmarking."""

    timestamp: str
    pmcid: str
    method: str
    judge_model: str | None
    source_file: str
    avg_score: float
    num_variants: int
    per_variant: list[dict]
    generated_extras: list[str]


def load_sentence_bench_data() -> dict[str, dict[str, list[str]]]:
    """Load the sentence benchmark data from the jsonl file.

    Returns:
        dict mapping pmcid -> variant -> list of ground truth sentences
    """
    data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "benchmark_v2"
        / "sentence_bench.jsonl"
    )

    pmcid_data: dict[str, dict[str, list[str]]] = {}

    with open(data_path) as f:
        for line in f:
            record = json.loads(line)
            pmcid = record["pmcid"]
            variant = record["variant"]
            sentences = record["sentences"]

            if pmcid not in pmcid_data:
                pmcid_data[pmcid] = {}

            pmcid_data[pmcid][variant] = sentences

    return pmcid_data


def jaccard_similarity(sent1: str, sent2: str) -> float:
    """Calculate Jaccard similarity between two sentences based on word tokens.

    Args:
        sent1: First sentence
        sent2: Second sentence

    Returns:
        Jaccard similarity score (0 to 1)
    """
    # Tokenize and normalize
    tokens1 = set(sent1.lower().split())
    tokens2 = set(sent2.lower().split())

    # Calculate Jaccard similarity
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def llm_judge_score(
    ground_truth_sentences: list[str],
    generated_sentences: list[str],
    variant: str,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[float, str]:
    """Use an LLM to judge the similarity between ground truth and generated sentences.

    Args:
        ground_truth_sentences: List of ground truth sentences for this variant
        generated_sentences: List of generated sentences for this variant
        variant: The variant name being evaluated
        model: LLM model to use for judging

    Returns:
        Tuple of (score from 0-1, explanation)
    """
    prompt = f"""You are evaluating whether generated pharmacogenomic sentences capture the same associations as ground truth sentences.

Variant: {variant}

Ground Truth Sentences:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(ground_truth_sentences))}

Generated Sentences:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(generated_sentences))}

Evaluate whether the generated sentences capture the same pharmacogenomic associations as the ground truth. Focus on:
- Variant/genotype mentioned
- Direction of association (increased/decreased/not associated)
- Effect type (dose, risk, likelihood, etc.)
- Phenotype/condition
- Drug mentioned
- Comparison groups

Provide a similarity score from 0 to 1:
- 1.0: Perfect match - all key associations are captured correctly
- 0.7-0.9: Most associations captured, minor differences in specificity or wording
- 0.4-0.6: Some associations captured but missing key details or has inaccuracies
- 0.1-0.3: Associations are mostly incorrect or contradictory
- 0.0: Completely incorrect or contradictory associations

Provide your response in this exact JSON format:
{{"score": <float between 0 and 1>, "explanation": "<brief explanation of your scoring>"}}"""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content
        # Parse JSON response
        result = json.loads(content)
        score = float(result["score"])
        explanation = result["explanation"]

        return score, explanation

    except Exception as e:
        print(f"Error in LLM judge for variant {variant}: {e}")
        # Return a middle score as fallback
        return 0.5, f"Error during evaluation: {str(e)}"


def score_variant_sentences(
    variant: str,
    ground_truth: list[str],
    generated: list[str],
    model: str = "claude-sonnet-4-20250514",
) -> VariantSentenceScore:
    """Score generated sentences for a single variant against ground truth.

    Args:
        variant: Variant identifier
        ground_truth: List of ground truth sentences
        generated: List of generated sentences
        model: LLM model to use for judging

    Returns:
        VariantSentenceScore with detailed metrics
    """
    # Get LLM judge score
    llm_score, _explanation = llm_judge_score(ground_truth, generated, variant, model)

    # Calculate best Jaccard similarity
    best_similarity = 0.0
    best_gt = ""
    best_gen = ""

    for gt_sent in ground_truth:
        for gen_sent in generated:
            sim = jaccard_similarity(gt_sent, gen_sent)
            if sim > best_similarity:
                best_similarity = sim
                best_gt = gt_sent
                best_gen = gen_sent

    # If no sentences, use first of each as best
    if not best_gt and ground_truth:
        best_gt = ground_truth[0]
    if not best_gen and generated:
        best_gen = generated[0]

    return VariantSentenceScore(
        variant=variant,
        ground_truth=ground_truth,
        generated=generated,
        score=llm_score,
        best_similarity=best_similarity,
        best_gt=best_gt,
        best_gen=best_gen,
    )


def score_generated_sentences(
    generated_sentences_path: str | Path,
    pmcid: str | None = None,
    method: str = "llm",
    model: str = "claude-sonnet-4-20250514",
) -> SentenceBenchResult:
    """Score generated sentences from a JSON file against ground truth.

    Args:
        generated_sentences_path: Path to JSON file with generated sentences.
            Expected format: {pmcid: {variant: [sentences], ...}, ...}
        pmcid: Optional PMCID to score. If None, scores the first PMCID in file.
        method: Method name for this evaluation
        model: LLM model to use for judging

    Returns:
        SentenceBenchResult with detailed scoring
    """
    generated_sentences_path = Path(generated_sentences_path)

    # Load generated sentences
    with open(generated_sentences_path) as f:
        generated_data = json.load(f)

    # Get PMCID to evaluate
    if pmcid is None:
        # Get first PMCID in file
        pmcid = next(k for k in generated_data.keys() if k.startswith("PMC"))

    if pmcid not in generated_data:
        raise ValueError(f"PMCID {pmcid} not found in generated sentences file")

    # Load ground truth data
    ground_truth_data = load_sentence_bench_data()

    if pmcid not in ground_truth_data:
        raise ValueError(f"PMCID {pmcid} not found in ground truth data")

    # Get sentences for this PMCID
    generated_variants = generated_data[pmcid]
    ground_truth_variants = ground_truth_data[pmcid]

    # Score each variant
    per_variant_scores = []
    total_score = 0.0
    num_variants = 0

    for variant, gt_sentences in ground_truth_variants.items():
        gen_sentences = generated_variants.get(variant, [])

        variant_score = score_variant_sentences(variant, gt_sentences, gen_sentences, model)

        per_variant_scores.append(
            {
                "variant": variant,
                "ground_truth": variant_score.ground_truth,
                "generated": variant_score.generated,
                "score": variant_score.score,
                "best_similarity": variant_score.best_similarity,
                "best_gt": variant_score.best_gt,
                "best_gen": variant_score.best_gen,
            }
        )

        total_score += variant_score.score
        num_variants += 1

    # Calculate average score
    avg_score = total_score / num_variants if num_variants > 0 else 0.0

    # Find extra variants in generated that aren't in ground truth
    generated_extras = [v for v in generated_variants.keys() if v not in ground_truth_variants]

    # Create result
    return SentenceBenchResult(
        timestamp=datetime.now().isoformat(),
        pmcid=pmcid,
        method=method,
        judge_model=model,
        source_file=str(generated_sentences_path),
        avg_score=round(avg_score, 3),
        num_variants=num_variants,
        per_variant=per_variant_scores,
        generated_extras=generated_extras,
    )


def score_and_save(
    generated_sentences_path: str | Path,
    pmcid: str | None = None,
    method: str = "llm",
    model: str = "claude-sonnet-4-20250514",
    output_path: str | Path | None = None,
) -> SentenceBenchResult:
    """Score generated sentences and save results to a JSON file.

    Args:
        generated_sentences_path: Path to JSON file with generated sentences
        pmcid: Optional PMCID to score
        method: Method name for this evaluation
        model: LLM model to use for judging
        output_path: Path to save results. If None, auto-generates name.

    Returns:
        SentenceBenchResult
    """
    result = score_generated_sentences(generated_sentences_path, pmcid, method, model)

    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = model.replace("/", "_").replace(":", "_") if model else "none"
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "benchmark_v2"
            / "sentence_bench_results"
            / f"sentence_scores_{result.pmcid}_{method}_{model_safe}_{timestamp}.json"
        )
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    result_dict = {
        "timestamp": result.timestamp,
        "pmcid": result.pmcid,
        "method": result.method,
        "judge_model": result.judge_model,
        "source_file": result.source_file,
        "avg_score": result.avg_score,
        "num_variants": result.num_variants,
        "per_variant": result.per_variant,
        "generated_extras": result.generated_extras,
    }

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Average score: {result.avg_score:.3f}")
    print(f"Number of variants: {result.num_variants}")

    return result


def main():
    """Test the sentence scoring functions."""
    # Example usage
    generated_file = (
        Path(__file__).parent.parent.parent
        / "src"
        / "experiments"
        / "sentence_generation"
        / "llm_judge_ask"
        / "outputs"
        / "openai_gpt-4o-mini_v1_20260119_223926.json"
    )

    print(f"Scoring sentences from: {generated_file}")

    # Score for PMC5508045
    result = score_and_save(
        generated_sentences_path=generated_file,
        pmcid="PMC5508045",
        method="llm",
        model="gpt-4o-mini",
    )

    # Print detailed results
    print("\n=== Detailed Results ===")
    for variant_result in result.per_variant:
        print(f"\nVariant: {variant_result['variant']}")
        print(f"  Score: {variant_result['score']:.3f}")
        print(f"  Jaccard Similarity: {variant_result['best_similarity']:.3f}")
        print(f"  Ground Truth: {variant_result['best_gt'][:100]}...")
        print(f"  Generated: {variant_result['best_gen'][:100]}...")


if __name__ == "__main__":
    main()
