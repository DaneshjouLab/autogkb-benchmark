"""
Compare var_fa_ann annotations between ground truth and benchmark JSONs
for the 5 PMCs present in data/benchmark_gt_annotations.json and store
results as a Markdown report.
"""

from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import sys

from src.fa_benchmark.fa_benchmark import evaluate_functional_analysis


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def get_fa_by_variant_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        vid = rec.get("Variant Annotation ID")
        if vid is None:
            # Fallback to normalized if present
            vid = rec.get("Variant Annotation ID_norm")
        if vid is None:
            # Skip entries without identifiers
            continue
        by_id[str(vid)] = rec
    return by_id


def align_annotations(
    gt_fa: List[Dict[str, Any]],
    bench_fa: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Align ground-truth and benchmark annotations by Variant Annotation ID.
    Returns (gt_list, pred_list, used_variant_ids)
    """
    gt_by_id = get_fa_by_variant_id(gt_fa)
    pred_by_id = get_fa_by_variant_id(bench_fa)

    common_ids = [vid for vid in gt_by_id.keys() if vid in pred_by_id]

    aligned_gt: List[Dict[str, Any]] = [gt_by_id[vid] for vid in common_ids]
    aligned_pred: List[Dict[str, Any]] = [pred_by_id[vid] for vid in common_ids]
    return aligned_gt, aligned_pred, common_ids


def format_field_scores(field_scores: Dict[str, Any]) -> str:
    lines: List[str] = []
    for field, scores in field_scores.items():
        mean_score = scores["mean_score"]
        lines.append(f"- **{field}**: {mean_score:.3f}")
    return "\n".join(lines)


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    gt_path = root / "data/benchmark_gt_annotations.json"
    bench_path = root / "data/benchmark_annotations.json"
    out_dir = root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_comparison.md"

    gt_json = load_json(gt_path)
    bench_json = load_json(bench_path)

    # If PMCs were provided as CLI args, use those; otherwise use all in GT (up to 5)
    cli_pmcs = [arg for arg in sys.argv[1:] if arg.startswith("PMC")]
    if cli_pmcs:
        pmcs = cli_pmcs
    else:
        pmcs = list(gt_json.keys())[:5]

    report_lines: List[str] = []
    report_lines.append("## Benchmark Comparison Report")
    report_lines.append("")
    report_lines.append(f"PMCs compared: {', '.join(pmcs)}")
    report_lines.append("")

    overall_scores: List[float] = []

    for pmcid in pmcs:
        gt_article = gt_json.get(pmcid, {})
        bench_article = bench_json.get(pmcid, {})

        gt_fa = gt_article.get("var_fa_ann", []) or []
        bench_fa = bench_article.get("var_fa_ann", []) or []

        if not gt_fa or not bench_fa:
            report_lines.append(f"### {pmcid}")
            report_lines.append("- **Status**: Missing var_fa_ann in one or both datasets")
            report_lines.append("")
            continue

        aligned_gt, aligned_pred, common_ids = align_annotations(gt_fa, bench_fa)

        if not aligned_gt:
            report_lines.append(f"### {pmcid}")
            report_lines.append("- **Status**: No overlapping Variant Annotation IDs to compare")
            report_lines.append("")
            continue

        results = evaluate_functional_analysis(aligned_gt, aligned_pred)

        report_lines.append(f"### {pmcid}")
        report_lines.append(f"- **Aligned Variant Annotation IDs**: {', '.join(common_ids)}")
        report_lines.append(f"- **Samples compared**: {results['total_samples']}")
        report_lines.append(f"- **Overall score**: {results['overall_score']:.3f}")
        report_lines.append("")
        report_lines.append("#### Average field scores")
        report_lines.append(format_field_scores(results["field_scores"]))
        report_lines.append("")
        
        # Dependency validation issues
        all_issues = []
        for sample in results['detailed_results']:
            if sample.get('dependency_issues'):
                all_issues.extend(sample['dependency_issues'])
        
        if all_issues:
            report_lines.append("#### Dependency Validation Issues")
            for issue in all_issues[:10]:  # Show first 10
                report_lines.append(f"- {issue}")
            if len(all_issues) > 10:
                report_lines.append(f"- ... and {len(all_issues) - 10} more")
            report_lines.append("")

        overall_scores.append(results["overall_score"])

    if overall_scores:
        macro_avg = sum(overall_scores) / len(overall_scores)
        report_lines.insert(3, f"Macro-average overall score: {macro_avg:.3f}")

    out_path.write_text("\n".join(report_lines))


if __name__ == "__main__":
    run()


