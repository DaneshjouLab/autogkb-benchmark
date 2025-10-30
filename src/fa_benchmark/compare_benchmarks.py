"""
Compare var_fa_ann annotations between ground truth and benchmark JSONs
for the 5 PMCs present in data/benchmark_gt_annotations.json and store
results as a Markdown report.
"""

from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import sys

from src.fa_benchmark.fa_benchmark import (
    evaluate_functional_analysis,
    expand_annotations_by_variant,
    normalize_variant,
)


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
    # Expand multi-variant annotations into per-variant records
    gt_expanded = expand_annotations_by_variant(gt_fa)
    pred_expanded = expand_annotations_by_variant(bench_fa)

    # Build maps keyed by (Variant Annotation ID, normalized_variant) with fallback to (id, '')
    def keyed_map(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        m: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rec in records:
            vid_raw = rec.get('Variant Annotation ID')
            vid = str(vid_raw) if vid_raw is not None else ''
            var = rec.get('Variant/Haplotypes') or ''
            key = (vid, normalize_variant(var) if var else '')
            # Prefer exact variant key when available
            if key not in m:
                m[key] = rec
        return m

    gt_map = keyed_map(gt_expanded)
    pred_map = keyed_map(pred_expanded)

    # Determine common keys; if either side lacks a variant token (empty ''),
    # try to match on (id, any) by falling back to (id, '')
    common_keys: List[Tuple[str, str]] = []
    for key in gt_map.keys():
        if key in pred_map:
            common_keys.append(key)
        else:
            vid, var = key
            fallback = (vid, '')
            if fallback in pred_map:
                common_keys.append(fallback)

    aligned_gt: List[Dict[str, Any]] = [gt_map[k] for k in common_keys]
    aligned_pred: List[Dict[str, Any]] = [pred_map[k] for k in common_keys]
    display_keys = [f"{k[0]}:{k[1]}" if k[1] else k[0] for k in common_keys]
    return aligned_gt, aligned_pred, display_keys


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

        overall_scores.append(results["overall_score"])

    if overall_scores:
        macro_avg = sum(overall_scores) / len(overall_scores)
        report_lines.insert(3, f"Macro-average overall score: {macro_avg:.3f}")

    out_path.write_text("\n".join(report_lines))


if __name__ == "__main__":
    run()


