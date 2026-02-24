from src.fa_benchmark.fa_benchmark import evaluate_functional_analysis
import json
from typing import Dict, Any

# Load your predictions
with open("./persistent_data/llm_outputs/combined_output_11_02_25.json", "r") as f:
    predictions: Dict[str, Any] = json.load(f)


# Load ground truth
with open("data/benchmark_annotations.json", "r") as f:
    data = json.load(f)

# compile predictions for common files
pmids_gt = [gt.get("PMID") for gt in ground_truth if gt.get("PMID")]
pmids_pred = [pred.get("PMID") for pred in predictions if pred.get("PMID")]
common_pmids = set(pmids_gt).intersection(set(pmids_pred))
ground_truth = [gt for gt in ground_truth if gt.get("PMID") in common_pmids]
predictions = [pred for pred in predictions if pred.get("PMID") in common_pmids]

# Extract functional analysis annotations
gt_annotations = []
for pmcid, article_data in data.items():
    if "var_fa_ann" in article_data:
        gt_annotations.extend(article_data["var_fa_ann"])


# Run evaluation
results = evaluate_functional_analysis(gt_annotations, preds)
print(f"Overall Score: {results['overall_score']:.3f}")
