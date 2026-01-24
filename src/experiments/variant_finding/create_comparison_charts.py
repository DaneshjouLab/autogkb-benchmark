"""
Create comprehensive comparison charts for all variant extraction methods.

Compares:
- Regex v1-v5
- Term normalization hybrid
- Regex + LLM filter
- LLM-only approaches
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_results(path):
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_metrics(results):
    """Extract key metrics from results."""
    # Handle LLM filter results (different structure)
    if "filtered_results" in results:
        filtered = results["filtered_results"]
        recall = filtered.get("avg_recall", 0)
        precision = filtered.get("avg_precision", 0)
        perfect = filtered.get("perfect_recalls", 0)
        total = results.get("articles_processed", 0)
    else:
        # Standard results structure
        recall = results.get("avg_recall", 0)
        precision = results.get("avg_precision", 0)
        perfect = sum(
            1
            for r in results.get("per_article_results", [])
            if r.get("recall", 0) == 1.0
        )
        total = len(results.get("per_article_results", []))

    return {
        "recall": recall * 100,
        "precision": precision * 100,
        "f1": 2 * recall * precision / (recall + precision + 1e-10) * 100,
        "perfect_recall": perfect,
        "total_articles": total,
    }


# Load all results
base_dir = Path(__file__).parent

results_data = {}

# Regex versions
for v in range(1, 6):
    path = base_dir / "regex_variants" / f"results_v{v}.json"
    if path.exists():
        results_data[f"Regex v{v}"] = extract_metrics(load_results(path))

# Term norm hybrid
term_norm_path = base_dir / "regex_term_norm" / "results_term_norm.json"
if term_norm_path.exists():
    results_data["Term Norm Hybrid"] = extract_metrics(load_results(term_norm_path))

# Regex + LLM filter (Note: only 2 articles tested, exclude from main comparison)
# llm_filter_dir = base_dir / 'regex_llm_filter' / 'results'
# if llm_filter_dir.exists():
#     for result_file in llm_filter_dir.glob('*.json'):
#         if 'claude-sonnet-4-5' in result_file.name:
#             model_name = 'Regex + Claude Filter (2 articles)'
#         elif 'gpt-4o' in result_file.name:
#             model_name = 'Regex + GPT-4o Filter (2 articles)'
#         else:
#             continue
#         results_data[model_name] = extract_metrics(load_results(result_file))

# Just ask (LLM only)
just_ask_dir = base_dir / "just_ask" / "results"
if just_ask_dir.exists():
    llm_results = {}
    for result_file in just_ask_dir.glob("*.json"):
        if "claude-opus-4-5" in result_file.name and "v3" in result_file.name:
            model_name = "LLM-Only (Opus 4.5)"
        elif "claude-sonnet-4-5" in result_file.name:
            model_name = "LLM-Only (Sonnet 4.5)"
        elif "gpt-4o" in result_file.name and "v2" in result_file.name:
            model_name = "LLM-Only (GPT-4o)"
        else:
            continue
        llm_results[model_name] = extract_metrics(load_results(result_file))

    # Add LLM results at the end
    results_data.update(llm_results)

print(f"Loaded {len(results_data)} results files")
for name, metrics in results_data.items():
    print(
        f"  {name}: {metrics['recall']:.1f}% recall, {metrics['precision']:.1f}% precision"
    )

# Create comprehensive comparison charts
fig = plt.figure(figsize=(16, 12))

# Color schemes
regex_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
llm_colors = ["#e377c2", "#7f7f7f", "#bcbd22"]

# Separate methods by category
regex_methods = [
    k
    for k in results_data.keys()
    if k.startswith("Regex") and "Filter" not in k and "Hybrid" not in k
]
hybrid_methods = [k for k in results_data.keys() if "Hybrid" in k]
llm_methods = [k for k in results_data.keys() if "LLM-Only" in k]

all_methods = regex_methods + hybrid_methods + llm_methods

# ============================================================================
# Chart 1: Recall Comparison
# ============================================================================
ax1 = plt.subplot(3, 2, 1)
recalls = [results_data[m]["recall"] for m in all_methods]
colors = (
    ["#1f77b4"] * len(regex_methods)
    + ["#ff7f0e"] * len(hybrid_methods)
    + ["#2ca02c"] * len(llm_methods)
)

bars = ax1.barh(
    range(len(all_methods)), recalls, color=colors, alpha=0.7, edgecolor="black"
)
ax1.set_yticks(range(len(all_methods)))
ax1.set_yticklabels(all_methods, fontsize=9)
ax1.set_xlabel("Recall (%)", fontweight="bold")
ax1.set_title("Recall Comparison", fontweight="bold", fontsize=12)
ax1.set_xlim(0, 100)
ax1.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, recalls)):
    ax1.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=8)

# ============================================================================
# Chart 2: Precision Comparison
# ============================================================================
ax2 = plt.subplot(3, 2, 2)
precisions = [results_data[m]["precision"] for m in all_methods]

bars = ax2.barh(
    range(len(all_methods)), precisions, color=colors, alpha=0.7, edgecolor="black"
)
ax2.set_yticks(range(len(all_methods)))
ax2.set_yticklabels(all_methods, fontsize=9)
ax2.set_xlabel("Precision (%)", fontweight="bold")
ax2.set_title("Precision Comparison", fontweight="bold", fontsize=12)
ax2.set_xlim(0, 100)
ax2.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, precisions)):
    ax2.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=8)

# ============================================================================
# Chart 3: F1 Score
# ============================================================================
ax3 = plt.subplot(3, 2, 3)
f1_scores = [results_data[m]["f1"] for m in all_methods]

bars = ax3.barh(
    range(len(all_methods)), f1_scores, color=colors, alpha=0.7, edgecolor="black"
)
ax3.set_yticks(range(len(all_methods)))
ax3.set_yticklabels(all_methods, fontsize=9)
ax3.set_xlabel("F1 Score (%)", fontweight="bold")
ax3.set_title("F1 Score (Harmonic Mean)", fontweight="bold", fontsize=12)
ax3.set_xlim(0, 100)
ax3.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, f1_scores)):
    ax3.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=8)

# ============================================================================
# Chart 4: Recall vs Precision Scatter
# ============================================================================
ax4 = plt.subplot(3, 2, 4)

for i, method in enumerate(all_methods):
    r = results_data[method]["recall"]
    p = results_data[method]["precision"]
    color = colors[i]
    marker = (
        "o"
        if "Regex" in method and "Hybrid" not in method
        else ("s" if "Hybrid" in method else "^")
    )
    ax4.scatter(
        r,
        p,
        s=150,
        c=[color],
        marker=marker,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
    )

    # Add labels for key methods
    if "v5" in method or "Hybrid" in method or "LLM-Only" in method:
        ax4.annotate(
            method,
            (r, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

ax4.set_xlabel("Recall (%)", fontweight="bold")
ax4.set_ylabel("Precision (%)", fontweight="bold")
ax4.set_title("Recall vs Precision Trade-off", fontweight="bold", fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 105)
ax4.set_ylim(0, 105)

# Add diagonal F1 lines
for f1 in [50, 60, 70]:
    x = np.linspace(1, 100, 100)
    y = (f1 * x) / (2 * x - f1)
    y = np.clip(y, 0, 100)
    ax4.plot(x, y, "--", alpha=0.2, color="gray", linewidth=1)
    ax4.text(95, (f1 * 95) / (2 * 95 - f1), f"F1={f1}", fontsize=7, alpha=0.5)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#1f77b4", alpha=0.7, label="Regex Methods"),
    Patch(facecolor="#ff7f0e", alpha=0.7, label="Hybrid Methods"),
    Patch(facecolor="#2ca02c", alpha=0.7, label="LLM-Only Methods"),
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=8)

# ============================================================================
# Chart 5: Perfect Recall Articles
# ============================================================================
ax5 = plt.subplot(3, 2, 5)
perfect_recalls = [results_data[m]["perfect_recall"] for m in all_methods]
total = results_data[all_methods[0]]["total_articles"]

bars = ax5.barh(
    range(len(all_methods)), perfect_recalls, color=colors, alpha=0.7, edgecolor="black"
)
ax5.set_yticks(range(len(all_methods)))
ax5.set_yticklabels(all_methods, fontsize=9)
ax5.set_xlabel("Articles with 100% Recall", fontweight="bold")
ax5.set_title(
    f"Perfect Recall Articles (out of {total})", fontweight="bold", fontsize=12
)
ax5.set_xlim(0, total + 2)
ax5.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, perfect_recalls)):
    pct = val / total * 100
    ax5.text(val + 0.3, i, f"{val}/{total} ({pct:.0f}%)", va="center", fontsize=8)

# ============================================================================
# Chart 6: Radar Chart - Multi-dimensional Comparison
# ============================================================================
ax6 = plt.subplot(3, 2, 6, projection="polar")

# Select key methods for radar chart
radar_methods = []
for method in all_methods:
    if any(x in method for x in ["v1", "v5", "Hybrid", "Opus"]):
        radar_methods.append(method)

categories = ["Recall", "Precision", "F1 Score", "Perfect\nRecall %"]
num_vars = len(categories)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot for each method
for method in radar_methods:
    metrics = results_data[method]
    values = [
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        metrics["perfect_recall"] / metrics["total_articles"] * 100,
    ]
    values += values[:1]  # Complete the circle

    if "v1" in method:
        color = "#1f77b4"
        label = "V1 (Baseline)"
    elif "v5" in method:
        color = "#d62728"
        label = "V5 (Best Regex)"
    elif "Hybrid" in method:
        color = "#ff7f0e"
        label = "Term Norm Hybrid"
    elif "Opus" in method:
        color = "#2ca02c"
        label = "LLM-Only (Opus)"
    else:
        continue

    ax6.plot(angles, values, "o-", linewidth=2, label=label, color=color, alpha=0.7)
    ax6.fill(angles, values, alpha=0.15, color=color)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=9)
ax6.set_ylim(0, 100)
ax6.set_title("Multi-dimensional Comparison", fontweight="bold", fontsize=12, y=1.08)
ax6.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "comparison_charts.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✓ Charts saved to: {output_path}")

# ============================================================================
# Create additional detailed comparison table
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED COMPARISON TABLE")
print("=" * 80)
print(f"{'Method':<30} {'Recall':>8} {'Precision':>10} {'F1':>6} {'Perfect':>8}")
print("-" * 80)

for method in all_methods:
    m = results_data[method]
    print(
        f"{method:<30} {m['recall']:>7.1f}% {m['precision']:>9.1f}% {m['f1']:>5.1f}% {m['perfect_recall']:>4}/{m['total_articles']:<2}"
    )

print("=" * 80)

# ============================================================================
# Calculate improvements
# ============================================================================
if "Regex v1" in results_data and "Regex v5" in results_data:
    v1_recall = results_data["Regex v1"]["recall"]
    v5_recall = results_data["Regex v5"]["recall"]
    improvement = v5_recall - v1_recall

    print("\n📈 Regex Evolution (v1 → v5):")
    print(
        f"   Recall improvement: +{improvement:.1f} percentage points ({v1_recall:.1f}% → {v5_recall:.1f}%)"
    )

    if "Term Norm Hybrid" in results_data:
        hybrid_recall = results_data["Term Norm Hybrid"]["recall"]
        hybrid_improvement = hybrid_recall - v5_recall
        print(
            f"   Term Norm additional gain: {hybrid_improvement:+.1f} percentage points"
        )

print("\n✨ Analysis complete!")

plt.show()
