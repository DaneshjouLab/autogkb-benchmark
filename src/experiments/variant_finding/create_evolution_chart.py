"""
Create a focused evolution chart showing the progression of regex methods.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 8)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_metrics(results):
    return {
        "recall": results.get("avg_recall", 0) * 100,
        "precision": results.get("avg_precision", 0) * 100,
        "perfect_recall": sum(
            1
            for r in results.get("per_article_results", [])
            if r.get("recall", 0) == 1.0
        ),
        "total_articles": len(results.get("per_article_results", [])),
    }


# Load regex evolution data
base_dir = Path(__file__).parent
versions = []
improvements = []

for v in range(1, 6):
    path = base_dir / "regex_variants" / f"results_v{v}.json"
    if path.exists():
        data = extract_metrics(load_results(path))
        versions.append(
            {
                "version": f"v{v}",
                "name": f"Version {v}",
                "recall": data["recall"],
                "precision": data["precision"],
                "perfect_recall": data["perfect_recall"],
                "total": data["total_articles"],
            }
        )

# Add term norm
term_norm_path = base_dir / "regex_term_norm" / "results_term_norm.json"
if term_norm_path.exists():
    data = extract_metrics(load_results(term_norm_path))
    versions.append(
        {
            "version": "term_norm",
            "name": "Term Norm\nHybrid",
            "recall": data["recall"],
            "precision": data["precision"],
            "perfect_recall": data["perfect_recall"],
            "total": data["total_articles"],
        }
    )

# Version labels with features
version_features = [
    "v1\nBaseline\n(Methods &\nConclusions)",
    "v2\nFull Text +\nFormat Norm",
    "v3\nContext-Aware\nStar Alleles",
    "v4\nSNP\nExpansion",
    "v5\nBioC\nSupplements",
    "Term Norm\nPost-Extraction\nValidation",
]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

# ============================================================================
# Top plot: Recall progression
# ============================================================================
x = np.arange(len(versions))
recalls = [v["recall"] for v in versions]

# Plot line with markers
line = ax1.plot(
    x,
    recalls,
    "o-",
    linewidth=3,
    markersize=12,
    color="#2E86AB",
    label="Recall",
    markeredgecolor="white",
    markeredgewidth=2,
)

# Shade area under curve
ax1.fill_between(x, 0, recalls, alpha=0.2, color="#2E86AB")

# Add value labels above each point
for i, (xi, recall) in enumerate(zip(x, recalls)):
    ax1.text(
        xi, recall + 2, f"{recall:.1f}%", ha="center", fontsize=11, fontweight="bold"
    )

    # Add improvement annotation
    if i > 0:
        prev_recall = recalls[i - 1]
        improvement = recall - prev_recall
        if improvement > 0.1:  # Only show if improvement > 0.1%
            mid_x = (x[i - 1] + x[i]) / 2
            mid_y = (recalls[i - 1] + recalls[i]) / 2
            ax1.annotate(
                f"+{improvement:.1f}%",
                xy=(mid_x, mid_y),
                fontsize=9,
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

# Styling
ax1.set_xticks(x)
ax1.set_xticklabels(version_features, fontsize=10)
ax1.set_ylabel("Recall (%)", fontsize=13, fontweight="bold")
ax1.set_title(
    "Regex Evolution: Progressive Improvement in Variant Extraction",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3, axis="y")

# Add horizontal line at 100%
ax1.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax1.text(len(versions) - 0.5, 100, "Perfect", fontsize=9, va="bottom", color="gray")

# Add annotation for total improvement
total_improvement = recalls[-2] - recalls[0]  # v5 vs v1 (excluding term norm)
ax1.text(
    0.5,
    50,
    f"Total v1→v5\nImprovement:\n+{total_improvement:.1f}%",
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.8", facecolor="yellow", alpha=0.3),
    ha="center",
)

# ============================================================================
# Bottom plot: Perfect recall articles
# ============================================================================
perfect_recalls = [v["perfect_recall"] for v in versions]
total = versions[0]["total"]

bars = ax2.bar(
    x,
    perfect_recalls,
    color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"],
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, perfect_recalls)):
    pct = val / total * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.5,
        f"{val}/{total}\n({pct:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Styling
ax2.set_xticks(x)
ax2.set_xticklabels(version_features, fontsize=10)
ax2.set_ylabel("Perfect Recall Articles", fontsize=13, fontweight="bold")
ax2.set_title("Articles with 100% Recall", fontsize=14, fontweight="bold", pad=10)
ax2.set_ylim(0, total + 3)
ax2.grid(True, alpha=0.3, axis="y")

# Add horizontal line at total
ax2.axhline(y=total, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax2.text(
    len(versions) - 0.5, total, f"All {total}", fontsize=9, va="bottom", color="gray"
)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "evolution_chart.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✓ Evolution chart saved to: {output_path}")

# Print summary
print("\n" + "=" * 70)
print("REGEX EVOLUTION SUMMARY")
print("=" * 70)
for i, v in enumerate(versions):
    improvement = ""
    if i > 0:
        diff = v["recall"] - versions[i - 1]["recall"]
        improvement = f" ({diff:+.1f}%)"
    print(
        f"{v['name']:20} {v['recall']:6.1f}% recall  {v['perfect_recall']:2}/{v['total']} perfect{improvement}"
    )
print("=" * 70)
print(f"Total improvement (v1→v5): +{recalls[-2] - recalls[0]:.1f} percentage points")
print("=" * 70)

plt.show()
