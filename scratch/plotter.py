"""
Line graph and bar chart plotters.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_line_graph(
    values: list[float],
    labels: list[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_deltas: bool = True,
    show_total_improvement: bool = True,
    value_format: str = "{:.1f}%",
    delta_format: str = "+{:.1f}%",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | None = None,
):
    """
    Create a line graph with filled area below, value labels, and delta annotations.

    Args:
        values: List of numerical values to plot
        labels: List of x-axis labels (can include newlines for multi-line labels)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_deltas: Whether to show green delta labels between points
        show_total_improvement: Whether to show total improvement annotation box
        value_format: Format string for value labels (default: "{:.1f}%")
        delta_format: Format string for delta labels (default: "+{:.1f}%")
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(values))

    # Set background color
    ax.set_facecolor("#E8E8E8")
    fig.patch.set_facecolor("white")

    # Plot filled area under the line
    ax.fill_between(x, values, alpha=0.3, color="#5B9BD5")

    # Plot the line
    ax.plot(x, values, color="#5B9BD5", linewidth=2, marker="o", markersize=8, markerfacecolor="#5B9BD5", markeredgecolor="white", markeredgewidth=2)

    # Add value labels above each point
    for i, (xi, yi) in enumerate(zip(x, values)):
        ax.annotate(
            value_format.format(yi),
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=15,
            fontweight="bold",
        )

    # Add delta labels between points
    if show_deltas and len(values) > 1:
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            if delta > 0:
                mid_x = (x[i - 1] + x[i]) / 2
                mid_y = (values[i - 1] + values[i]) / 2
                ax.annotate(
                    delta_format.format(delta),
                    (mid_x, mid_y),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha="center",
                    fontsize=15,
                    color="#2E7D32",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
                )

    # Add total improvement annotation box
    if show_total_improvement and len(values) > 1:
        total_improvement = values[-1] - values[0]
        annotation_text = f"Total v1→v{len(values)}\nImprovement:\n{delta_format.format(total_improvement)}"
        ax.annotate(
            annotation_text,
            (0.5, values[1] - 10),
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#E8F5E9",
                edgecolor="#81C784",
                linewidth=2,
            ),
        )

    # Configure axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, ha="center")
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)

    # Set y-axis limits with some padding
    y_min = 0
    y_max = max(values) * 1.15
    ax.set_ylim(y_min, y_max)

    # Add horizontal dashed line at 100 if applicable
    if max(values) <= 100:
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(y_min, 105)

    # Grid styling
    ax.yaxis.grid(True, color="white", linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # Title
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_horizontal_bar(
    values: list[float],
    labels: list[str],
    total: int | None = None,
    title: str = "",
    xlabel: str = "",
    bar_color: str = "#8BC48A",
    figsize: tuple[int, int] = (10, 4),
    show_fraction: bool = True,
    show_percent: bool = True,
    save_path: str | None = None,
):
    """
    Create a horizontal bar chart with value labels.

    Args:
        values: List of numerical values for each bar
        labels: List of labels for each bar (displayed on left)
        total: Total for fraction display (e.g., 32 for "17/32"). If None, fractions not shown.
        title: Chart title
        xlabel: X-axis label
        bar_color: Color of the bars (default: soft green)
        figsize: Figure size as (width, height)
        show_fraction: Whether to show "X/Y" format (requires total)
        show_percent: Whether to show percentage
        save_path: Path to save the figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(values))
    bar_height = 0.5

    # Plot horizontal bars - no edge, soft green color
    bars = ax.barh(y, values, height=bar_height, color=bar_color, edgecolor="none")

    # Add value labels at end of each bar
    for bar, val in zip(bars, values):
        label_parts = []
        if show_fraction and total is not None:
            label_parts.append(f"{int(val)}/{total}")
        if show_percent:
            if total is not None:
                pct = (val / total) * 100
            else:
                pct = val
            label_parts.append(f"({pct:.0f}%)")

        label_text = " ".join(label_parts)
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=11,
            color="#555555",
        )

    # Configure axes
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # Top-to-bottom order

    # Set x limits with padding for labels
    max_val = max(values)
    ax.set_xlim(0, max_val * 1.5)

    # No grid lines
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # Light gray border around the plot area
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("#E0E0E0")
        spine.set_linewidth(1)

    # Hide x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xlabel(xlabel, fontsize=11)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # White background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Example data matching the image
    values = [35.0, 55.0, 77.0, 88.0, 70.0, 85.0]
    labels = [
        "GPT 4o",
        "Claude Sonnet 4",
        "GPT 5 Base",
        "GPT 5\nMulti-Ask\nMerge",
        "GPT 5\nClaude Opus 4.5\nMerge",
        "GPT 5\nMulti-Ask\nMerge\n(Prompt v13)",
    ]

    plot_line_graph(
        values=values,
        labels=labels,
        title="Sentence Similarity Eval",
        ylabel="Accuracy (%)",
        save_path="sentence_similarity_eval.png",
    )

    # Example horizontal bar chart
    bar_values = [17, 10, 8]
    bar_labels = [
        "LLM-Only (Opus 4.5)",
        "LLM-Only (Sonnet 4.5)",
        "LLM-Only (GPT-4o)",
    ]

    plot_horizontal_bar(
        values=bar_values,
        labels=bar_labels,
        total=32,
        title="Perfect Recall Articles (out of 32)",
        save_path="perfect_recall_articles.png",
    )
