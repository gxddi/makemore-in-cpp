#!/usr/bin/env python3
"""Render the final makemore README banner."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUTPUT = Path(__file__).with_name("makemore.png")
BACKGROUND = "#faf7ef"
INK = "#292524"

# Context length 5 × embedding dimension 10, hidden width 200, vocabulary 27.
LAYER_COUNTS = (50, 200, 27)


def neuron_positions(x, count):
    """Preserve equal spacing, including neurons outside the crop."""
    return [(x, (i - (count - 1) / 2) * 0.045) for i in range(count)]


def main():
    fig, ax = plt.subplots(figsize=(16, 4), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    layers = [neuron_positions(x, count)
              for x, count in zip((0, 4, 8), LAYER_COUNTS)]

    for source, destination in zip(layers, layers[1:]):
        ax.add_collection(LineCollection(
            [(start, end) for start in source for end in destination],
            colors="#57534e", linewidths=0.18, alpha=0.025, zorder=1,
        ))

    for positions, label, alignment in zip(
        layers, ("c", "l1", "l2"), ("left", "center", "right")
    ):
        xs, ys = zip(*positions)
        ax.scatter(xs, ys, s=36, color=INK, edgecolor=BACKGROUND,
                   linewidth=0.25, zorder=2)
        ax.text(positions[0][0], 0, label, ha=alignment, va="center",
                fontsize=28, color=INK, zorder=4,
                bbox=dict(facecolor=BACKGROUND, edgecolor=INK,
                          linewidth=1.2, pad=5), clip_on=True)

    # Center the title horizontally, above the middle layer label.
    ax.text(4, 0.73, "makemore", ha="center", va="center",
            fontsize=42, fontweight="bold", color=INK, zorder=5,
            bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=6))

    # The hidden layer extends beyond this crop; retain small outer margins.
    ax.set_xlim(-0.12, 8.12)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.025, top=0.975)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(OUTPUT, dpi=220, facecolor=BACKGROUND)
    plt.close(fig)


if __name__ == "__main__":
    main()
