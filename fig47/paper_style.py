"""Drawing primitives matching the paper's exact visual grammar: cream
"stacked" supernode boxes, maroon arrows, orange -Nx inhibition badges,
faded/greyed suppressed paths. Used by both per-algorithm Figure 4 and
Figure 7 renderers so the two match each other and the reference.
"""
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

CREAM = '#e9e5d6'
CREAM_EDGE = '#8a8571'
MAROON = '#6b3410'
ORANGE = '#e0801f'
FADE_FILL = '#f0eee6'
FADE_EDGE = '#c9c6ba'
FADE_ARROW = '#d4cfc0'
TARGET_FILL = '#fdf6e3'
TARGET_EDGE = '#b8860b'


def stacked_box(ax, x, y, w, h, text, faded=False, n_stack=2, fontsize=8, target=False, bold=False):
    fill = FADE_FILL if faded else (TARGET_FILL if target else CREAM)
    edge = FADE_EDGE if faded else (TARGET_EDGE if target else CREAM_EDGE)
    tcolor = '#aaaaaa' if faded else '#1a1a1a'
    if not faded:
        for i in range(n_stack, 0, -1):
            off = i * 0.035
            ax.add_patch(mpatches.FancyBboxPatch((x + off, y - off), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                        fc=fill, ec=edge, linewidth=0.7, alpha=0.55, zorder=2))
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                fc=fill, ec=edge, linewidth=1.1, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
           color=tcolor, weight='bold' if bold else 'normal', zorder=4, wrap=True)


def paper_arrow(ax, start, end, pct_label=None, faded=False, lw=2.2, rad=0.15):
    color = FADE_ARROW if faded else MAROON
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12, color=color,
                            linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=1, alpha=0.9 if not faded else 0.6)
    ax.add_patch(arrow)
    if pct_label is not None:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + rad
        ax.text(mx, my, pct_label, fontsize=7, color='#888' if faded else '#333', ha='center', zorder=5)


def badge(ax, x, y, text='-2x', color=ORANGE):
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.28, y - 0.13), 0.56, 0.26, boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=color, ec='none', zorder=6))
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5, color='white', weight='bold', zorder=7)


def draw_mini_graph(ax, x0, y0, w, h, n, adj, source, highlight_node=None, node_labels=True):
    """Small circular node-link rendering of the actual sampled graph for
    this example -- so the reader can see exactly what input the diagram is
    explaining, not just an abstract box-and-arrow schematic."""
    import math as _math
    cx, cy, r = x0 + w / 2, y0 + h / 2, min(w, h) * 0.42
    pos = [(cx + r * _math.cos(-_math.pi / 2 + 2 * _math.pi * i / n),
           cy + r * _math.sin(-_math.pi / 2 + 2 * _math.pi * i / n)) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] > 0:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color='#ccc', linewidth=0.7, zorder=1)
    node_r = r * 0.15
    for i in range(n):
        if i == highlight_node:
            fc, ec, lw = '#a78bfa', '#5b21b6', 1.4
        elif i == source:
            fc, ec, lw = '#f59e0b', '#92400e', 1.0
        else:
            fc, ec, lw = '#e5e7eb', '#555', 0.6
        ax.add_patch(mpatches.Circle(pos[i], node_r, fc=fc, ec=ec, linewidth=lw, zorder=2))
        if node_labels and n <= 20:
            ax.text(pos[i][0], pos[i][1], str(i), fontsize=node_r * 55, ha='center', va='center', zorder=3)
