"""
Visualize the internal (hint) steps of BFS on a sample graph — the
ground-truth trajectory the CLRS-30 MPNN model was trained to predict.

This mirrors the paper's own Figure 2 style (insertion sort hint sequence):
one panel per algorithm "step" (here: one panel per BFS frontier level),
showing exactly which nodes are known/unknown at that point and which
edges were just used to discover new nodes.

NOTE: this shows the GROUND-TRUTH hint trajectory (what the model was
trained to reproduce), not the trained network's own predictions. See
the companion script `visualize_bfs_model_predictions.py` (run.py-style,
for your local machine with the real clrs/jax/haiku env) to see what the
actual trained best.pkl predicts at each step, which can then be compared
directly against these ground-truth panels.
"""
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os


def bfs_with_levels(edges, n_nodes, source):
    """Run BFS, recording the frontier discovered at each level (=hint step),
    and the predecessor of every newly-discovered node. Mirrors the parallel,
    level-synchronous structure CLRS-30 uses for its BFS hint trajectory."""
    adj = {i: set() for i in range(n_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = {source}
    pred = {source: source}  # convention: source points to itself
    levels = [{'newly_reached': {source}, 'frontier_edges': [], 'visited_after': set(visited)}]

    frontier = {source}
    while frontier:
        next_frontier = set()
        frontier_edges = []
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    next_frontier.add(v)
                    pred[v] = u
        for v in next_frontier:
            visited.add(v)
            frontier_edges.append((pred[v], v))
        if not next_frontier:
            break
        levels.append({
            'newly_reached': next_frontier,
            'frontier_edges': frontier_edges,
            'visited_after': set(visited),
        })
        frontier = next_frontier

    unreached = set(range(n_nodes)) - visited
    return levels, pred, unreached


def draw_step(ax, G, pos, level_info, step_idx, n_steps, source, unreached_final):
    visited_after = level_info['visited_after']
    newly = level_info['newly_reached']
    frontier_edges = set(level_info['frontier_edges']) | set((b, a) for a, b in level_info['frontier_edges'])

    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('#F59E0B')       # source: amber
        elif node in newly:
            node_colors.append('#22C55E')       # newly reached this step: green
        elif node in visited_after:
            node_colors.append('#60A5FA')       # already known: blue
        else:
            node_colors.append('#D1D5DB')       # unreached so far: grey

    edge_colors, edge_widths = [], []
    for e in G.edges():
        if e in frontier_edges:
            edge_colors.append('#16A34A')
            edge_widths.append(3.0)
        else:
            edge_colors.append('#CBD5E1')
            edge_widths.append(1.0)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=550,
                            edgecolors='#374151', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    ax.set_title(f'step {step_idx} / {n_steps}\n'
                 f'reach_h: {len(visited_after)}/{len(G.nodes())} nodes reached',
                 fontsize=10)
    ax.axis('off')


def visualize(edges, n_nodes, source, out_dir, weights=None):
    """
    edges: list of (u, v) tuples (0-indexed node ids)
    n_nodes: total node count
    source: BFS source node id
    weights: optional dict {(u,v): w} — BFS itself ignores weights (unweighted
             traversal), but they are drawn as edge labels for reference since
             the user may reuse this same sample graph for a weighted algorithm
             (Dijkstra/Bellman-Ford) later.
    """
    os.makedirs(out_dir, exist_ok=True)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=7)

    levels, pred, unreached = bfs_with_levels(edges, n_nodes, source)
    n_steps = len(levels) - 1

    # ---- one PNG per step ----
    step_files = []
    for i, level_info in enumerate(levels):
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_step(ax, G, pos, level_info, i, n_steps, source, unreached)
        if weights:
            nx.draw_networkx_edge_labels(
                G, pos, ax=ax,
                edge_labels={e: weights.get(e, weights.get((e[1], e[0]), '')) for e in G.edges()},
                font_size=7, font_color='#6B7280')
        legend_handles = [
            mpatches.Patch(color='#F59E0B', label='source'),
            mpatches.Patch(color='#22C55E', label='newly reached (this step)'),
            mpatches.Patch(color='#60A5FA', label='already reached'),
            mpatches.Patch(color='#D1D5DB', label='not yet reached'),
        ]
        ax.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7, frameon=False)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'bfs_step_{i:02d}.png')
        plt.savefig(fname, dpi=150, facecolor='white')
        plt.close(fig)
        step_files.append(fname)

    # ---- combined strip, paper-Figure-2 style ----
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4.2))
    if len(levels) == 1:
        axes = [axes]
    for i, (ax, level_info) in enumerate(zip(axes, levels)):
        draw_step(ax, G, pos, level_info, i, n_steps, source, unreached)
    fig.suptitle('BFS ground-truth hint trajectory (reach_h per level)\n'
                 f'source = node {source}  ·  {len(unreached)} node(s) unreachable',
                 fontsize=12, y=1.05)
    legend_handles = [
        mpatches.Patch(color='#F59E0B', label='source'),
        mpatches.Patch(color='#22C55E', label='newly reached (this step)'),
        mpatches.Patch(color='#60A5FA', label='already reached'),
        mpatches.Patch(color='#D1D5DB', label='not yet reached'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    strip_file = os.path.join(out_dir, 'bfs_all_steps_strip.png')
    plt.savefig(strip_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ---- dump the raw hint trajectory as JSON too, for reference/comparison ----
    traj = []
    for i, level_info in enumerate(levels):
        traj.append({
            'step': i,
            'newly_reached': sorted(level_info['newly_reached']),
            'frontier_edges_used': level_info['frontier_edges'],
            'reach_h_after_step': sorted(level_info['visited_after']),
        })
    with open(os.path.join(out_dir, 'bfs_hint_trajectory.json'), 'w') as f:
        json.dump({
            'source': source,
            'n_nodes': n_nodes,
            'edges': edges,
            'final_predecessors_pi': {str(k): v for k, v in pred.items()},
            'unreachable_nodes': sorted(unreached),
            'trajectory': traj,
        }, f, indent=2)

    return step_files, strip_file


if __name__ == '__main__':
    # ── SAMPLE GRAPH — edit this block to try your own data ──────────────
    # 8-node graph, unweighted edges (BFS ignores weights; included here only
    # as edge labels for reference / reuse with a weighted algorithm later)
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    N_NODES = 8
    SOURCE = 0
    WEIGHTS = {(0, 1): 4, (0, 2): 1, (1, 3): 2, (2, 3): 5, (2, 4): 8,
               (3, 5): 3, (4, 5): 2, (5, 6): 6, (6, 7): 1}

    step_files, strip_file = visualize(EDGES, N_NODES, SOURCE, '/home/claude/bfs_viz/out', WEIGHTS)
    print(f"Wrote {len(step_files)} per-step PNGs and 1 combined strip:")
    for f in step_files:
        print(" ", f)
    print(" ", strip_file)
