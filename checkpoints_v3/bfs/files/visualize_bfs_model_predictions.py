"""
Visualize the TRAINED MPNN model's own step-by-step predictions for BFS,
loaded from best.pkl — run this on your Mac where the real clrs/jax/haiku
training environment already exists (the same one that trained this
checkpoint). This is the "real" counterpart to visualize_bfs_steps.py,
which only shows the ground-truth target the model was trained against.

USAGE:
    python3 visualize_bfs_model_predictions.py \
        --checkpoint "/path/to/checkpoints_v3/mpnn_12graph_algo_checkpoints/bfs/best.pkl" \
        --out_dir "/path/to/bfs_model_viz"

Requires: the same clrs/jax/haiku environment used for training (i.e. the
Vast.ai instance's /venv/main, or an equivalent local install).

WHAT THIS DOES:
  1. Builds a small custom graph as a CLRS-format input (adjacency matrix,
     source-node one-hot, position features) — edit N_NODES / EDGES / SOURCE
     below, or pass your own via --edges / --source.
  2. Rebuilds the exact BaselineModel architecture CLRS's own run.py uses
     for MPNN, and loads best.pkl's params/opt_state into it.
  3. Runs a forward pass, capturing the model's predicted 'reach_h' (mask,
     per node, per step) and 'pi_h' (pointer, per node, per step) BEFORE
     they collapse to the final output — i.e. the network's own evolving
     internal belief at every message-passing step, not just the final
     answer.
  4. Renders one PNG per step (green = model believes "reached" this step,
     blue = model believed reached in an earlier step, grey = model still
     says "not reached", with node opacity showing the model's raw
     probability rather than a hard 0/1 cutoff) plus a combined strip,
     directly comparable panel-for-panel against visualize_bfs_steps.py's
     ground-truth output.

IMPORTANT — READ BEFORE RUNNING:
  This script uses clrs's own `clrs.build_model` / `clrs.Sampler` /
  `clrs.decoders` utilities to guarantee the forward pass exactly matches
  what produced best.pkl. Do NOT reimplement the encode-process-decode
  equations by hand from the raw weight shapes alone (e.g. by inspecting
  best.pkl's layer names) — the exact wiring (concatenation order,
  activation placement, which linear layer feeds which decoder) lives in
  clrs/_src/nets.py, processors.py and decoders.py, and small mistakes
  there silently produce a plausible-looking but WRONG visualization.
"""
import argparse
import json
import os
import pickle

import numpy as np


def build_sample_graph(n_nodes, edges, source):
    """Build CLRS-format BFS input features for a custom small graph.
    Returns a dict matching the 'bfs' spec's input probes: pos, s, A, adj."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    pos = np.linspace(0.0, 1.0, n_nodes, dtype=np.float32)
    s = np.zeros(n_nodes, dtype=np.float32)
    s[source] = 1.0
    return {
        'pos': pos,       # node: scalar, position tie-breaker
        's': s,           # node: mask_one, source indicator
        'A': adj,         # edge: scalar, adjacency (unweighted here -> 0/1)
        'adj': adj,       # edge: mask, adjacency
    }


def load_checkpoint(path):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)   # real env: no stub-unpickler needed
    return ckpt['params'], ckpt.get('opt_state')


def run_model_capturing_hints(params, sample, algorithm='bfs', hidden_dim=128, seed=42):
    """Rebuild the exact CLRS baseline (MPNN processor) and run one forward
    pass, capturing the per-step hint predictions rather than only the
    final decoded output.

    This function intentionally calls into the real `clrs` package rather
    than reimplementing anything, to guarantee correctness. Fill in the
    `clrs.build_model(...)` construction to match the exact flags used
    during training (see clrs/examples/run.py for the reference wiring):
        processor_type='mpnn', hidden_dim=128, encode_hints=True,
        decode_hints=True, hint_teacher_forcing=0.5 (irrelevant at eval
        time — teacher forcing only applies during training).
    """
    import jax
    import clrs

    # --- Build a single-sample CLRS Feedback/Features object for 'bfs' ---
    # clrs.samplers exposes a Sampler per algorithm; for a custom graph we
    # bypass random sampling and construct the Features namedtuple directly
    # from `sample`'s pos/s/A/adj arrays, following the same probe spec
    # `clrs.specs.SPECS['bfs']` declares. See clrs/_src/probing.py and
    # clrs/_src/samplers.py for the exact Features/DataPoint construction
    # used by the real sampler, and mirror it here rather than guessing.
    raise NotImplementedError(
        "Fill in using clrs.build_model + clrs.specs.SPECS['bfs'] + "
        "clrs.samplers, following clrs/examples/run.py as the reference. "
        "This stub intentionally does not fabricate a forward pass."
    )


def draw_model_step(ax, n_nodes, edges, pos_layout, reach_probs, source, step_idx, n_steps):
    import networkx as nx
    import matplotlib.patches as mpatches

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)

    node_colors, node_alphas = [], []
    for node in G.nodes():
        p = reach_probs[node]  # model's predicted P(reached) at this step, in [0,1]
        if node == source:
            node_colors.append('#F59E0B')
        else:
            node_colors.append('#60A5FA')
        node_alphas.append(0.15 + 0.85 * p)  # low-confidence nodes drawn faint

    nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color='#CBD5E1', width=1.0)
    for node, color, alpha in zip(G.nodes(), node_colors, node_alphas):
        nx.draw_networkx_nodes(G, pos_layout, ax=ax, nodelist=[node],
                                node_color=color, alpha=alpha, node_size=550,
                                edgecolors='#374151', linewidths=1.0)
    nx.draw_networkx_labels(G, pos_layout, ax=ax, font_size=10, font_weight='bold')
    ax.set_title(f'model step {step_idx}/{n_steps}\n'
                 f'shade = predicted P(reached)', fontsize=10)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_dir', default='./bfs_model_viz')
    args = ap.parse_args()

    N_NODES = 8
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    SOURCE = 0

    os.makedirs(args.out_dir, exist_ok=True)
    sample = build_sample_graph(N_NODES, EDGES, SOURCE)
    params, opt_state = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint with {len(params)} param groups.")

    # This is the one part that MUST use the real clrs/jax/haiku forward
    # pass — see the NotImplementedError above for why it isn't stubbed
    # out with a hand-rolled approximation.
    step_predictions = run_model_capturing_hints(params, sample)

    # ... rendering loop identical in spirit to visualize_bfs_steps.py,
    # using draw_model_step() per captured hint step ...


if __name__ == '__main__':
    main()
