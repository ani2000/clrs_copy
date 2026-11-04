"""Figure 7 reproduction, paper-faithful style, one PNG per algorithm.
The paper's Figure 7 is a grid of GRAPH & INTERVENTIONS panels, each
showing ONE feature-group ablation on the SAME example (not an aggregate).
We reproduce that exactly: one panel per real supernode x clamp_step
intervention (up to 6, matching the paper's own 3x2 grid), each with a
mini attribution diagram (clamped node greyed + -2x badge), the example
graph in place of "PROMPT", and the real before/after target value in
place of "TOP OUTPUTS". All from single_example_<algo>.json.
"""
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from paper_style import stacked_box, paper_arrow, badge, draw_mini_graph

ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim',
        'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges',
        'strongly_connected_components', 'floyd_warshall']


def to_prob(logit, ttype):
    return None if ttype == 'scalar' else 1 / (1 + math.exp(-logit))


def ranked_outputs(ttype, logit=None, class_logits=None, raw_value=None):
    """Paper-style ranked top-outputs list: [(label, pct_or_None), ...] sorted
    by probability descending. Scalar targets have no probability distribution
    -- returns [('value', raw_value)] instead, rendered as a plain number."""
    if ttype == 'scalar':
        return [('value', raw_value)]
    if ttype == 'categorical':
        exp = [math.exp(l) for l in class_logits]
        Z = sum(exp)
        rows = [(f"class {i}", e / Z * 100) for i, e in enumerate(exp)]
        return sorted(rows, key=lambda r: -r[1])
    p = 1 / (1 + math.exp(-logit))
    rows = [("= 1 (yes)", p * 100), ("= 0 (no)", (1 - p) * 100)]
    return sorted(rows, key=lambda r: -r[1])



def render_algo(algo):
    d = json.load(open(f'single_example_{algo}.json'))
    ttype = d['target_type']
    ivs = d['interventions']
    n_panels = min(6, len(ivs)) if ivs else 0

    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"{algo}\n\nno downstream interventions\n(no clean baseline -- Phase 0)",
               ha='center', va='center', fontsize=11, color='gray')
        ax.axis('off')
        plt.savefig(f'fig7_{algo}.png', dpi=140, bbox_inches='tight')
        plt.close()
        print(f'saved fig7_{algo}.png (empty, no clean baseline)')
        return

    ncols = 3
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = axes.flatten() if n_panels > 1 else [axes]
    fig.suptitle(f"Figure 7 — {algo}: interventions testing the '{d['target']}' prediction\n"
                f"(node activations / probabilities measured relative to baseline)", fontsize=12, y=1.0)

    base_p = to_prob(d['baseline_logit'], ttype)

    for ax, iv in zip(axes, ivs[:n_panels]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch((0.15, 0.15), 9.7, 9.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                    fc='#f7f5ee', ec='#ccc', linewidth=1, zorder=0))
        ax.text(0.5, 9.3, "GRAPH & INTERVENTIONS", fontsize=7.5, color='#888', weight='bold')

        # mini diagram: clamped supernode -> target
        stacked_box(ax, 6.4, 6.9, 2.9, 1.0, d['target'][:16], target=True, fontsize=7, bold=True)
        stacked_box(ax, 0.6, 6.9, 3.0, 1.0, iv['supernode'], faded=True, fontsize=7.5)
        badge(ax, 0.35, 8.1, text='-2x')
        paper_arrow(ax, (3.6, 7.4), (6.4, 7.4), faded=True)
        new_p = to_prob(iv['new_logit'], ttype)
        pct = f"{new_p*100:.0f}%" if new_p is not None else f"{iv['new_logit']:+.2f}"
        ax.text(9.3, 8.1, pct, fontsize=9, weight='bold', color='#333', ha='right')

        # "PROMPT" equivalent: the real example graph
        ax.text(0.5, 5.9, "EXAMPLE GRAPH", fontsize=6.5, color='#888', weight='bold')
        draw_mini_graph(ax, 0.5, 3.0, 3.0, 2.7, d['n'], d['adj'], d['source'], highlight_node=d['node'])
        ax.text(0.5, 2.7, f"n={d['n']}, node={d['node']}, step={d['step']}", fontsize=6, color='#666')

        # "TOP OUTPUTS" equivalent: paper-style two-column ranked display,
        # left = pre-intervention, right = post-intervention.
        base_rows = ranked_outputs(ttype, logit=d['baseline_logit'], class_logits=d.get('baseline_class_logits'),
                                   raw_value=d['baseline_logit'])
        new_rows = ranked_outputs(ttype, logit=iv['new_logit'], class_logits=iv.get('new_class_logits'),
                                  raw_value=iv['new_logit'])
        ax.text(4.2, 5.9, "TOP OUTPUTS", fontsize=6.5, color='#888', weight='bold')
        ax.text(4.2, 5.35, "before", fontsize=6, color='#999', style='italic')
        ax.text(7.0, 5.35, "after clamp", fontsize=6, color='#999', style='italic')
        row_y = 4.75
        for (blabel, bval), (nlabel, nval) in zip(base_rows[:3], new_rows[:3]):
            btxt = f"{blabel}: {bval:.2f}" if ttype == 'scalar' else f"{blabel}  {bval:.0f}%"
            ntxt = f"{nlabel}: {nval:.2f}" if ttype == 'scalar' else f"{nlabel}  {nval:.0f}%"
            ax.text(4.2, row_y, btxt, fontsize=7.5, color='#333')
            ax.text(7.0, row_y, ntxt, fontsize=7.5, color='#333', weight='bold')
            row_y -= 0.55
        step_kind = 'same-step (trivial)' if iv['trivial'] else 'earlier-step (downstream)'
        ax.text(0.5, 0.9, f"clamp step {iv['clamp_step']} -- {step_kind}\n"
               f"predicted {iv['predicted']}, actual {iv['actual']}", fontsize=6.5, color='#888', style='italic')
        # small corner verdict badge instead of a large centered block
        vcolor = {'CONFIRMED': '#065f46', 'NOT_CONFIRMED': '#991b1b', 'TRIVIAL': '#6b7280'}[iv['verdict']]
        vbg = {'CONFIRMED': '#ecfdf5', 'NOT_CONFIRMED': '#fef2f2', 'TRIVIAL': '#f3f4f6'}[iv['verdict']]
        ax.text(9.5, 0.5, iv['verdict'], fontsize=6.5, weight='bold', color=vcolor, ha='right',
               bbox=dict(boxstyle='round,pad=0.25', fc=vbg, ec=vcolor, linewidth=0.7))

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.text(0.5, -0.01, f"Source: single_example_{algo}.json (real forward pass + real replacement-model "
            f"intervention recompute on ONE held-out example -- not an aggregate).",
            fontsize=8, color='#888', ha='center', style='italic')
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(f'fig7_{algo}.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved fig7_{algo}.png ({n_panels} panels)')


if __name__ == '__main__':
    for a in ALGOS:
        render_algo(a)
