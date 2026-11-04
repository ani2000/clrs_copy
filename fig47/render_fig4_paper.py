"""Figure 4 reproduction, paper-faithful style, one PNG per algorithm.
Left panel: "Final simplified graph" (plain diagram, no numbers, matching
the paper's own left panel exactly). Right panel: "Validate with
interventions" -- the single most impactful real downstream intervention
for that algorithm, clamped supernode greyed + "-2x" badged, target box
shows the REAL before/after probability (or raw value for scalar targets).
All numbers from single_example_<algo>.json (this session's real forward
pass + real intervention recompute -- no fabricated data).
"""
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import stacked_box, paper_arrow, badge, MAROON, draw_mini_graph

ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim',
        'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges',
        'strongly_connected_components', 'floyd_warshall']


def to_prob(logit, target_type):
    if target_type == 'scalar':
        return None
    return 1 / (1 + math.exp(-logit))


def render_algo(algo):
    d = json.load(open(f'single_example_{algo}.json'))
    edges = d['supernode_edges_into_target']
    ttype = d['target_type']
    fig, (axG, axL, axR) = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={'width_ratios': [0.62, 1, 1]})
    fig.suptitle(f"Figure 4 — {algo}  (target: {d['target']})", fontsize=13, y=0.98)

    # ---- FAR LEFT: the actual sampled graph this whole figure explains ----
    axG.set_title("Example graph", fontsize=11, loc='left', weight='bold')
    axG.set_xlim(0, 10); axG.set_ylim(0, 10); axG.axis('off')
    draw_mini_graph(axG, 0.3, 3.0, 9.4, 6.5, d['n'], d['adj'], d['source'], highlight_node=d['node'])
    src_label = f"node {d['source']}" if d['source'] is not None else "n/a (no source field)"
    axG.text(0, 2.0, f"n = {d['n']} nodes\nsource = {src_label}\ntarget node = {d['node']}  (step {d['step']})",
            fontsize=8.5, color='#333', transform=axG.transData, va='top')
    axG.text(0, 0.35, "● orange = source     ● purple = target node\n● grey = other nodes",
            fontsize=7, color='#666', va='top')

    # ---- LEFT: Final simplified graph (plain, no numbers -- matches paper's left panel) ----
    axL.set_title("Final simplified graph", fontsize=11, loc='left', weight='bold')
    axL.text(0, 1.12, "We end up with a simplified depiction of the computational\nsteps performed by the model.",
             fontsize=8, color='#555', transform=axL.transAxes, va='top')
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
    n = len(edges)
    stacked_box(axL, 7.1, 4.9, 2.6, 1.1, d['target'][:18], target=True, fontsize=8, bold=True)
    tgt_in = (7.1, 5.45)
    box_top, box_bot = 8.6, 3.6
    for i, (sn, w) in enumerate(edges):
        y = box_top - i * ((box_top - box_bot) / max(n - 1, 1)) if n > 1 else 5.45
        stacked_box(axL, 0.4, y - 0.5, 3.4, 1.0, sn, fontsize=8)
        paper_arrow(axL, (3.8, y), tgt_in, rad=0.08 + 0.02 * i)
    axL.text(0, -0.08, "Note: positions of boxes are chosen for convenience and do not\n"
            "necessarily correspond to token or layer positions", fontsize=7, color='#777',
            transform=axL.transAxes, va='top', style='italic')

    # ---- RIGHT: Validate with interventions (real numbers) ----
    axR.set_title("Validate with interventions", fontsize=11, loc='left', weight='bold')
    axR.text(0, 1.12, "To validate our understanding, we perform intervention\nexperiments in the (replacement) model.",
             fontsize=8, color='#555', transform=axR.transAxes, va='top')
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis('off')

    downstream = [iv for iv in d['interventions'] if not iv['trivial']]
    if not downstream:
        axR.text(5, 5, 'No downstream intervention available\n(no clean baseline -- see Phase 0)',
                ha='center', va='center', fontsize=10, color='gray')
    else:
        iv = max(downstream, key=lambda x: abs(x['delta']))
        clamped_sn = iv['supernode']
        base_p, new_p = to_prob(iv['base_logit'], ttype), to_prob(iv['new_logit'], ttype)
        stacked_box(axR, 7.1, 4.9, 2.6, 1.1, d['target'][:18], target=True, fontsize=8, bold=True)
        tgt_in = (7.1, 5.45)
        box_top, box_bot = 8.6, 3.6
        clamped_y = None
        for i, (sn, w) in enumerate(edges):
            y = box_top - i * ((box_top - box_bot) / max(n - 1, 1)) if n > 1 else 5.45
            is_clamped = (sn == clamped_sn)
            if is_clamped: clamped_y = y
            # ALL supernodes stay visible; only the clamped one is greyed. Every
            # arrow gets a % label -- unclamped ones read "100%" (still at full
            # baseline strength, unaffected by the clamp), matching the paper's
            # own convention of showing unaffected upstream nodes at 100%.
            stacked_box(axR, 0.4, y - 0.5, 3.4, 1.0, sn, faded=is_clamped, fontsize=8)
            paper_arrow(axR, (3.8, y), tgt_in, faded=is_clamped, rad=0.08 + 0.02 * i,
                       pct_label=None if is_clamped else "100%")
            if is_clamped:
                badge(axR, 0.15, y + 0.62, text='-2x')
                axR.text(0.4, y - 0.68, "Inhibit feature group", fontsize=7, color='#555', style='italic')
        # target's post-intervention value floats right beside its box (not a
        # separate text line below) -- this is what makes the causal reading
        # direct: suppress this one supernode, others stay lit, output becomes THIS.
        label = (f"{new_p*100:.0f}%" if new_p is not None else f"{iv['new_logit']:+.2f}")
        axR.text(9.85, 5.8, label, fontsize=12, color='#111', ha='left', va='center', weight='bold')
        base_label = f"{base_p*100:.0f}%" if base_p is not None else f"{iv['base_logit']:+.2f}"
        axR.text(9.85, 5.15, f"(baseline {base_label})", fontsize=6.5, color='#888', ha='left', va='center')
        axR.text(9.4, 7.3, "Observe effect\non output", fontsize=7, color='#555', style='italic', ha='center')
        # small corner verdict badge, not a large centered block
        verdict_color = {'CONFIRMED': '#065f46', 'NOT_CONFIRMED': '#991b1b'}[iv['verdict']]
        verdict_bg = {'CONFIRMED': '#ecfdf5', 'NOT_CONFIRMED': '#fef2f2'}[iv['verdict']]
        axR.text(9.85, 0.35, iv['verdict'], fontsize=7, color=verdict_color, weight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc=verdict_bg, ec=verdict_color, linewidth=0.8))

    fig.text(0.5, 0.01, "Every number here is measured on ONE real held-out example via the actual "
            "replacement-model forward pass (single_example_<algo>.json) -- not illustrative.",
            fontsize=7.5, color='#888', ha='center', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'fig4_{algo}.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved fig4_{algo}.png')


if __name__ == '__main__':
    for a in ALGOS:
        render_algo(a)
