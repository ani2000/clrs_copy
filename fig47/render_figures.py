"""Render static Figure-4-style and Figure-7-style PNGs, all 12 algorithms,
from the exact same saved data the viewer_fig47.html page reads
(fig4_viewer_data.json) -- no re-inference, no new numbers.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

DATA = json.load(open('fig4_viewer_data.json'))
ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim',
        'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges',
        'strongly_connected_components', 'floyd_warshall']

# ---------------- FIGURE 4: distilled supernode diagram, all 12 ----------------
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle('Figure 4 — Distilled supernode diagrams (measured, n=1 illustrative example per algorithm)\n'
            'Source: fig4_baseline_<algo>.json  |  arrow width/color = net edge weight into target', fontsize=13, y=0.995)

for ax, algo in zip(axes.flat, ALGOS):
    d = DATA[algo]
    edges = [e for e in d['baseline_supernode_edges'] if e[0] != 'bias']
    cls = 'CLEAN' if d['clean'] else 'SEVERE'
    ax.set_title(f"{algo}  [{cls}]  test_f1={d['test_f1']:.3f}", fontsize=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    if not edges:
        ax.text(5, 5, 'no clean baseline\n(Phase 0)', ha='center', va='center', fontsize=10, color='gray')
        continue
    n = len(edges)
    max_w = max(abs(w) for _, w in edges) or 1e-6
    target_xy = (8.5, 5)
    ax.add_patch(mpatches.FancyBboxPatch((7.3, 4.5), 2.3, 1, boxstyle="round,pad=0.05",
                fc='#fef3c7', ec='#f59e0b', linewidth=1.5))
    tname = d['baseline_target']
    ax.text(8.45, 5, tname[:16] + ('...' if len(tname) > 16 else ''), ha='center', va='center', fontsize=7, weight='bold')
    for i, (sn, w) in enumerate(edges):
        y = 9 - i * (8 / max(n - 1, 1)) if n > 1 else 5
        f1 = d['mean_f1_per_supernode'].get(sn)
        ax.add_patch(mpatches.FancyBboxPatch((0.3, y - 0.4), 3.4, 0.8, boxstyle="round,pad=0.05",
                    fc='#eef2ff', ec='#6366f1', linewidth=1.2))
        label = f"{sn}" + (f" (F1={f1:.2f})" if f1 is not None else "")
        ax.text(2.0, y, label[:26], ha='center', va='center', fontsize=6.5)
        color = '#2563eb' if w > 0 else '#dc2626'
        lw = 0.8 + 4 * abs(w) / max_w
        arrow = FancyArrowPatch((3.7, y), (7.3, 5), arrowstyle='-|>', mutation_scale=10,
                                color=color, linewidth=lw, alpha=0.75, connectionstyle="arc3,rad=0.05")
        ax.add_patch(arrow)
        mid_x, mid_y = 5.5, (y + 5) / 2
        ax.text(mid_x, mid_y + 0.15, f"{w:+.2f}", fontsize=6, color=color, ha='center')

pos_patch = mpatches.Patch(color='#2563eb', label='positive weight')
neg_patch = mpatches.Patch(color='#dc2626', label='negative weight')
fig.legend(handles=[pos_patch, neg_patch], loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig('fig4.png', dpi=130, bbox_inches='tight')
plt.close()
print('saved fig4.png')

# ---------------- FIGURE 7: aggregate intervention confirmation, all 12 ----------------
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle('Figure 7 — Aggregate intervention confirmation (15 independently-sampled graphs per algorithm)\n'
            'Source: phase2agg_<algo>.json + phase3_maskchanged_<algo>.json (per-supernode breakdown)', fontsize=13, y=0.995)

for ax, algo in zip(axes.flat, ALGOS):
    d = DATA[algo]
    cls = 'CLEAN' if d['clean'] else 'SEVERE'
    per_sn = d.get('per_supernode_verdicts', {})
    if not per_sn:
        ax.set_title(f"{algo}  [{cls}]", fontsize=10)
        ax.text(0.5, 0.5, 'no downstream tests\n(no clean baseline)', ha='center', va='center',
               fontsize=10, color='gray', transform=ax.transAxes)
        ax.axis('off')
        continue
    sns = list(per_sn.keys())
    confirmed = [per_sn[s]['CONFIRMED'] for s in sns]
    notconf = [per_sn[s]['NOT_CONFIRMED'] for s in sns]
    trivial = [per_sn[s]['TRIVIAL'] for s in sns]
    totals = [c+n+t for c,n,t in zip(confirmed,notconf,trivial)]
    y_pos = range(len(sns))
    conf_frac = [c/t*100 if t else 0 for c,t in zip(confirmed,totals)]
    notconf_frac = [n/t*100 if t else 0 for n,t in zip(notconf,totals)]
    trivial_frac = [tr/t*100 if t else 0 for tr,t in zip(trivial,totals)]
    ax.barh(y_pos, conf_frac, color='#34d399', label='CONFIRMED')
    ax.barh(y_pos, notconf_frac, left=conf_frac, color='#f87171', label='NOT_CONFIRMED')
    ax.barh(y_pos, trivial_frac, left=[c+n for c,n in zip(conf_frac,notconf_frac)], color='#d1d5db', label='TRIVIAL')
    ax.set_yticks(list(y_pos)); ax.set_yticklabels([s[:16] for s in sns], fontsize=7)
    ax.set_xlim(0, 100)
    headline = f"{d['downstream_confirmed']}/{d['downstream_total']} ({d['downstream_frac']*100:.0f}%)" if d['downstream_total'] else 'N/A'
    ax.set_title(f"{algo}  [{cls}]  downstream: {headline}", fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

handles = [mpatches.Patch(color='#34d399', label='CONFIRMED'), mpatches.Patch(color='#f87171', label='NOT_CONFIRMED'),
          mpatches.Patch(color='#d1d5db', label='TRIVIAL (same-step, not counted as validation)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig('fig7.png', dpi=130, bbox_inches='tight')
plt.close()
print('saved fig7.png')
