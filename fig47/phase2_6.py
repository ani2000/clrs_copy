"""Phase 2.6: diagnose the accuracy/confirmability inversion (bfs: 99.8% OOD
accuracy but only 55% downstream confirmation; articulation_points/dijkstra:
94.9%/97.7% accuracy, 93%/90% confirmation). Re-derives the SAME 15 examples
per algorithm phase2_aggregate.py used (identical deterministic seeds -- not
a new random draw), this time saving the per-panel/per-feature detail that
was computed transiently but never persisted, so the three diagnostics below
are fully checkable rather than approximated.

NOISE_FLOOR = 0.05 (confirmed): a supernode's edge into target counts as
"live" (Diagnostic 1) if |weight| exceeds 5% of that example's single largest
edge weight into target.
"""
import json
import numpy as np, jax, clrs
from graph_engine import build_attribution, to_supernodes, run_intervention_group
from phase1 import SUPERNODE_MAP, strip_negation, load_weights, HIDDEN, CKPT_DIR, ALGOS
from phase2_aggregate import build_net_and_sampler, examples_from_batch, N_EXAMPLES, FACTOR, BASE_SEED

NOISE_FLOOR = 0.05


def collect_example_detail(algo, meta, label_by_feat, W, smap, G):
    step, node = G['picked_step'], G['picked_node']
    A = build_attribution(W, G, step, node)

    def mapped_label(fid):
        r = label_by_feat.get(fid)
        if not r or not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            return None
        sn = smap.get(strip_negation(r['label']))
        return {'label': sn} if sn else None

    label_map = {f: mapped_label(f) for f in label_by_feat}
    supernode_edges, _ = to_supernodes(A['edges'], A, label_map)
    into_target = [(s, w) for s, d, w in supernode_edges if d == A['target_name'] and s != 'bias']
    predictions = {sn: ('decrease' if w > 0 else 'increase') for sn, w in into_target}  # frozen, per-example

    # Diagnostic 1: breadth -- count of "live" supernodes (>= NOISE_FLOOR * max |w|)
    max_w = max((abs(w) for _, w in into_target), default=0.0)
    live_supernodes = [(s, w) for s, w in into_target if max_w > 0 and abs(w) >= NOISE_FLOOR * max_w]

    sn_features = {}
    for f, r in label_by_feat.items():
        lab = label_map.get(f)
        if lab:
            sn_features.setdefault(lab['label'], []).append(f)

    # raw direct edges feature->target, pre-supernode-merge (for Diagnostic 2)
    direct_to_target = {s: w for s, d, w in A['edges'] if d == A['target_name'] and s.startswith('f')}

    panels = []
    for sn, predicted in predictions.items():
        feats = sn_features.get(sn, [])
        if not feats:
            continue
        for clamp_step, trivial in [(step - 1, False), (step, True)]:
            if clamp_step < 0:
                continue
            active_members = [f for f in feats if f'f{f}@{clamp_step}:{node}' in direct_to_target]
            member_weights = [direct_to_target[f'f{f}@{clamp_step}:{node}'] for f in active_members]
            net_signed = sum(member_weights)
            mean_abs = float(np.mean(np.abs(member_weights))) if member_weights else None
            n_active = len(member_weights)
            cancel_ratio = (abs(net_signed) / (mean_abs * n_active)) if (mean_abs and n_active) else None

            res = run_intervention_group(W, G, A, feats, clamp_step, node, FACTOR)
            if res is None:
                continue
            actual = 'increase' if res['delta'] > 0 else 'decrease'
            verdict = 'TRIVIAL' if trivial else ('CONFIRMED' if actual == predicted else 'NOT_CONFIRMED')
            panels.append(dict(supernode=sn, clamp_step=clamp_step, trivial=trivial, predicted=predicted,
                               actual=actual, verdict=verdict, delta=res['delta'],
                               net_signed_weight=net_signed, mean_abs_member_weight=mean_abs,
                               n_active_members=n_active, member_weights=member_weights,
                               cancellation_ratio=cancel_ratio))
    return dict(target=A['target_name'], step=step, node=node, n_live_supernodes=len(live_supernodes),
               live_supernodes=live_supernodes, n_total_supernodes_into_target=len(into_target), panels=panels)


def run_algo(algo):
    spec = clrs.SPECS[algo]
    meta = json.load(open(f'../algos/{algo}/meta.json'))
    fl = json.load(open(f'../algos/{algo}/feature_labels.json'))
    label_by_feat = {r['feature']: r for r in fl['rows']}
    smap = SUPERNODE_MAP.get(algo, {})
    W = load_weights(algo, meta)

    examples = []
    tries = 0
    while len(examples) < N_EXAMPLES and tries < 8:
        seed = BASE_SEED + (sum(ord(c) for c in algo) * 31) % 1000 + tries * 7919
        net, params, sampler, fb0 = build_net_and_sampler(algo, spec, seed)
        fb = fb0
        for batch_i in range(3):
            if len(examples) >= N_EXAMPLES:
                break
            batch_examples = examples_from_batch(algo, meta, fb, net, params, jax.random.PRNGKey(tries * 10 + batch_i))
            examples.extend(batch_examples)
            fb = next(sampler)
        tries += 1
    examples = examples[:N_EXAMPLES]

    detail = [collect_example_detail(algo, meta, label_by_feat, W, smap, ex) for ex in examples]
    json.dump({'algo': algo, 'noise_floor': NOISE_FLOOR, 'examples': detail},
             open(f'phase2_6_detail_{algo}.json', 'w'), indent=2, default=str)

    # sanity check against phase2agg_<algo>.json (must match exactly -- same data, same interventions)
    agg = json.load(open(f'phase2agg_{algo}.json'))
    ds = [p for ex in detail for p in ex['panels'] if not p['trivial']]
    n_conf = sum(1 for p in ds if p['verdict'] == 'CONFIRMED')
    match = (n_conf == agg['n_downstream_confirmed'] and len(ds) == agg['n_downstream_tests'])
    print(f"{algo:32s} live_supernodes(mean)={np.mean([e['n_live_supernodes'] for e in detail]):.2f}  "
         f"downstream {n_conf}/{len(ds)}  matches_phase2agg={match}")
    if not match:
        print(f"  !! MISMATCH vs phase2agg: agg had {agg['n_downstream_confirmed']}/{agg['n_downstream_tests']}")
    return detail


if __name__ == '__main__':
    import sys
    algos = sys.argv[1:] or ALGOS
    for a in algos:
        try:
            run_algo(a)
        except Exception as e:
            import traceback
            print(f'{a}: FAILED: {e}')
            traceback.print_exc()
    print('\nPhase 2.6 raw detail collection done.')
