"""Regenerate the SAME single example fig4_baseline_<algo>.json already used
(identical seed logic from phase1.py -- verified reproducible), and run real
supernode-level interventions on THAT example (not the 15-example aggregate
set, which used different seeds). This matches the paper's own methodology:
Figure 4 and Figure 7 both describe ONE example (the Dallas-capital prompt),
swept across feature-group ablations -- not an aggregate.
"""
import json
import numpy as np, jax, clrs
from graph_engine import build_attribution, to_supernodes, run_intervention_group
from phase1 import SUPERNODE_MAP, strip_negation, build_net, get_example, load_weights, SEED, ALGOS

clrs_run_flags_set = False

def run_algo(algo):
    spec = clrs.SPECS[algo]
    meta = json.load(open(f'../algos/{algo}/meta.json'))
    fl = json.load(open(f'../algos/{algo}/feature_labels.json'))
    label_by_feat = {r['feature']: r for r in fl['rows']}
    smap = SUPERNODE_MAP.get(algo, {})
    W = load_weights(algo, meta)

    net = params = fb = G = None
    for tries in range(6):
        net, params, fb, sampler = build_net(algo, meta, spec, seed=SEED + tries * 1000)
        G = get_example(algo, net, params, fb, meta)
        if G['length'] >= 3:
            break
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
    predictions = {sn: ('decrease' if w > 0 else 'increase') for sn, w in into_target}

    sn_features = {}
    for f, r in label_by_feat.items():
        lab = label_map.get(f)
        if lab:
            sn_features.setdefault(lab['label'], []).append(f)

    interventions = []
    for sn, predicted in predictions.items():
        feats = sn_features.get(sn, [])
        if not feats:
            continue
        for clamp_step, trivial in [(step - 1, False), (step, True)]:
            if clamp_step < 0:
                continue
            res = run_intervention_group(W, G, A, feats, clamp_step, node, -2.0)
            if res is None:
                continue
            actual = 'increase' if res['delta'] > 0 else 'decrease'
            verdict = 'TRIVIAL' if trivial else ('CONFIRMED' if actual == predicted else 'NOT_CONFIRMED')
            interventions.append(dict(supernode=sn, clamp_step=clamp_step, trivial=trivial,
                                      predicted=predicted, actual=actual, verdict=verdict,
                                      base_logit=res['base_logit'], new_logit=res['new_logit'],
                                      delta=res['delta'], mask_changed=res['mask_changed'],
                                      new_class_logits=res['new_class_logits'],
                                      net_weight_into_target=[w for s2, w in into_target if s2 == sn][0]))

    out = dict(algo=algo, target=A['target_name'], target_type=meta['target_type'],
               step=step, node=node, n=G['n'], adj=G['adj'].tolist(), source=G['source'],
               baseline_logit=A['logit'], baseline_class_logits=A['class_logits'],
               supernode_edges_into_target=into_target,
               mean_f1_per_supernode={sn: float(np.mean([label_by_feat[f]['f1'] for f in sn_features.get(sn, [])
                                                        if label_by_feat[f].get('f1') is not None])) if sn_features.get(sn) else None
                                     for sn in predictions},
               interventions=interventions)
    json.dump(out, open(f'single_example_{algo}.json', 'w'), indent=2, default=str)
    print(f"{algo:32s} n_supernodes={len(into_target)}  n_interventions={len(interventions)}")
    return out


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
    print('done')
